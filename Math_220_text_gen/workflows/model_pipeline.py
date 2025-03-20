import vertexai
import logging

from prefect import flow, task
from prefect.cache_policies import TASK_SOURCE, INPUTS

from transformers import GPT2TokenizerFast, AutoModelForCausalLM, BitsAndBytesConfig
from typing_extensions import List
from datasets import load_dataset
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

cache_policy = (TASK_SOURCE + INPUTS).configure(key_storage="W:\ML-DL-GENAI\MLOps\Projects\Optimizing LLMs\Math_220\cache")

# --------------Configuration Settings------------------

class VertexAIConfig(BaseSettings):
    PROJECT_ID: str
    CREDENTIALS: str

    def init_vertexai(self):
        vertexai.init(
            PROJECT_ID=self.PROJECT_ID,
            CREDENTIALS=self.CREDENTIALS
        )

# ---------------Model Pipeline------------------------

# @task(cache_policy=TASK_SOURCE + INPUTS)
# def vertex_initialization(PROJECT_ID:str, CREDENTIALS:str)->None:
#     """
#     Initializes Vertex AI.
#
#     Args:
#         PROJECT_ID (str): The project id as per gcp.
#         CREDENTIALS (str): Authorization credentials.
#
#     Returns:
#         None
#     """
#     settings = VertexAIConfig(
#         PROJECT_ID=PROJECT_ID,
#         CREDENTIALS=CREDENTIALS
#     )
#     settings.init_vertexai()
#     logging.info("Initialized Vertex AI with project ID: %s", PROJECT_ID)

@task(cache_policy=TASK_SOURCE + INPUTS)
def load_model(model_name:str, attn_implementation:str) -> AutoModelForCausalLM:
    """
    Loads the model as per the specified attention mechanism.

    Args:
        model_name (str) : The name of the model to load.
        attn_implementation (str) : The attention mechanism to incorporate.

    Returns:
        (AutoModelForCausalLM) : The loaded model object.
    """
    config = BitsAndBytesConfig(
        load_in_4bit=True,          # Or load_in_8bit=True for 8-bit quantization
        bnb_4bit_use_dType32=False, # Do not force FP32 weights conversion
        bnb_4bit_compute_dtype=torch.float16  # Use FP16 for computation
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation=attn_implementation, device_map="auto", quantization_config=config)
    logging.info("Loaded model: %s", model_name)
    return model

@task(cache_policy=TASK_SOURCE + INPUTS)
def peft_configuration(model: AutoModelForCausalLM, rank: int, lora_alpha:int,
                       lora_dropout:float, target_modules:List[str]) -> AutoModelForCausalLM:
    """
    Configures peft-lora adapter for the model.

    Args:
        model (AutoModelForCausalLM) : The loaded model.
        rank (int) : For generating rank decomposition metrics.
        lora_alpha (int) : Scaling factor.
        lora_dropout (float) : The dropout effect applied to the lora layers.
        target_modules (List[str]) : The attention modules in the loaded model.

    Returns:
        (AutoModelForCausalLM) : The peft model.
    """
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, peft_config)
    logging.info("Configured PEFT with LoRA parameters: rank=%d, alpha=%d, dropout=%.2f", rank, lora_alpha, lora_dropout)
    return peft_model

@task(cache_policy=TASK_SOURCE + INPUTS)
def model_parameters(model: AutoModelForCausalLM) -> None:
  """
  Calculates the model parameters both training and total.

  Args:
    model (AutoModelForCausalLM): The model to calculate the parameters of.

  Returns:
    (None)
  """
  trainable_params = 0
  total_params = 0

  logging.info("Calculating the model parameters...")
  for params in model.parameters():
    total_params += params.numel()
    if params.requires_grad:
      trainable_params += params.numel()

  logging.info("Trainable parameters: %d", trainable_params)
  logging.info("Total parameters: %d", total_params)

@task(cache_policy=TASK_SOURCE + INPUTS)
def train_model(model: AutoModelForCausalLM, train_dataset:HFDataset, eval_dataset:HFDataset, output_dir: str, tokenizer:GPT2TokenizerFast) -> None:
    """
    Fine tunes the model using sft.

    Args:
        model (AutoModelForCausalLM) : The loaded model[peft].
        train_dataset (str) : The URI to the train dataset in the gcp bucket.
        eval_dataset (str): The URI to the eval dataset in the gcp bucket.
        output_dir (str) : The output path.
        tokenizer (AutoTokenizer) : The tokenizer used in the pre-training of the original model.

    Returns:
        (None)
    """
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            num_train_epochs=10,
            output_dir=output_dir,
            label_names=['labels'],
            do_eval=True,
            eval_strategy="epoch",
            logging_steps=1000,          # Log training loss every 10 steps
            logging_dir="./logs",      # Specify a directory for logs
            save_strategy="epoch",     # Optional: Save checkpoints at epoch boundaries
        ),
        tokenizer=tokenizer
    )
    trainer.train()
    logging.info("Model training completed. Output saved to: %s", output_dir)

@flow(name="model-pipeline-math", log_prints=True)
def model_training_pipeline(model_name:str, attn_implementation:str, rank:int,
                            lora_alpha:int, lora_dropout:float, target_modules:List[str], output_dir:str,
                            train_dataset:str, eval_dataset:str, tokenizer:GPT2TokenizerFast) -> None:
    """
    The model-training pipeline.

    Args:
        model_name (str) : The name of the model to fine tune.
        attn_implementation (str) : The attention mechanism to incorporate.
        rank (int) : For generating rank decomposition metrics.
        lora_alpha (int) : Scaling factor.
        lora_dropout (int) : The dropout effect applied to the lora layers.
        target_modules (List[str]) : The attention modules in the loaded model.
        output_dir (str) : The output path.
        train_dataset (str) : The jsonl file path of the tuning dataset.
        eval_dataset (str) : The jsonl file path of the eval dataset.
        tokenizer (GPT2TokenizerFast) : The tokenizer used in the pre-training of the original model.

    Returns:
        (None)
    """
    # Initialize Vertex AI
    # vertex_initialization(PROJECT_ID=PROJECT_ID, CREDENTIALS=CREDENTIALS)

    # Load model
    model = load_model(model_name=model_name, attn_implementation=attn_implementation)

    # Configure PEFT
    peft_model = peft_configuration(model=model, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                    target_modules=target_modules)

    # Calculate model parameters
    model_parameters(model=peft_model)

    train_dataset = load_dataset('json', data_files=train_dataset)
    eval_dataset = load_dataset('json', data_files=eval_dataset)

    # Train model
    train_model(model=peft_model, train_dataset=train_dataset['train'],eval_dataset=eval_dataset['train'], output_dir=output_dir,
                tokenizer=tokenizer)






