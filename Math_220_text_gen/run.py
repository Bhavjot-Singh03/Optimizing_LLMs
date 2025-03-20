import logging
import pprint

from workflows.model_pipeline import model_training_pipeline
from workflows.data_pipeline import data_pipeline

from transformers import GPT2TokenizerFast
from datasets import load_dataset

from prefect.utilities.annotations import quote
logging.basicConfig(level=logging.INFO, force=True)

# from google.auth import default
# _, CREDENTIALS = default()
# REGION= 'us-central1'
# BUCKET_URI = 'gs://starborn-1/Optimizing_LLMS/open-r1'

tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

Math_220 = load_dataset("open-r1/OpenR1-Math-220k", "default")

if __name__ == "__main__":
    tuning_data_filename, validation_data_filename, test_data_filename=data_pipeline(dataset=quote(Math_220['train']),
                                                                                     tokenizer=tokenizer)
    pprint.pprint(f"{tuning_data_filename}, {validation_data_filename}, {test_data_filename}")
    model_training_pipeline(
        model_name='openai-community/gpt2',
        attn_implementation='sdpa',
        rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['c_attn', 'c_proj', 'c_fc'],
        output_dir='GPT2-SFT/GPT-Math-2.3M',
        train_dataset=tuning_data_filename,
        eval_dataset=validation_data_filename,
        tokenizer=tokenizer
    )