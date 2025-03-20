import logging
import os
import pprint
import datetime

from datasets import Dataset as HFDataset
from transformers import GPT2TokenizerFast
from typing_extensions import List, Tuple
from prefect import flow, task
from prefect.cache_policies import TASK_SOURCE, INPUTS
logging.basicConfig(level=logging.INFO, force=True)

cache_policy = (TASK_SOURCE + INPUTS).configure(key_storage="W:\ML-DL-GENAI\MLOps\Projects\Optimizing LLMs\Math_220\cache")

@task(cache_policy=TASK_SOURCE + INPUTS)
def filter_data(dataset:HFDataset) -> HFDataset:
  """
  Applies filter operation to generate a subset.

  Args:
    dataset (HFDataset): The hugging face dataset to filter.

  Returns:
      None
  """
  dataset = dataset.filter(lambda example, index: index%10==0, with_indices=True)
  return dataset

@task(cache_policy=TASK_SOURCE + INPUTS)
def format_data(dataset:HFDataset, split_size:float, seed:int, to_remove:List[str]) -> Tuple[
    HFDataset, HFDataset, HFDataset]:
    """
    Creates train_test_val splits of the input dataset.

    Args:
        dataset (HFDataset): The dataset to create the splits from.
        split_size (float): The size of the test split.
        seed (int): Reproduction.
        to_remove (List[str]): List of columns to remove.

    Returns:
        (Tuple) The formatted datasets.
    """

    logging.info("Creating the splits...")
    split_1 = dataset.train_test_split(test_size=split_size, seed=seed)
    train_ds = split_1['train']
    temp_ds = split_1['test']

    split_2 = temp_ds.train_test_split(test_size=0.5, seed=seed)
    val_ds = split_2['train']
    test_ds = split_2['test']

    pprint.pprint(f"Length of train dataset : {len(train_ds)}")
    pprint.pprint(f"Length of validation dataset : {len(val_ds)}")
    pprint.pprint(f"Length of test dataset : {len(test_ds)}")

    logging.info("Removing the requested columns..")
    train_ds = train_ds.remove_columns(to_remove)
    val_ds = val_ds.remove_columns(to_remove)
    test_ds = test_ds.remove_columns(to_remove)

    logging.info("Adding the instruction template..")
    INSTRUCTION_TEMPLATE = """
    You are an expert in Mathematics \
    Focus on logical structure and formalisms \
    Respond to the user's query using a multi-step reasoning process \
    Regularly evaluate progress using <reflection> tags \
    Be critical and honest about your reasoning process \
    Assign a quality score between 0.0 and 1.0 using <reward> tags after each reflection \
    Use this to guide your approach: 0.8+: Continue current approach 0.5-0.7: Consider minor adjustments Below 0.5: Seriously consider backtracking and trying a different approach \
    If unsure or if reward score is low, backtrack and try a different approach, explaining your decision within <thinking> tags \
    Show all work explicitly using LaTeX for formal notation and provide detailed proofs \
    All <thinking> tags should be shown to the user, as well as any intermediate steps <steps> \
    """
    train_ds = train_ds.map(lambda x: {'input_text_instruct': INSTRUCTION_TEMPLATE})
    val_ds = val_ds.map(lambda x: {'input_text_instruct': INSTRUCTION_TEMPLATE})

    return train_ds, val_ds, test_ds

@task(cache_policy=TASK_SOURCE + INPUTS)
def format_messages(dataset:HFDataset) -> HFDataset:
  """
  This function creates an appropriate prompt in JSONl format to train the model on.

  Args:
    dataset (HFDataset): The path to the dataset object.

  Returns:
    (HFDataset): The formatted dataset.
  """

  dataset = dataset.map(
      lambda row : {
      "messages" : [
          {'role':'system', 'content': row['input_text_instruct']},
          {'role':'user', 'content': row['problem']},
          {'role':'assistant', 'content': f"Problem_type:{row['problem_type']}, Solution:{row['solution']}"}
      ]
  },
      remove_columns=['problem', 'solution', 'problem_type', 'input_text_instruct'],
      num_proc=os.cpu_count()
      )
  return dataset

@task(cache_policy=TASK_SOURCE + INPUTS)
def apply_chat_template(dataset: HFDataset, tokenizer:GPT2TokenizerFast) -> HFDataset:
  """
  Applies an appropriate chat template for gpt-2.

  Args:
    dataset (HFDataset): Contains the messages to apply the template on.

  Returns:
    (HFDataset) The dataset with the appropriate template.
  """
  tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
  processed_prompt = tokenizer.apply_chat_template(dataset['messages'], tokenize=False, add_generation_prompt=False)
  dataset = dataset.add_column('text', processed_prompt)
  dataset = dataset.remove_columns(['messages'])

  return dataset

@task(cache_policy=TASK_SOURCE + INPUTS)
def to_json(dataset:HFDataset, path:str) -> str:
  """
  Converts the dataset to JSONl format.

  Args:
    dataset (HFDataset): The HFDataset object.
    path (str): The file path to save the dataset at.

  Returns:
    (str) The path to the JSONl dataset.
  """
  dataset = dataset.to_json(orient='records', lines=True, path_or_buf=path)

  return path

@flow(name="data-pipeline-math", log_prints=True)
def data_pipeline(dataset:HFDataset, tokenizer:GPT2TokenizerFast) -> Tuple[
  str, str, str]:
  """
  Workflow to prepare the data for training.

  Args:
      dataset (HFDataset): The HFDataset.
      tokenizer (GPT2TokenizerFast): To tokenize the target labels.

  Returns:
      (Tuple) The file paths.
  """
  logging.info("[Component : create_subset]")
  dataset = filter_data(dataset=dataset)

  logging.info("[Component : format_data]")
  formated_data = format_data(dataset=dataset, split_size=0.2, seed=42,
                            to_remove=['source', 'uuid', 'is_reasoning_complete',
                                       'correctness_math_verify', 'correctness_llama',
                                       'finish_reasons', 'correctness_count', 'messages',
                                       'question_type'])

  # **Arranging the message column in the format expected by SFT**
  logging.info("[Component : format_messages]")
  train, val, test = formated_data
  tuning_data = format_messages(dataset=train)
  validation_data = format_messages(dataset=val)

  # **Applying the appropriate template to prepare the data for training.**
  logging.info("[Component : apply_chat_template]")
  train_dataset = apply_chat_template(dataset=tuning_data, tokenizer=tokenizer)
  validation_dataset = apply_chat_template(dataset=validation_data, tokenizer=tokenizer)

  date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  tuning_data_filename = f"tuning_data_{date}.jsonl"
  validation_data_filename = f"validation_data_{date}.jsonl"
  test_data_filename = f"test_data_{date}.jsonl"

  # **Convert the data to JSONl format.**
  logging.info("[Component : to_json]")
  tuning_data_filename = to_json(dataset=train_dataset, path=tuning_data_filename)
  validation_data_filename = to_json(dataset=validation_dataset, path=validation_data_filename)
  test_data_filename = to_json(dataset=test, path=test_data_filename)

  return tuning_data_filename, validation_data_filename, test_data_filename