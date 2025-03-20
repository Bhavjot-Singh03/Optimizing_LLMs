import pprint
from transformers import AutoModelForCausalLM, GPT2TokenizerFast
from datasets import load_dataset

# Loading the fine tuned model
adapter_path = r'C:\Users\singh\PycharmProjects\PythonProject\GPT2-SFT\GPT-Math-2.3M'
model_name = 'openai-community/gpt2'
tokenizer = GPT2TokenizerFast.from_pretrained('openai-community/gpt2')

model = AutoModelForCausalLM.from_pretrained(model_name)
model.load_adapter(adapter_path)

test_ds = load_dataset('json', data_files='Dataset/test_data_2025-03-18-09-08-21.jsonl')
sample = test_ds['train']['problem'][1]

inputs = tokenizer(sample, return_tensors="pt")
output = model.generate(**inputs)
pprint.pprint(f'question: {sample}, output:{tokenizer.decode(output[0], skip_special_tokens=True)}')


