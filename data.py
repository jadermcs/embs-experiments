import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from config import config


from itertools import chain
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

base_model = AutoModelForMaskedLM.from_pretrained(config['base_model']).to(config['device'])
base_model.eval()
tokenizer = AutoTokenizer.from_pretrained(config['base_model'])

dataset = load_dataset(*config['data'], num_proc=32)


def tokenizer_function(examples):
    output = tokenizer(examples["text"])
    return output


column_names = dataset["train"].column_names

dataset = dataset.map(
    tokenizer_function,
    batched=True,
    num_proc=32,
    remove_columns=column_names,
)


dataset = dataset["train"].train_test_split(test_size=0.001, shuffle=True)


def group_texts(examples):
    concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= config['token_length']:
        total_length = (total_length // config['token_length']) * config['token_length']
    res = {
        k: [t[i: i + config['token_length']] for i in range(0, total_length,
            config['token_length'])]
        for k, t in concatenated_examples.items()
    }
    res['labels'] = res['input_ids'].copy()
    return res


dataset = dataset.map(
    group_texts,
    batched=True,
    num_proc=64,
)

#dataset["train"] = dataset["train"].shuffle(seed=42).select(range(100_000))
dataset = dataset.with_format("torch", device=config['device'])

def embedding(inputs):
    with torch.no_grad():
        output = base_model(**inputs, output_hidden_states=True)
    output = output.hidden_states[-1]
    return {"tensors": output}

column_names = dataset["train"].column_names
dataset = dataset.map(
    embedding,
    batched=True,
    batch_size=32,
    remove_columns=column_names,
)
dataset.set_format(type="torch", columns=list(dataset["train"].features.keys()), device="cpu")

print(dataset)
dataset.save_to_disk("tensors")
