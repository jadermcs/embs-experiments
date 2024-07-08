import os
import re
import pandas as pd
from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
from huggingface_hub import snapshot_download


model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
print(model)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
# PREPARE DATA

folder = snapshot_download(
    "cis-lmu/glotcc-v1",
    repo_type="dataset",
    local_dir="./glotcc-v1/",
    allow_patterns="v1.0/ltz-Latn/*"
)

# Load the dataset from a Parquet file
# Replace the file path with the path to the desired language's Parquet file

data = []
for path, subdirs, files in os.walk("../LOD-Corpus/Texter"):
    for name in files:
        fpath = os.path.join(path, name)
        with open(fpath) as fin:
            content = fin.read()
        data.append({"source": fpath, "content": content})

data = pd.DataFrame(data)

CLEANR = re.compile('<.*?>')


def cleanhtml(raw_html):
  return re.sub(CLEANR, '', raw_html)


data.content = data.content.apply(cleanhtml)
dataset = pd.read_parquet('./glotcc-v1/v1.0/ltz-Latn/ltz-Latn_0.parquet')
dataset = pd.concat([data, dataset], ignore_index=True)
dataset = Dataset.from_pandas(dataset[["content"]])


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["content"]])


dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset.column_names,
)

block_size = 1024


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported
    # it instead of this drop, you can customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


dataset = dataset.map(group_texts, batched=True, num_proc=4)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        target_modules='all-linear',
        r=8,
        lora_alpha=32,
        lora_dropout=0.1)

model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())


training_args = TrainingArguments(
    output_dir="LIST/ltz-phi-lora",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
