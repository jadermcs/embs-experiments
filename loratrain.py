import numpy as np
from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset


model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
print(model)
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
new_tokens = ["<sep>"]

# check if the tokens are already in the vocabulary
new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

# add the tokens to the tokenizer vocabulary
tokenizer.add_tokens(list(new_tokens))

# add new, random embeddings for the new tokens
model.resize_token_embeddings(len(tokenizer))

paths = {x: f"data/dimension.{x}.csv" for x in ("train", "valid", "test")}
dataset = load_dataset("csv", data_files=paths, delimiter="\t")


def preprocess_function(example):
    label = "Identesch" if example['label'] == "identical" else "Ënnerschiddlech"
    item = {
            "prompt": f"{example['sentence1']}<sep>{example['sentence2']}",
            "answer": f"Ass eng Bedeitung vu '{example['lemma']}' identesch oder ënnerschiddlech? {label}",
            }
    item["len_prompt"] = len(tokenizer(item["prompt"]))
    item["len_answer"] = len(tokenizer(item["answer"]))
    return item


dataset = dataset.map(
    preprocess_function,
    num_proc=4,
)
print(dataset)

max_source = int(np.percentile(dataset["train"]["len_prompt"], 90))
max_target = int(np.percentile(dataset["train"]["len_answer"], 90))


def tokenize_function(example):
    item = tokenizer(example["prompt"], max_length=max_source, padding="max_length", truncation=True)
    labels = tokenizer(text_target=example["answer"], max_length=max_target, padding="max_length", truncation=True)
    item["labels"] = labels["input_ids"]
    return item


dataset = dataset.map(
    tokenize_function,
    num_proc=4,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
print(dataset)

tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        target_modules='all-linear',
        r=8,
        lora_alpha=32,
        lora_dropout=0.1)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token,
    pad_to_multiple_of=8
)


training_args = Seq2SeqTrainingArguments(
    output_dir="LIST/ltz-phi-lora",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
