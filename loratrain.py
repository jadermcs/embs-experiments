import numpy as np
from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset


model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
print(model)
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

paths = {x: f"data/dimension.{x}.csv" for x in ("train", "valid", "test")}
dataset = load_dataset("csv", data_files=paths, delimiter="\t")


def preprocess_function(example):
    label = "Identesch" if example['label'] == "identical" else "Ënnerschiddlech"
    item = {
            "prompt": f"{example['sentence1']}\r\n{example['sentence2']}",
            "answer": f"Ass eng Bedeitung vu '{example['lemma']}' identesch oder ënnerschiddlech? {label}",
            }
    item["len_prompt"] = len(tokenizer(item["prompt"]).input_ids)
    item["len_answer"] = len(tokenizer(item["answer"]).input_ids)
    return item


dataset = dataset.map(
    preprocess_function,
    num_proc=4,
    remove_columns=dataset["train"].column_names,
)
print(dataset)

max_source = int(np.percentile(dataset["train"]["len_prompt"], 99))
print(max_source)
max_target = int(np.percentile(dataset["train"]["len_answer"], 99))
print(max_target)


def tokenize_function(example):
    item = tokenizer(
            example["prompt"],
            max_length=max_source,
            padding="max_length", truncation=True)
    labels = tokenizer(
            text_target=example["answer"],
            max_length=max_target,
            padding="max_length", truncation=True).input_ids
    labels = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels
        ]

    item["labels"] = labels
    return item


dataset = dataset.map(
    tokenize_function,
    num_proc=4,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
print(dataset)
print(dataset["train"][0])

peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "v"],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)


training_args = Seq2SeqTrainingArguments(
    output_dir="LIST/ltz-mt5-lora",
    learning_rate=1e-3,
    auto_find_batch_size=True,
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
    eval_dataset=dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
