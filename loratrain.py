import numpy as np
import evaluate
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Load dataset
paths = {x: f"data/dimension.{x}.csv" for x in ("train", "valid", "test")}
dataset = load_dataset("csv", data_files=paths, delimiter="\t")

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Validation dataset size: {len(dataset['valid'])}")


# Set model and load tokenizer
model_id = "google/mt5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets(
        [dataset["train"], dataset["valid"]]
        ).map(lambda x: tokenizer(x["prompt"], truncation=True),
              batched=True,
              remove_columns=dataset["train"].column_names)
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# take 85 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 99))
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets(
        [dataset["train"], dataset["valid"]]
        ).map(lambda x: tokenizer(x["answer"], truncation=True),
              batched=True,
              remove_columns=dataset["train"].column_names)
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# take 90 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 99))
print(f"Max target length: {max_target_length}")


def preprocess_function(sample, padding="max_length"):
    # add prefix to the input for t5
    inputs = ["Sätz:\r\n" + item for item in sample["prompt"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs,
                             max_length=max_source_length,
                             padding=padding,
                             truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["answer"],
                       max_length=max_target_length,
                       padding=padding,
                       truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels
    # by -100 when we want to ignore padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] \
                    for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function,
                                batched=True,
                                remove_columns=dataset["train"].column_names)
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")
# Define LoRA Config
lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)
# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    preds = [1 if x.lower().endswith("richteg") else 0 for x in preds]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = [1 if x.lower().endswith("richteg") else 0 for x in labels]

    result = metric.compute(predictions=preds, references=labels)
    return result


output_dir = "lora-mt5"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="epoch",
    eval_strategy="epoch",
    predict_with_generate=True,
    generation_max_length=20,
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    compute_metrics=compute_metrics,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# train model
trainer.train()
