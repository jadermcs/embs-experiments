import numpy as np
import evaluate
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments

# Load dataset
paths = {x: f"data/dimension.{x}.csv" for x in ("train", "valid", "test")}
dataset = load_dataset("csv", data_files=paths, delimiter="\t")


def convert_labels(example):
    example["label"] = int(example["label"] == "identical")
    # dataset is changing the lema 'null' to None
    if not example["lemma"]:
        example["lemma"] = "null"
    try:
        example["text"] = example["sentence_1"] + "</s></s>" + \
            example["sentence_2"] + "</s/s>" + example["lemma"]
    except:
        print(example)
        exit()
    return example


dataset = dataset.map(
        convert_labels,
        remove_columns=["prompt", "answer", "sentence_1", "sentence_2", "lemma"])

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Validation dataset size: {len(dataset['valid'])}")


# Set model and load tokenizer
model_id = "FacebookAI/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def tokenize(example):
    return tokenizer(example["text"], truncation=True)


# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets(
        [dataset["train"], dataset["valid"]]
        ).map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# take 85 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 99))
print(f"Max source length: {max_source_length}")

tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names.remove("label"))
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")


id2label = {0: "different", 1: "identical"}
label2id = {"different": 0, "identical": 1}
# load model from the hub
model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        device_map="auto")
# Define LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type=TaskType.SEQ_CLS
)
# add LoRA adaptor
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)

    result = metric.compute(predictions=preds, references=labels)
    return result


output_dir = "lora-xlmr"

# Define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-5,
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="mlflow",
)

# Create Trainer instance
trainer = Trainer(
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
