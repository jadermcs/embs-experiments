import numpy as np
import evaluate
import torch
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load dataset
paths = {x: f"data/dimension.{x}.csv" for x in ("train", "valid", "test")}
dataset = load_dataset("csv", data_files=paths, delimiter="\t")


def convert_labels(example):
    example["label"] = int(example["label"] == "identical")
    # dataset is changing the lema 'null' to None
    if not example["lemma"] and "null" in example["sentence_1"]:
        example["lemma"] = "null"
    try:
        example["text"] = "</s></s>".join([
            example["lemma"], example["sentence_1"], example["sentence_2"]
            ])
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
        "lora-xlmr",
        id2label=id2label,
        label2id=label2id,
        device_map="auto")
model.eval()

# Data collator
metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    result = metric.compute(predictions=preds, references=labels)
    return result


predictions, true_labels = [], []


def prepare_input(x):
    return torch.tensor(x).to(model.device).unsqueeze(0)


# Disable gradient calculation for efficiency
for example in tqdm(dataset["test"]):
    inputs = tokenize(example)
    inputs = {k: prepare_input(v) for k, v in inputs.items() if k != 'label'}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    labels = [example['label']]

    predictions.extend(preds)
    true_labels.extend(labels)

print(compute_metrics((predictions, true_labels)))
