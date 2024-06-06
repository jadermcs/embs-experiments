import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm


from itertools import chain
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

config = {
    'exp_name': 'sao-semantics',
    'device': 'cpu',
    'token_length': 512,
    'n_features': 2**14,
    'epochs': 1,
    'learning_rate': 1e-3,
    'lambda_reg': 5,
    'batch_size': 16,
    'accumulation_steps': 4,
    'warmup_steps': 200,
    'data': ('monology/pile-uncopyrighted',),
    'data': ('wikitext', 'wikitext-103-v1'),
    'base_model': 'microsoft/deberta-v3-base',
}

class SparseAutoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Linear(model.config.hidden_size, n_features)
        self.decoder = nn.Linear(n_features, model.config.hidden_size)
        self.relu = nn.ReLU()

    def encode(self, x_in):
        x = x_in - self.decoder.bias
        f = self.relu(self.encoder(x))
        return f

    def forward(self, x_in, compute_loss=False):
        f = self.encode(x_in)
        x = self.decoder(f)
        if compute_loss:
            recon_loss = F.mse_loss(x, x_in)
            reg_loss = f.abs().sum(dim=-1).mean()
        else:
            recon_loss = None
            reg_loss = None
        return x, recon_loss, reg_loss

    def normalize_decoder_weights(self):
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, p=2, dim=1)

    def save_pretrained(self, path):
        pass


base_model = AutoModelForMaskedLM.from_pretrained(config['base_model'],
                torch_dtype=torch.bfloat16)
base_model.eval()
tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
model = SparseAutoencoder(config['n_features']).to(config['device'])


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


def embeddings(inputs):
    with torch.no_grad():
        output = base_model(**inputs, output_hidden_states=True)
    return output.hidden_states[-1]


dataset = dataset.map(
    embeddings,
    batched=True,
)

del base_model

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
    num_proc=32,
)


training_args = TrainingArguments(
        config['exp_name'],
        num_train_epochs=config['epochs'],
        eval_strategy="steps",
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['accumulation_steps'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        report_to="wandb",
        load_best_model_at_end=True,
        save_total_limit=5,
        remove_unused_columns=False,
        fp16=True,
)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        _, recon_loss, reg_loss, outputs = model(**inputs, return_outputs=return_outputs, compute_loss=True)
        reg_loss = reg_loss * config['lambda_reg']
        loss = recon_loss + reg_loss
        model.normalize_decoder_weights()
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()
torch.save(model.state_dict(), config['exp_name'])
