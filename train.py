import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from tqdm import tqdm
from config import config

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

class SparseAutoencoder(nn.Module):
    def __init__(self, hidden_size, n_features):
        super().__init__()
        self.encoder = nn.Linear(hidden_size, n_features)
        self.decoder = nn.Linear(n_features, hidden_size)
        self.relu = nn.ReLU()

    def encode(self, x_in):
        x = x_in - self.decoder.bias
        f = self.relu(self.encoder(x))
        return f

    def forward(self, tensors, return_loss=True):
        f = self.encode(tensors)
        x = self.decoder(f)
        if return_loss:
            recon_loss = F.mse_loss(x, tensors)
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

dataset = load_from_disk("tensors")
model = SparseAutoencoder(config['hidden_size'], config['n_features']).to(config['device'])

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
        x, recon_loss, reg_loss = model(**inputs)
        reg_loss = reg_loss * config['lambda_reg']
        loss = recon_loss + reg_loss
        model.normalize_decoder_weights()
        return (loss, x) if return_outputs else loss

    def compute_metrics(self, eval_pred):
        tensors, x = eval_pred
        loss = self.compute_loss(model=self.model, tensors=tensors, return_outputs=False)
        return {"eval_loss": loss.item()}


trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()
