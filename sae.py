import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm


from itertools import chain
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


class SparseAutoencoder(nn.Module):
    def __init__(self, n_features, model):
        super().__init__()
        self.model = model
        self.encoder = nn.Linear(model.config.dim, n_features)
        self.decoder = nn.Linear(n_features, model.config.dim)
        self.relu = nn.ReLU()

    def encode(self, x_in):
        x = x_in - self.decoder.bias
        f = self.relu(self.encoder(x))
        return f

    def forward(self, input_data, return_outputs=False, compute_loss=False):
        output = self.model(**input_data, output_hidden_states=True)
        x_in = output.hidden_states[-1]
        f = self.encode(x_in)
        x = self.decoder(f)
        if compute_loss:
            recon_loss = F.mse_loss(x, x_in)
            reg_loss = f.abs().sum(dim=-1).mean()
        else:
            recon_loss = None
            reg_loss = None
        if return_outputs:
            return x, recon_loss, reg_loss, output
        return x, recon_loss, reg_loss, None

    def normalize_decoder_weights(self):
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, p=2, dim=1)


# In[57]:


config = {
    'exp_name': 'sao-semantics',
    'device': 'cpu',
    'token_length': 512,
    'n_features': 2048,
    'learning_rate':3e-5,
    'lambda_reg': 5,
    'batch_size': 64,
    'accumulation_steps': 1,
    'weight_decay': 0.1,
    'warmup_steps': 200,
    'data': 'monology/pile-uncopyrighted',
    'data': ('wikitext', 'wikitext-103-v1'),
    'base_model': 'distilbert/distilbert-base-uncased',
}


# In[58]:


base_model = AutoModelForMaskedLM.from_pretrained(config['base_model'])
tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
model = SparseAutoencoder(
    config['n_features'],
    base_model
).to(config['device'])


# In[59]:


input_data = tokenizer("ola", return_tensors="pt")
output = model(input_data)
output


# In[7]:


dataset = load_dataset(*config['data'])


def tokenizer_function(examples):
    output = tokenizer(examples["text"])
    return output


# In[9]:


column_names = dataset["train"].column_names


# In[10]:


dataset = dataset.map(
    tokenizer_function,
    batched=True,
    num_proc=8,
    remove_columns=column_names,
)


# In[11]:


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


# In[12]:


dataset = dataset.map(
    group_texts,
    batched=True,
    num_proc=8,
)


# In[52]:


training_args = TrainingArguments(
        config['exp_name'],
        evaluation_strategy="steps",
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['accumulation_steps'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
        report_to="wandb",
        load_best_model_at_end=True,
        save_total_limit=5,
        remove_unused_columns=False,
        fp16=True,
)


# In[53]:


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        _, recon_loss, reg_loss, outputs = model(inputs, return_outputs=return_outputs, compute_loss=True)
        reg_loss = reg_loss * config['lambda_reg']
        loss = recon_loss + reg_loss
        model.normalize_decoder_weights()
        return (loss, outputs) if return_outputs else loss


# In[60]:


trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)


# In[61]:


trainer.train()
trainer.save_model()
