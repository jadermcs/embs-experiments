import torch
import argparse

from itertools import chain
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from variational_gpt import GPT2ModelWithVI
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config


parser = argparse.ArgumentParser(description='PyTorch RNN')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--emb_size', type=int, default=768,
                    help='size of word embeddings')
parser.add_argument('--hidden_size', type=int, default=384,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--token_length', type=int, default=128,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--vae', action='store_true', default=False,
                    help='use vae')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument("--learning_rate", type=float, default=2e-5,
                    help="Initial learning rate to use.")
parser.add_argument("--weight_decay", type=float, default=0.1,
                    help="Weight decay to use.")
parser.add_argument("--max_steps", type=int, default=100,
                    help="Total number of training epochs to perform.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of updates steps to accumulate for a backward/update pass.")
parser.add_argument("--num_warmup_steps", type=int, default=1,
                    help="Number of steps for the warmup in the lr scheduler.")
parser.add_argument('--path', type=str, default="models/",
                    help='path to save trained models')
args = parser.parse_args()

if args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config()
if not args.vae:
    model = GPT2LMHeadModel(config)
else:
    model = GPT2ModelWithVI(config)


def kl_criterion(mu, logsigma):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = torch.log(
            -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp()))
    return KLD


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        mu = outputs.mu
        logsigma = outputs.logsigma
        if args.vae:
            loss += kl_criterion(mu, logsigma)
        return (loss, outputs) if return_outputs else loss


datasets = load_dataset("wikitext", "wikitext-103-v1")


def tokenizer_function(examples):
    output = tokenizer(examples["text"])
    return output


column_names = datasets["train"].column_names

print("Tokenizing data.")
datasets = datasets.map(
    tokenizer_function,
    batched=True,
    num_proc=8,
    remove_columns=column_names,
)


def group_texts(examples):
    concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= args.token_length:
        total_length = (total_length // args.token_length) * args.token_length
    res = {
        k: [t[i: i + args.token_length] for i in range(0, total_length,
            args.token_length)]
        for k, t in concatenated_examples.items()
    }
    res['labels'] = res['input_ids'].copy()
    return res


print("Grouping data.")
datasets = datasets.map(
    group_texts,
    batched=True,
    num_proc=8,
)

print("Training.")

if args.vae:
    exp_name = "rnn_vae"
else:
    exp_name = "rnn"

training_args = TrainingArguments(
        exp_name,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.num_warmup_steps,
        report_to="mlflow",
        load_best_model_at_end=True,
        save_total_limit=5,
        # fp16=True,
)

trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()
trainer.save_model()
