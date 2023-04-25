import torch
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Model


class VGPT2Embeddings(nn.Module):
    def __init__(self, config):
        super(VGPT2Embeddings).__init__()
        self.word_embeddings = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id)
        self.dev_word_embeddings = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        mu = self.word_embeddings(input_ids)
        sigma = torch.exp(.5*self.dev_word_embeddings(input_ids))
        eps = torch.randn_like(sigma)
        embeddings = mu + eps * sigma

        return embeddings, mu, sigma


class VariationalGPT(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = VGPT2Embeddings(config)
        self.decoder = GPT2LMHeadModel(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        embs, mu, sigma = self.encoder(input_ids)
        output = self.decoder(inputs_embeds=embs)
        return output, mu, sigma


class Encoder(nn.Module):

    def __init__(self, ntoken, ninp, conditional=False):
        super(Encoder, self).__init__()
        self.ntoken = ntoken
        self.embed = nn.Embedding(ntoken, ninp)
        if conditional:
            self.embed_dev = nn.Embedding(ntoken, ninp)
        self.conditional = conditional
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embed.weight, -initrange, initrange)
        if self.conditional:
            nn.init.uniform_(self.embed_dev.weight, -initrange, initrange)

    def forward(self, input_ids):
        emb = self.embed(input_ids)
        if self.conditional:
            embed_dev = self.embed_dev(input_ids)
            embed_dev = torch.exp(embed_dev)
            return emb, embed_dev
        return emb


class Decoder(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity='tanh',
                          dropout=dropout)
        self.mlp = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.ntoken = ntoken
        self.nlayers = nlayers
        self.nhid = nhid

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.mlp.weight, -initrange, initrange)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, input_emb, hidden):
        emb = self.drop(input_emb)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        output = self.mlp(output)
        return F.log_softmax(output, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)


class RWKVDecoder(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RWKVDecoder, self).__init__()
        pass

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    def channel_mixing(self, x, state, i, time_mix_k, time_mix_r, kw, vw, rw):
        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))  # square relu, primer paper
        return r * (vw @ k)

    def time_mixing(self, x, state, i, time_mix_k, time_mix_v, time_mix_r,
                    time_first, time_decay, kw, vw, rw, ow):
        xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
        state[5*i+1] = x
        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + k
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b
        ww = pp + time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = qq
        return ow @ (r * wkv)

    def forward(self, token, state=None):
        if state is None:
            state = torch.zeros(self.args.n_layer * 5, self.args.n_embd)
            for i in range(self.args.n_layer):
                state[5*i+4] = -1e30  # -infinity
        x = self.w.emb.weight[token]
        x = self.layer_norm(x, self.w.blocks[0].ln0)
        for i in range(self.args.n_layer):
            att = self.w.blocks[i].att
            x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i, 
                att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay, 
                att.key.weight, att.value.weight, att.receptance.weight, att.output.weight)
            ffn = self.w.blocks[i].ffn
            x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i, 
                ffn.time_mix_k, ffn.time_mix_r, 
                ffn.key.weight, ffn.value.weight, ffn.receptance.weight)

        x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
        return x.float(), state


class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.encoder = Encoder(ntoken, ninp)
        self.decoder = Decoder(ntoken, ninp, nhid, nlayers, dropout)
        self.init_hidden = self.decoder.init_hidden

    def forward(self, input_ids, hidden, attention_mask=None, labels=None):
        output = self.encoder(input_ids)
        output, hidden = self.decoder(output, hidden)
        return output, hidden


class BayesianRNN(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.encoder = Encoder(ntoken, ninp, conditional=True)
        self.decoder = Decoder(ntoken, ninp, nhid, nlayers, dropout)
        self.init_hidden = self.decoder.init_hidden
        self.z_dim = ninp

    def forward(self, input_ids, hidden, attention_mask=None, labels=None):
        mu, logvar = self.encoder(input_ids)
        eps = torch.randn_like(logvar)
        z = mu + eps * torch.exp(.5*logvar)
        output, hidden = self.decoder(z, hidden)
        return output, hidden, mu, logvar
