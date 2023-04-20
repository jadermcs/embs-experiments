import torch
import torch.nn as nn
import torch.nn.functional as F


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
        output = output.view(-1, self.ntoken)
        return F.log_softmax(output, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)


class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.encoder = Encoder(ntoken, ninp)
        self.decoder = Decoder(ntoken, ninp, nhid, nlayers, dropout)
        self.init_hidden = self.decoder.init_hidden

    def forward(self, input_ids, hidden):
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

    def forward(self, input_ids, hidden):
        mu, var = self.encoder(input_ids)
        eps = torch.randn_like(var)
        z = mu + eps * var
        return self.decoder(z), mu, var
