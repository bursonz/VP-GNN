import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .model_base import TimeSeriesModel
from .positional_encoding import PositionalEncoding

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.dk = hidden_dim // num_heads
        self.Wq = nn.Parameter(torch.empty((hidden_dim, hidden_dim)), requires_grad=True)
        self.Wk = nn.Parameter(torch.empty((hidden_dim, hidden_dim)), requires_grad=True)
        self.Wv = nn.Parameter(torch.empty((hidden_dim, hidden_dim)), requires_grad=True)
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)
        self.Wo = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, x, mask):
        # x: bsz, T, d
        bsz, T, d = x.size()
        # device = x.device
        q = torch.matmul(x, self.Wq).view(bsz, T, self.num_heads, self.dk)
        q = q / np.sqrt(self.dk)
        k = torch.matmul(x, self.Wk).view(bsz, T, self.num_heads, self.dk)
        v = torch.matmul(x, self.Wv).view(bsz, T, self.num_heads, self.dk)
        A = torch.einsum('bthd,blhd->bhtl', q, k) + mask  # bsz, h, T, T
        A = F.softmax(A, dim=-1)
        A = F.dropout(A, self.dropout)
        x = torch.einsum('bhtl,bthd->bhtd', A, v)
        x = self.Wo(x.reshape((bsz, T, d)))
        return x


class FeedForward(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim*2, hidden_dim, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForward(hidden_dim)
        self.norm_mha = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, x, mask):
        x2 = F.dropout(self.mha(x, mask), self.dropout, self.training)
        x = self.norm_mha(x+x2)
        x2 = F.dropout(self.ffn(x), self.dropout, self.training)
        x = self.norm_ffn(x+x2)
        return x


class DenseInterpolation(nn.Module):
    def __init__(self, M, seq_len):
        super().__init__()
        cols = torch.arange(M).reshape((1, M)) / M
        rows = torch.arange(seq_len).reshape((seq_len, 1)) / seq_len
        self.W = (1-torch.abs(rows-cols))**2
        self.W = nn.Parameter(self.W, requires_grad=False)

    def forward(self, x):
        bsz = x.size()[0]
        x = torch.matmul(x.transpose(1, 2), self.W)  # bsz,V,M
        return x.reshape((bsz, -1))


class SAND_Model(TimeSeriesModel):
    """ SAND model with context embedding, aggregation
    Inputs:
        temp_dim: dimension of temporal features
        pe_dim: dimension of positional encoding
        static_dim: dimension of positional encoding
        nhead = number of heads in multihead-attention
        hidden_dim: dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length
        n_classes = number of classes
    """

    def __init__(self, temp_dim, static_dim, hidden_dim, nheads, nlayers, R, M, dropout, max_len, aggreg, n_classes):
        super(SAND_Model, self).__init__()
        self.model_type = 'SAND_Model'
        self.temp_encoder = nn.Linear(temp_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)

        self.static = static_dim > 0
        if self.static:
            self.emb = nn.Linear(static_dim, hidden_dim)

        indices = torch.arange(max_len)
        self.mask = torch.logical_and(indices[None, :] <= indices[:, None],
                                      indices[None, :] >= indices[:, None]-R).float()
        self.mask = (1-self.mask)*torch.finfo(self.mask.dtype).min
        self.mask = nn.Parameter(self.mask, requires_grad=False)

        self.transformer_list = nn.ModuleList([TransformerBlock(hidden_dim, nheads, dropout)
                                               for i in range(nlayers)])

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )
        self.aggreg = aggreg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        self.temp_encoder.reset_parameters()
        if self.static:
            self.emb.reset_parameters()
        self.mlp._modules['0'].reset_parameters()
        self.mlp._modules['2'].reset_parameters()

    def forward(self, Xtemp, Xtimes, Xstatic, lengths):
        maxlen, _, _ = Xtemp.shape
        Xtemp = Xtemp[:, :, :int(Xtemp.shape[2] / 2)]
        Xtemp = self.temp_encoder(Xtemp)
        pe = self.pos_encoder(Xtimes)

        if Xstatic is not None:
            emb = self.emb(Xstatic)
            emb.unsqueeze(0)
            x = Xtemp + pe + emb
        else:
            x = Xtemp + pe

        x = x.permute((1, 0, 2))
        # bcs, seq_len, hidden_dim
        for layer in self.transformer_list:
            x = layer(x, self.mask)

        # output = self.dense_interpolation(x)
        output = x.permute((0, 2, 1))

        # mask out the all-zero rows
        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask2 = mask.long().to(Xtemp.device)

        if self.aggreg == 'mean':
            output = torch.sum(output * (1 - mask2), dim=2) / (lengths + 1)
        elif self.aggreg == 'max':
            output, _ = torch.max(output * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=2)

        output = self.mlp(output)
        return output
