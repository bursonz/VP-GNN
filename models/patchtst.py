import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class PatchTST(nn.Module):
    """ Transformer model with context embedding, aggregation
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

    def __init__(self, temp_dim, static_dim, hidden_dim, nhead, nlayers, dropout, max_len, aggreg, n_classes, patch_len=16, stride=8):
        super(PatchTST, self).__init__()
        self.model_type = 'PatchTST'
        padding=stride
        # patching and embedding
        self.embedding = PatchEmbedding(
            hidden_dim, patch_len, stride, padding, dropout)

        self.static = static_dim > 0
        if self.static:
            self.emb = nn.Linear(static_dim, hidden_dim)

        # encoder
        encoder_layers = TransformerEncoderLayer(hidden_dim, nhead, hidden_dim, dropout)
        self.encoder = TransformerEncoder(
            encoder_layers,
            nlayers,
            norm=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(hidden_dim), Transpose(1,2)),
        )

        # decoder
        self.head_nf = hidden_dim * \
                       int((max_len - patch_len) / stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(self.head_nf * temp_dim, n_classes)

    def forward(self, xtemp, xtimes, xstatic, lengths):
        xtemp = xtemp[:, :, :int(xtemp.shape[2] / 2)]
        #[T,B,V]
        xtemp = xtemp.permute(1, 0, 2)
        #[B,T,V]
        x_enc = xtemp.permute(0,2,1)
        enc_out, n_vars = self.embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)

        return output
