import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .positional_encoding import PositionalEncoding


class TransformerModel(nn.Module):
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

    def __init__(self, temp_dim, static_dim, hidden_dim, nhead, nlayers, dropout, max_len, aggreg, n_classes):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.temp_encoder = nn.Linear(temp_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        
        encoder_layers = TransformerEncoderLayer(hidden_dim, nhead, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.static = static_dim > 0
        if self.static:
            self.emb = nn.Linear(static_dim, hidden_dim)

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
        # self.transformer_encoder.reset_parameters()
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

        # mask out the all-zero rows
        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).to(Xtemp.device)
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        mask2 = mask.permute(1, 0).unsqueeze(2).long()
        
        if self.aggreg == 'mean':
            output = torch.sum(output * (1 - mask2), dim=0) / (lengths + 1)
        elif self.aggreg == 'max':
            output, _ = torch.max(output * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=0)

        output = self.dropout(output)
        output = self.mlp(output)
        return output
