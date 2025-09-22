import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .positional_encoding import PositionalEncoding


class TransformerModelV2(nn.Module):
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
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

    def __init__(self, temp_dim, pe_dim, static_dim, nhead, hidden_dim, nlayers, dropout, max_len, aggreg, n_classes):
        super(TransformerModelV2, self).__init__()
        self.model_type = 'TransformerModelV2'
        d_model = (temp_dim * 2 + nhead - 1) // nhead * nhead
        pe_dim = d_model - temp_dim * 2
        self.temp_encoder = nn.Linear(temp_dim, temp_dim * 2)
        self.pos_encoder = PositionalEncoding(pe_dim, max_len)
        
        encoder_layers = TransformerEncoderLayer(temp_dim * 2 + pe_dim, nhead, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.static = static_dim > 0
        if self.static:
            self.emb = nn.Linear(static_dim, static_dim * 2)

        if self.static == False:
            final_dim = temp_dim * 2 + pe_dim
        else:
            final_dim = temp_dim * 2 + pe_dim + static_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, n_classes),
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

        x = torch.cat([Xtemp, pe], axis=2)

        # mask out the all-zero rows
        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).to(Xtemp.device)
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        mask2 = mask.permute(1, 0).unsqueeze(2).long()
        
        if self.aggreg == 'mean':
            output = torch.sum(output * (1 - mask2), dim=0) / (lengths + 1)
        elif self.aggreg == 'max':
            output, _ = torch.max(output * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=0)

        if Xstatic is not None:
            emb = self.emb(Xstatic)
            output = torch.cat([output, emb], dim=1)

        output = self.dropout(output)
        output = self.mlp(output)
        return output
