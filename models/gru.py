import torch
import torch.nn as nn

from .model_base import TimeSeriesModel
from .positional_encoding import PositionalEncoding


class GRU_Model(TimeSeriesModel):
    def __init__(self, temp_dim, static_dim, hidden_dim, nlayers, dropout, max_len, aggreg, n_classes):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.model_type = 'GRU_Model'
        self.temp_encoder = nn.Linear(temp_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, nlayers, batch_first=True, dropout=dropout)
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
        output = self.gru(x)[0]
        output = output.permute((0, 2, 1))

        # mask out the all-zero rows
        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask2 = mask.long().to(Xtemp.device)
        
        if self.aggreg == 'mean':
            output = torch.sum(output * (1 - mask2), dim=2) / (lengths + 1)
        elif self.aggreg == 'max':
            output, _ = torch.max(output * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=2)

        output = self.dropout(output)
        output = self.mlp(output)
        return output
