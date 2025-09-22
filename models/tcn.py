import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .model_base import TimeSeriesModel
from .positional_encoding import PositionalEncoding

# Reference: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    

class TemporalConvNet(nn.Module):
    def __init__(self, hidden_dim, hidden_dim_list, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(hidden_dim_list)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = hidden_dim if i == 0 else hidden_dim_list[i-1]
            out_channels = hidden_dim_list[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalConvModel(TimeSeriesModel):
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

    def __init__(self, temp_dim, static_dim, hidden_dim, nlayers, kernel_size, dropout, max_len, aggreg, n_classes):
        super(TemporalConvModel, self).__init__()
        self.model_type = 'TemporalConvModel'
        self.temp_encoder = nn.Linear(temp_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        
        self.static = static_dim > 0
        if self.static:
            self.emb = nn.Linear(static_dim, hidden_dim)

        self.net = TemporalConvNet(hidden_dim, [hidden_dim]*nlayers, kernel_size, dropout)

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

        x = x.permute((1, 2, 0))
        output = self.net(x)

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
