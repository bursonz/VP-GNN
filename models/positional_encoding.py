import torch
from torch import nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        self.max_len = max_len
        self._num_timescales = emb_dim // 2

    def forward(self, Xtime: torch.Tensor):
        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)
        timescales = timescales.reshape((1, 1, -1))
        timescales = torch.from_numpy(timescales).float().to(Xtime.device)
        scaled_time = Xtime / timescales
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x emb_dim
        return pe
