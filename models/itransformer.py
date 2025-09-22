import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
            Xtemp:   (T, B, V)   # 前一半通道是数值特征，后一半通道是变量可见性掩码
            Xtimes:  (T, B, d)    # 用于位置编码的时间特征/索引（需与PositionalEncoding匹配）
        """
        x = x.permute(1, 2, 0)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(1, 2, 0)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)

class iTransformerModel(nn.Module):
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
        super(iTransformerModel, self).__init__()
        self.model_type = 'iTransformer'
        self.temp_dim = temp_dim
        # emb
        self.embedding = DataEmbedding_inverted(max_len, hidden_dim)
        self.static = static_dim > 0
        if self.static:
            self.emb = nn.Linear(static_dim, hidden_dim)

        # encoder
        encoder_layers = TransformerEncoderLayer(hidden_dim, nhead, hidden_dim, dropout)
        self.encoder = TransformerEncoder(
            encoder_layers,
            nlayers,
            norm=torch.nn.LayerNorm(hidden_dim)
        )
        # decoder
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(hidden_dim * temp_dim, n_classes),
        )

    def forward(self, xtemp, xtimes, xstatic, lengths):
        xtemp = xtemp[:, :, :int(xtemp.shape[2] / 2)]
        T,B,V = xtemp.shape
        emb_out = self.embedding(xtemp,xtimes)

        enc_out = self.encoder(emb_out)[:,:V,:]
        dec_out = self.classifier(enc_out)

        return dec_out
