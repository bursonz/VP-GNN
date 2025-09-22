from torch_geometric.nn import GNNFF

from models.PT_SGMP import PT_SGMP
from models.model_base import TimeSeriesModel
from models.svmpt_seg import SVMPT_Seg_Model


def make_model(dataset, model_name) -> TimeSeriesModel:

    if dataset == 'P12':
        temp_dim = 36
        static_dim = 9
        max_len = 215
        n_classes = 2
    elif dataset == 'P19':
        temp_dim = 34
        static_dim = 4
        max_len = 60
        n_classes = 2
    elif dataset == 'eICU':
        temp_dim = 14
        static_dim = 399
        max_len = 300
        n_classes = 2
    elif dataset == 'PAM':
        temp_dim = 17
        static_dim = 0
        max_len = 600
        n_classes = 8

    pe_dim = 8
    nhead = 8
    hidden_dim = 64
    nlayers = 2
    dropout = 0.5
    aggreg = 'mean'

    if dataset in {'P12', 'P19', 'eICU'}:
        if model_name == 'VPGNN_wo_p':
            from models.vpgnn_wo_p import VPGNN
            model = VPGNN(
                temp_dim, static_dim, hidden_dim, nhead,
                nlayers, dropout, max_len, aggreg, n_classes
            )
            return model
        elif model_name == 'VPGNN_wo_s':
            from models.vpgnn_wo_s import VPGNN
            model = VPGNN(
                temp_dim, static_dim, hidden_dim, nhead,
                nlayers, dropout, max_len, aggreg, n_classes
            )
            return model
        elif model_name == 'PatchTST':
            from models.patchtst import PatchTST
            model = PatchTST(
                temp_dim, static_dim, hidden_dim, nhead,
                nlayers, dropout, max_len, aggreg, n_classes
            )
            return model
        elif model_name == 'iTransformer':
            from models.itransformer import iTransformerModel
            model = iTransformerModel(
                temp_dim, static_dim, hidden_dim, nhead,
                nlayers, dropout, max_len, aggreg, n_classes
            )
            return model
        elif model_name == 'VPGNN':
            hidden_dim = 256
            nhead = 16
            nlayers = 4
            dropout = 0.6
            from models.vpgnn import VPGNN
            model = VPGNN(
                temp_dim, static_dim, hidden_dim, nhead,
                nlayers, dropout, max_len, aggreg, n_classes
            )
            return model
        elif model_name == 'TransformerModelV2':
            from models.transformer_model_v2 import TransformerModelV2
            model = TransformerModelV2(
                temp_dim, pe_dim, static_dim, nhead, hidden_dim, 
                nlayers, dropout, max_len, aggreg, n_classes
            )
            return model
        elif model_name == 'TransformerModel':
            from models.transformer_model import TransformerModel
            model = TransformerModel(
                temp_dim, static_dim, hidden_dim, nhead,
                nlayers, dropout, max_len, aggreg, n_classes
            )
            return model
        elif model_name in {'TemporalConvModel', 'TCN'}:
            from models.tcn import TemporalConvModel
            kernel_size = 4
            model = TemporalConvModel(
                temp_dim, static_dim, hidden_dim, nlayers,
                kernel_size, dropout, max_len, aggreg, n_classes
            )
            return model
        elif model_name == 'GRU':
            from models.gru import GRU_Model
            model = GRU_Model(
                temp_dim, static_dim, hidden_dim, nlayers,
                dropout, max_len, aggreg, n_classes
            )
            return model
        elif model_name == 'SAND':
            from models.sand import SAND_Model
            R, M = 24, 12
            model = SAND_Model(
                temp_dim, static_dim, hidden_dim, nhead, 
                nlayers, R, M, dropout, max_len, aggreg, n_classes
            )
            return model
        elif model_name == 'Raindrop':
            from models.raindrop import Raindrop
            model = Raindrop(
                d_inp=temp_dim, max_len=max_len, d_static=static_dim, n_classes=n_classes
            )
            return model

        elif model_name == 'ContiFormer':
            from models.contiformer import ContiFormer
            model = ContiFormer(
                temp_dim, static_dim, max_len, hidden_dim, hidden_dim,
                d_k=hidden_dim//4, d_v=hidden_dim//4
            )
            return model
        elif model_name == 'MTGNN':
            from models.mtgnn import MTGNN
            model = MTGNN(True, True, 2, temp_dim, num_static_features=static_dim,
                            node_dim=max_len, dilation_exponential=2, conv_channels=16, residual_channels=16, skip_channels=32,
                            end_channels=64, seq_length=max_len, in_dim=1, out_dim=1, layers=5, layer_norm_affline=False)
            return model
