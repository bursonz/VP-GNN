import torch.nn as nn
import argparse
import torch
import torch.nn.functional as F
from logging import Logger


def count_parameters(logger: Logger, model: nn.Module):
    """Print no. of parameters in model, no. of traininable parameters,
     no. of parameters in each dtype."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('\nModel details:')
    logger.info('# parameters: '+str(total))
    logger.info('# trainable parameters: '+str(trainable)+', '
                + str(100*trainable/total)+'%')

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    logger.info('#params by dtype:')
    for k, v in dtypes.items():
        logger.info(str(k)+': '+str(v)+', '+str(100*v/total)+'%')


class TimeSeriesModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, Xtemp, Xtimes, Xstatic, lengths):
        pass


class TimeSeriesModel2(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        if args.model_type != 'istrats':
            self.static_emb = nn.Sequential(
                nn.Linear(args.static_dim, args.hid_dim*2),
                nn.GELU(), nn.Linear(args.hid_dim*2, args.hid_dim)
            )
        if args.model_type == 'istrats':
            temp_static_emb_dim = args.hid_dim+args.D
        elif args.model_type == 'sand':
            temp_static_emb_dim = args.hid_dim*args.M+args.hid_dim
        else:
            temp_static_emb_dim = args.hid_dim*2
        self.pretrain = args.pretrain == 1
        self.finetune = args.load_ckpt_path is not None
        if self.pretrain:
            self.forecast_head = nn.Linear(temp_static_emb_dim, args.V)
        elif self.finetune:
            self.forecast_head = nn.Linear(temp_static_emb_dim, args.V)
            self.binary_head = nn.Linear(args.V, 1)
            self.pos_class_weight = torch.tensor(args.pos_class_weight)
        else:
            self.binary_head = nn.Linear(temp_static_emb_dim, 1)
            self.pos_class_weight = torch.tensor(args.pos_class_weight)

    def binary_cls_final(self, logits, labels):
        if labels is not None:
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self.pos_class_weight)
        else:
            return F.sigmoid(logits)

    def forecast_final(self, ts_emb, forecast_values, forecast_mask):
        pred = self.forecast_head(ts_emb)  # bsz, V
        return (forecast_mask*(pred-forecast_values)**2).sum()/forecast_mask.sum()
