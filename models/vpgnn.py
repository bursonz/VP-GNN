import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# 假设 model_base.py 在可访问的路径中
from models.model_base import TimeSeriesModel


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
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, x):
        """
        x: [bsz, V, d]  # V=变量数, d=hidden_dim
        return:
            A: [bsz, h, V, V]       # 变量对变量的注意力打分(未softmax)
            v: [bsz, V, d]          # 仅做了 Wv 的线性映射结果
        """
        # x: bsz, Vs, d
        bsz, V, d = x.size()
        # device = x.device

        # 线性映射得到 q, k, v
        # q: (bsz, V, d) -> (bsz, V, h, dk)
        q = torch.matmul(x, self.Wq).view(bsz, V, self.num_heads, self.dk)
        q = q / np.sqrt(self.dk)

        # k: (bsz, V, d) -> (bsz, V, h, dk)
        k = torch.matmul(x, self.Wk).view(bsz, V, self.num_heads, self.dk)
        k = k / np.sqrt(self.dk)

        # v: (bsz, V, d)
        v = torch.matmul(x, self.Wv)
        # v = v.view(bsz, V, self.num_heads, self.dk)

        # 注意力打分 A = q @ k^T ，按变量维度计算
        # q: (bsz, V, h, dk) 与 k: (bsz, V, h, dk) -> A: (bsz, h, V, V)
        A = torch.einsum('bthd,blhd->bhtl', q, k)  # bsz, h, V, V
        # A = F.softmax(A, dim=-1)
        # A = F.dropout(A, self.dropout)
        # x = torch.einsum('bhvu,bvhd->bhvd', A, v)
        # x = self.Wo(x.reshape((bsz, T, d)))

        # 这里不做softmax/dropout，交给上层结合掩码后再做
        return A, v


class SelectiveVariableWiseMessagePassingBlock(nn.Module):
    def __init__(self, temp_dim: int, static_dim: int, max_len: int,
                 hidden_dim: int, num_heads: int, mha_dropout: float) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.temp_dim = temp_dim                   # 输入的变量数 V
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim               # 隐层维度 d
        self.num_heads = num_heads

        # 对每个变量通道做独立的 1x1 Conv（等价分组线性）：(B, V, T) -> 升到 (B, V*d, T)
        # 实际Conv1d接收 (B, in_channels=temp_dim, T) ，输出 (B, temp_dim*hidden_dim, T)
        self.temp_encoder = nn.Conv1d(temp_dim, temp_dim * hidden_dim, 1, groups=temp_dim)

        # 基于时间戳的PE，输出 (T, B, d)；后续会rearrange到 (B, d, 1, T)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)

        self.static = static_dim > 0
        if self.static:
            # 将静态特征 (B, static_dim) -> (B, V*d)
            self.emb = nn.Linear(static_dim, temp_dim * hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(mha_dropout)
        self.mha = MultiHeadAttention(hidden_dim, num_heads, mha_dropout)

        # 时序点上的“1D-MLP”（用Conv1d实现）：输入将会是 (B, d*V, T) ，输出 (B, d, T)
        self.mlp = nn.Sequential(
            nn.Conv1d(temp_dim * hidden_dim, hidden_dim * 4, 1),   # (B, d*V, T) -> (B, 4d, T)
            nn.BatchNorm1d(hidden_dim * 4),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 4, hidden_dim, 1),              # (B, 4d, T) -> (B, d, T)
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, Xtemp: Tensor, Xtimes: Tensor, Xstatic: Tensor | None, mask: Tensor) -> Tensor:
        """
        输入:
          Xtemp: (T, B, V)         # 时间×批×变量 的数值特征
          Xtimes: (T, B, d)        # 用于位置编码的时间表示（PositionalEncoding需要的形状）
          Xstatic: (B, static_dim) 或 None
          mask:  (T, B, V)         # 与Xtemp同形的可见性/有效性掩码(0/1)

        输出:
          z: (B, d, T)             # 每个时间步的跨变量融合表示
        """
        # 调整到以批为首的布局，便于Conv1d与后续计算
        Xtemp = Xtemp.permute((1, 2, 0))  # (B, V, T)
        mask = mask.permute((1, 2, 0))  # (B, V, T)

        seq_len = Xtemp.shape[-1]  # T

        # 对每个变量独立升维： (B, V, T) -> (B, V*d, T)
        Htemp = self.temp_encoder(Xtemp)  # (B, V*d, T)

        # 将通道拆回 (V, d) 结构，再排列成 (B, d, V, T)
        Htemp = Htemp.unflatten(1, (self.temp_dim, self.hidden_dim))  # (B, V, d, T)
        Htemp = Htemp.permute((0, 2, 1, 3))  # (B, d, V, T)

        # 位置编码：pos_encoder期望输入 (T, B, d)，返回 (T, B, d)
        Hpos = self.pos_encoder(Xtimes)  # (T, B, d)
        Hpos = Hpos.permute((1, 2, 0)).unsqueeze(2)  # (B, d, 1, T) -> 广播到变量维

        if Xstatic is not None:
            # 静态特征： (B, static_dim) -> (B, V*d) -> (B, V, d) -> (B, d, V, 1)
            emb = self.emb(Xstatic).view((-1, self.temp_dim, self.hidden_dim))  # (B, V, d)
            emb = emb.permute((0, 2, 1)).unsqueeze(-1)  # (B, d, V, 1)
            H = Htemp + Hpos + emb  # (B, d, V, T)
        else:
            H = Htemp + Hpos  # (B, d, V, T)

        # === 先在时间维做带掩码的均值，得到每个变量的表示 (B, V, d) 作为变量注意力的输入 ===
        # mask: (B, V, T) -> (B, 1, V, T) 与 H 对齐
        mask_4attn = mask.unsqueeze(1)  # (B, 1, V, T)
        h1 = H * mask_4attn  # (B, d, V, T)
        h2 = h1.sum(-1)  # (B, d, V)
        length_time = mask_4attn.sum(-1) + 1  # (B, 1, V)  +1防除零
        h3 = h2 / length_time  # (B, d, V)
        h4 = h3.permute((0, 2, 1))  # (B, V, d)

        # === 变量间多头注意力 ===
        A, mean_v = self.mha(h4)  # A:(B,h,V,V), mean_v:(B,V,d)

        # 将变量注意力权重扩展到每个时间步，并结合掩码做屏蔽
        A2 = A.unsqueeze(2)  # (B, h, 1, V, V)
        A3 = torch.repeat_interleave(A2, seq_len, 2)  # (B, h, T, V, V)

        # 变量可见性掩码，用于屏蔽 key 变量（最后一维）
        # mask2: (B, 1, T, V)  表示query的时间步上某变量是否有效
        mask2 = mask.permute((0, 1, 2))  # (B, V, T)
        mask2 = mask2.permute((0, 2, 1)).unsqueeze(1)  # (B, 1, T, V)

        # 构造 query/key 的二维掩码 (B,1,T,V,V)；当前实现对query侧未强约束，仅对key侧做屏蔽
        # 若需要双侧严格掩码，可将mask3改为mask2.unsqueeze(-1)
        mask3 = torch.ones_like(mask2).unsqueeze(-1)  # (B, 1, T, V, 1)
        mask4 = mask2.unsqueeze(3)  # (B, 1, T, 1, V)
        mask5 = mask3 * mask4  # (B, 1, T, V, V)

        # 大负数屏蔽无效 key
        A4 = A3 - 1e6 * (1 - mask5)  # (B, h, T, V, V)

        # softmax归一化 + dropout
        S = F.softmax(A4, dim=-1)  # (B, h, T, V, V)
        S = self.mha_dropout(S)

        # === 用注意力权重将 H 聚合（对变量维进行加权和），得到每个时间步融合后的表示 ===
        # 先把 H 拆成多头：(B, d, V, T) -> (B, h, d/h, V, T)
        H2 = H.unflatten(1, (self.num_heads, self.hidden_dim // self.num_heads))  # (B, h, d_h, V, T)

        # 按变量维聚合：S:(B,h,T,V,V) 与 H2:(B,h,d_h,V,T) 进行爱因斯坦求和
        # 目标：对最后一个 V（key维）聚合到每个 query 变量位置
        v2 = torch.einsum('bhlvu,bhdul->bhdvl', S, H2)  # (B, h, d_h, V, T)

        # 合并多头：(B, h, d_h, V, T) -> (B, d, V, T)
        v3 = v2.flatten(1, 2)  # (B, d, V, T)

        # 将 mean_v (B, V, d) 变为 (B, d, V, 1) 广播相加，形成一种残差/全局项
        v4 = v3 + mean_v.permute((0, 2, 1)).unsqueeze(3)  # (B, d, V, T)

        # 展平变量维，给到逐时间步的Conv-MLP
        v5 = v4.flatten(1, 2)  # (B, d*V, T)

        # 点上MLP： (B, d*V, T) -> (B, d, T)
        z = self.mlp(v5)  # (B, d, T)
        z = z + self.norm(z.permute(0,2,1)).permute(0,2,1)
        return z

class PatchGCNAggregationBlock(nn.Module):
    """
    多层 Patch-GCN-Aggregation：将输入 [B, d, T] 递归patch聚合到 [B, d, 1]
    自动计算每层patch_len、mask无效填0，有效长度递推。
    支持batch化GCN聚合，显著提升速度。
    """
    def __init__(self, hidden_dim, nlayers, maxlen):
        super().__init__()
        self.nlayers = nlayers
        self.maxlen = maxlen
        self.hidden_dim = hidden_dim

        # 自动推算每层patch_len/num
        self.patch_lens, self.patch_nums = self.compute_patch_scheme(maxlen, nlayers)
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(nlayers)
        ])
        self.edge_indices = [self.build_patch_edge_index(l) for l in self.patch_lens]
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def compute_patch_scheme(maxlen, nlayers):
        patch_lens = []
        patch_nums = []
        cur_len = maxlen
        for i in range(nlayers):
            remain_layers = nlayers - i
            patch_num = math.ceil(cur_len ** (1 / remain_layers))
            patch_len = math.ceil(cur_len / patch_num)
            actual_patch_num = math.ceil(cur_len / patch_len)
            patch_lens.append(patch_len)
            patch_nums.append(actual_patch_num)
            cur_len = actual_patch_num
        return patch_lens, patch_nums

    @staticmethod
    def build_patch_edge_index(patch_len):
        edge_index = []
        for i in range(patch_len - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
        for i in range(patch_len):
            edge_index.append([i, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return edge_index

    @staticmethod
    def patchify_with_mask(x, patch_len):
        B, d, T = x.shape
        pad_len = (patch_len - (T % patch_len)) % patch_len
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(B, d, pad_len, device=x.device, dtype=x.dtype)], dim=2)
        T_new = x.shape[-1]
        patch_num = T_new // patch_len
        x_patch = x.view(B, d, patch_num, patch_len)   # [B, d, patch_num, patch_len]
        return x_patch, patch_num, pad_len

    @staticmethod
    def build_patch_mask(lengths, patch_num, patch_len):
        B = lengths.shape[0]
        masks = []
        for b in range(B):
            seq_mask = torch.arange(patch_num * patch_len, device=lengths.device) < lengths[b]
            seq_mask = seq_mask.view(patch_num, patch_len)  # [patch_num, patch_len]
            masks.append(seq_mask)
        masks = torch.stack(masks, dim=0).float()  # [B, patch_num, patch_len]
        return masks

    def forward(self, x, lengths):
        """
        x: [B, d, T]
        lengths: [B]
        """
        for i, gnn in enumerate(self.gnn_layers):
            patch_len = self.patch_lens[i]
            edge_index = self.edge_indices[i].to(x.device)
            x_patch, patch_num, pad_len = self.patchify_with_mask(x, patch_len)   # [B, d, patch_num, patch_len]
            mask = self.build_patch_mask(lengths, patch_num, patch_len)           # [B, patch_num, patch_len]

            # 1. reshape到batch模式，拼成一大图
            # x_patch: [B, d, patch_num, patch_len] -> [B*patch_num, patch_len, d]
            x_patch = x_patch.permute(0,2,3,1).contiguous().view(-1, patch_len, self.hidden_dim)
            mask = mask.contiguous().view(-1, patch_len, 1)  # [B*patch_num, patch_len, 1]
            total_patches = x_patch.shape[0]

            # 2. batch化edge_index拼接（每个patch都一样，只需偏移node id）
            edge_index_list = []
            node_offset = 0
            for _ in range(total_patches):
                edge_index_list.append(edge_index + node_offset)
                node_offset += patch_len
            edge_index_batch = torch.cat(edge_index_list, dim=1)  # [2, total_edges]

            # 3. 合成所有patch节点特征
            x_input = x_patch.reshape(-1, self.hidden_dim)   # [B*patch_num*patch_len, d]

            # 4. 一次性GCN（大幅提升速度）
            out = gnn(x_input, edge_index_batch)    # [B*patch_num*patch_len, d]
            out = out.view(total_patches, patch_len, self.hidden_dim)  # [B*patch_num, patch_len, d]

            # 5. mask池化回 patch 特征
            sum_out = (out * mask).sum(dim=1)   # [B*patch_num, d]
            valid = mask.sum(dim=1).clamp(min=1.0) # [B*patch_num, 1]
            feats = sum_out / valid    # [B*patch_num, d]

            # 6. reshape回[batch, d, patch_num]
            B = lengths.shape[0]
            feats = feats.view(B, patch_num, self.hidden_dim).permute(0,2,1)  # [B, d, patch_num]
            x = feats
            lengths = torch.full((B,), patch_num, dtype=lengths.dtype, device=x.device)
        # 最后x: [B, d, 1]
        x = x.squeeze(-1)   # [B, d]

        return x

class PatchGATAggregationBlock(nn.Module):
    """
    多层 Patch-GAT-Aggregation：将输入 [B, d, T] 递归patch聚合到 [B, d, 1]
    自动计算每层patch_len、mask无效填0，有效长度递推。
    """
    def __init__(self, hidden_dim, nlayers, maxlen):
        super().__init__()
        self.nlayers = nlayers
        self.maxlen = maxlen
        self.hidden_dim = hidden_dim

        # 自动推算每层patch_len/num
        self.patch_lens, self.patch_nums = self.compute_patch_scheme(maxlen, nlayers)
        self.gnn_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim) for _ in range(nlayers)
        ])
        self.edge_indices = [self.build_patch_edge_index(l) for l in self.patch_lens]

    @staticmethod
    def compute_patch_scheme(maxlen, nlayers):
        patch_lens = []
        patch_nums = []
        cur_len = maxlen
        for i in range(nlayers):
            remain_layers = nlayers - i
            patch_num = math.ceil(cur_len ** (1 / remain_layers))
            patch_len = math.ceil(cur_len / patch_num)
            actual_patch_num = math.ceil(cur_len / patch_len)
            patch_lens.append(patch_len)
            patch_nums.append(actual_patch_num)
            cur_len = actual_patch_num
        return patch_lens, patch_nums

    @staticmethod
    def build_patch_edge_index(patch_len):
        edge_index = []
        for i in range(patch_len - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
        for i in range(patch_len):
            edge_index.append([i, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return edge_index

    @staticmethod
    def patchify_with_mask(x, patch_len):
        B, d, T = x.shape
        pad_len = (patch_len - (T % patch_len)) % patch_len
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(B, d, pad_len, device=x.device, dtype=x.dtype)], dim=2)
        T_new = x.shape[-1]
        patch_num = T_new // patch_len
        x_patch = x.view(B, d, patch_num, patch_len)   # [B, d, patch_num, patch_len]
        return x_patch, patch_num, pad_len

    @staticmethod
    def build_patch_mask(lengths, patch_num, patch_len):
        B = lengths.shape[0]
        masks = []
        for b in range(B):
            seq_mask = torch.arange(patch_num * patch_len, device=lengths.device) < lengths[b]
            seq_mask = seq_mask.view(patch_num, patch_len)  # [patch_num, patch_len]
            masks.append(seq_mask)
        masks = torch.stack(masks, dim=0).float()  # [B, patch_num, patch_len]
        return masks

    def forward(self, x, lengths):
        """
        x: [B, d, T]
        lengths: [B]
        """
        for i, gnn in enumerate(self.gnn_layers):
            patch_len = self.patch_lens[i]
            edge_index = self.edge_indices[i].to(x.device)
            x_patch, patch_num, pad_len = self.patchify_with_mask(x, patch_len)   # [B, d, patch_num, patch_len]
            mask = self.build_patch_mask(lengths, patch_num, patch_len)           # [B, patch_num, patch_len]
            x_patch = x_patch.permute(0,2,3,1).contiguous()                       # [B, patch_num, patch_len, d]
            B, patch_num, patch_len, d = x_patch.shape
            outputs = []
            for b in range(B):
                feats = []
                for p in range(patch_num):
                    x_patch_bp = x_patch[b,p]                # [patch_len, d]
                    mask_bp = mask[b,p].unsqueeze(-1)        # [patch_len, 1]
                    out = gnn(x_patch_bp, edge_index)        # [patch_len, d]
                    sum_out = (out * mask_bp).sum(dim=0)
                    valid = mask_bp.sum()
                    feat = sum_out / valid.clamp(min=1.0)
                    feats.append(feat)
                feats = torch.stack(feats, dim=0)   # [patch_num, d]
                outputs.append(feats)
            x = torch.stack(outputs, dim=0).transpose(1,2)   # [B, d, patch_num]
            lengths = torch.full((x.shape[0],), patch_num, dtype=lengths.dtype, device=x.device)
        # 最后x: [B, d, 1]
        return x.squeeze(-1)   # [B, d]

class Classifier(nn.Module):
    def __init__(self, hidden_dim,n_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x,x_mark):
        x = self.fc1(x)
        x = self.relu(x)
        if x_mark is not None:
            x = self.fc2(torch.cat([x, x_mark], dim=-1))
            x = self.relu(x)
        x = self.fc3(x)
        return x

class VPGNN(TimeSeriesModel):
    def __init__(
            self,
            temp_dim:int=36,
            static_dim:int=9,
            hidden_dim:int=64,
            nhead:int=8,
            nlayers:int=3,   # 推荐多层聚合
            dropout:float=0.5,
            max_len:int=215,
            aggreg:str='mean',
            n_classes:int=2,
    ):
        super().__init__()
        self.model_type = 'PGAAN'
        self.aggreg = aggreg if not None else 'mean'
        self.SVMPBlock = SelectiveVariableWiseMessagePassingBlock(temp_dim, static_dim, max_len, hidden_dim, nhead, dropout)
        self.PatchGATBlock = PatchGCNAggregationBlock(hidden_dim, nlayers, max_len)

        # cls-token
        # self.cls_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.static_embedding = nn.Linear(static_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = Classifier(hidden_dim, n_classes)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, Xtemp, Xtimes, Xstatic, lengths):
        """
        input:
            Xtemp:   (T, B, 2V)   # 前一半通道是数值特征，后一半通道是变量可见性掩码
            Xtimes:  (T, B, d)    # 用于位置编码的时间特征/索引（需与PositionalEncoding匹配）
            Xstatic: (B, static_dim) 或 None
            lengths: (B,)         # 每个样本有效长度（用于时间维屏蔽）

        return:
            logits: (B, n_classes)
        """
        maxlen, batch_size, V_double = Xtemp.shape
        vmask = Xtemp[:, :, int(Xtemp.shape[2] / 2):]
        Xtemp = Xtemp[:, :, :int(Xtemp.shape[2] / 2)]

        # 1. 选择性变量维度消息传递
        x = self.SVMPBlock(Xtemp, Xtimes, Xstatic, vmask)        # [B, D, T]

        # 2. Patch-GAT-Aggregation Block
        # 注意lengths shape为[B,]，确保传入的是int tensor
        if lengths.dim() > 1:
            lengths = lengths.squeeze(-1)
        x = self.PatchGATBlock(x, lengths)                       # [B, D]
        if x.ndim == 3:
            x = x.mean(dim=-1)
        x = x + self.norm(x)

        # print('x shape before classifier:', x.shape)
        return self.classifier(self.dropout(x), self.static_embedding(Xstatic))                  # [B, n_classes]
