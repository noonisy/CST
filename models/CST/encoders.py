import math
import torch
from torch import nn
from torch.nn import functional as F
from .utils import PositionWiseFeedForward
from .attention import MultiHeadBoxAttention
from .grid_aug import BoxRelationalEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, M=2, p=0.4, identity_map_reordering=False):
        super(EncoderLayer, self).__init__()
        self.M = M
        self.p = p
        self.identity_map_reordering = identity_map_reordering
        self.mhatts = nn.ModuleList([MultiHeadBoxAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering)
                                     for _ in range(M)])
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, pos=None):
        grids_pos = pos
        v = values + pos
        if self.training:
            rho = self.p
        else:
            rho = 0.0
        pro = torch.rand(self.M)  # .cuda()
        pro = (pro >= rho).float()
        
        att = None
        for i in range(self.M):
            if i == 0:
                att = (self.mhatts[i](queries, keys, v, grids_pos, relative_geometry_weights, attention_mask) * pro[i]) / (1 - rho)
            else:
                att += (self.mhatts[i](queries, keys, v,  grids_pos, relative_geometry_weights, attention_mask) * pro[i]) / (1 - rho)
        att /= self.M

        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


def spatial_shift1(x):
    b, w, h, d = x.size()
    x[:, 1:, :, :d//4] = x[:, :w-1, :, :d//4]
    x[:, :w-1, :, d//4:d//2] = x[:, 1:, :, d//4:d//2]
    x[:, :, 1:, d//2:d*3//4] = x[:, :, :h-1, d//2:d*3//4]
    x[:, :, :h-1, 3*d//4:] = x[:, :, 1:, 3*d//4:]
    return x


def spatial_shift2(x):
    b, w, h, d = x.size()
    x[:, :, 1:, :d//4] = x[:, :, :h-1, :d//4]
    x[:, :, :h-1, d//4:d//2] = x[:, :, 1:, d//4:d//2]
    x[:, 1:, :, d//2:d*3//4] = x[:, :w-1, :, d//2:d*3//4]
    x[:, :w-1, :, 3*d//4:] = x[:, 1:, :, 3*d//4:]
    return x


class SSA(nn.Module):
    def __init__(self, d, N):
        super().__init__()
        self.N = N
        self.d = d
        self.CNN = nn.Sequential(
            nn.Conv2d(N*d, N*d, kernel_size=1, bias=False),
            nn.BatchNorm2d(N*d),
            nn.ReLU(),
            nn.Conv2d(N*d, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d)
        )

    def forward(self, x):
        bs, n, c = x.size()
        h, w = int(math.sqrt(n)), int(math.sqrt(n))
        x = x.view(bs, h, w, c)
        c_each = c//3
        x1 = spatial_shift1(x[:, :, :, :c_each])
        x2 = spatial_shift2(x[:, :, :, c_each:c_each*2])
        x3 = x[:, :, :, c_each*2:]
        x_all = torch.cat([x1, x2, x3], -1)  # bs,h,w,3*dim
        out = self.CNN(x_all.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # bs,h,w,dim
        out = out.view(bs, -1, self.d)  # bs,n,dim
        return out


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, M=2, p=0.4, identity_map_reordering=False):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, M, p,
                                                  identity_map_reordering=identity_map_reordering)
                                     for _ in range(N)])
        self.padding_idx = padding_idx
        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])
        self.SSA = SSA(d_model, N)

    def forward(self, input, pos=None):
        # input (b_s, seq_len, d_model)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        relative_geometry_embeddings = BoxRelationalEmbedding(input)

        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = \
            [layer(flatten_relative_geometry_embeddings).view(box_size_per_head) for layer in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        outs = []
        out = input
        for l in self.layers:
            out = l(out, out, out, relative_geometry_weights, attention_mask, pos=pos)
            outs.append(out)

        stack_out = torch.cat(outs, -1)
        stack_out = self.SSA(stack_out)
        out = out + 0.2 * stack_out

        return out, attention_mask


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, pos=None):
        mask = (torch.sum(input, dim=-1) == 0).unsqueeze(-1)
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)
        return super(TransformerEncoder, self).forward(out, pos=pos)
