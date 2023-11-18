import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim


class GetLayerNorm(nn.Module):
    def __init__(self, channel=2):
        super(GetLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(channel)

    def forward(self, x):  # n c t v
        x = x.permute(0, 2, 3, 1)  # N T V C
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvTemporalGraphical(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )
        self.pattn = PhysicalAttention()

    def forward(self, x, A, vgg):
        assert A.size(0) == self.kernel_size
        end_pos = x[:, :, -1, :]
        T = x.size(2)
        end_pos = end_pos.permute(0, 2, 1)  # N V C
        sequential_scene_attention = self.pattn(vgg, end_pos)  # V *32
        sequential_scene_attention = sequential_scene_attention.unsqueeze(0)  # 1 v c
        sequential_scene_attention = sequential_scene_attention.unsqueeze(1)  # 1 1 v c
        sequential_scene_attention = sequential_scene_attention.repeat(1, T, 1, 1)
        x = x.permute(0, 2, 3, 1)  # N T V C
        x = torch.cat((x, sequential_scene_attention), 3)
        x = x.view(1, T, A.size(2), -1)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = torch.einsum("nctv,tvw->nctw", (x, A))
        return x.contiguous(), A


class SceneAttentionShare(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        use_mdn=False,
        stride=1,
        dropout=0,
        residual=True,
    ):
        super(SceneAttentionShare, self).__init__()

        #         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(5, out_channels, kernel_size[1])
        self.in_channels = in_channels
        self.tcn = nn.Sequential(
            GetLayerNorm(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            GetLayerNorm(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        else:
            self.residual = lambda x: x

        self.prelu = nn.PReLU()

    def forward(self, x, A, vgg):

        res = self.residual(x)
        x, A = self.gcn(x, A, vgg)
        x = self.tcn(x)
        x_part = x[:, : self.in_channels, :, :] + res
        phs_part = x[:, self.in_channels :, :, :]
        x = torch.cat([x_part, phs_part], dim=1)
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


def make_mlp(dim_list):
    """
    批量生成全连接层
    :param dim_list: 维度列表
    :return: 一系列线性层的感知机
    """
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class PhysicalAttention(nn.Module):
    def __init__(
        self, attn_L=196, attn_D=512, ATTN_D_DOWN=16, bottleneck_dim=3, embedding_dim=10
    ):
        super(PhysicalAttention, self).__init__()

        self.L = attn_L  # 196=14*14
        self.D = attn_D  # 512
        self.D_down = ATTN_D_DOWN  # 16
        self.bottleneck_dim = bottleneck_dim  # 30
        self.embedding_dim = embedding_dim  # 16

        self.spatial_embedding = nn.Linear(2, self.embedding_dim)  # 2-16
        self.pre_att_proj = nn.Linear(self.D, self.D_down)  # 512-16

        mlp_pre_dim = self.embedding_dim + self.D_down  # 32
        mlp_pre_attn_dims = [mlp_pre_dim, 512, self.bottleneck_dim]
        self.mlp_pre_attn = make_mlp(mlp_pre_attn_dims)  # 32-512-32

        self.attn = nn.Linear(self.L * self.bottleneck_dim, self.L)  # 196*32--196

    def forward(self, vgg, end_pos):

        npeds = end_pos.size(1)  # N V C
        end_pos = end_pos[0, :, :]  # n*2
        curr_rel_embedding = self.spatial_embedding(end_pos)  # n*16
        curr_rel_embedding = curr_rel_embedding.view(-1, 1, self.embedding_dim).repeat(
            1, self.L, 1
        )  # n*196*16
        vgg = vgg.repeat(npeds, 1, 1, 1)  # n,14,14,512
        vgg = vgg.view(-1, self.D)  # n*196,512
        features_proj = self.pre_att_proj(vgg)  # n*196,16   x=n*900
        features_proj = features_proj.view(-1, self.L, self.D_down)  # n,196,16

        mlp_h_input = torch.cat([features_proj, curr_rel_embedding], dim=2)  # n*196*32
        attn_h = self.mlp_pre_attn(
            mlp_h_input.view(-1, self.embedding_dim + self.D_down)
        )  # -1，32--32
        attn_h = attn_h.view(npeds, self.L, self.bottleneck_dim)  # n*196*32

        attn_w = Func.softmax(self.attn(attn_h.view(npeds, -1)), dim=1)  # n*6272--n*196
        attn_w = attn_w.view(npeds, self.L, 1)  # n*196*1

        attn_h = torch.sum(attn_h * attn_w, dim=1)  # n*196*32
        return attn_h


class SocialSoftAttentionGCN(nn.Module):
    def __init__(
        self,
        n_ssagcn=1,
        n_txpcnn=1,
        input_feat=2,
        output_feat=5,
        seq_len=8,
        pred_seq_len=12,
        kernel_size=3,
    ):
        super(SocialSoftAttentionGCN, self).__init__()
        self.n_ssagcn = n_ssagcn
        self.n_txpcnn = n_txpcnn

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(
            SceneAttentionShare(input_feat, output_feat, (kernel_size, seq_len))
        )
        for j in range(1, self.n_ssagcn):
            self.st_gcns.append(
                SceneAttentionShare(output_feat, output_feat, (kernel_size, seq_len))
            )

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

    def forward(self, v, a, vgg):

        for k in range(self.n_ssagcn):
            v, a = self.st_gcns[k](v, a, vgg)

        v = v.permute(0, 2, 1, 3)

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)
        v = v.permute(0, 2, 1, 3)

        return v, a
