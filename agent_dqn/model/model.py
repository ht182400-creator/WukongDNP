#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import List
from agent_dqn.conf.conf import Config

import sys
import os

if os.path.basename(sys.argv[0]) == "learner.py":
    import torch

    torch.set_num_interop_threads(2)
    torch.set_num_threads(2)
else:
    import torch

    torch.set_num_interop_threads(4)
    torch.set_num_threads(4)


class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        self.treasure_mlp = MLP([6, 64, 64], "treasure_mlp")
        self.buff_mlp = MLP([8, 64, 64], "buff_mlp")
        self.end_mlp = MLP([6, 64, 64], "end_mlp")
        self.dist_mlp = MLP([16 + 8 + 8, 64, 64], "dist_mlp")
        self.global_mlp = MLP([9, 64, 64], "global_mlp")
        self.visited_mlp = MLP([16 * 16, 64, 64], "visited_mlp")
        self.main_fc_dim_list = [64 * 6, 512, 256]
        self.main_mlp_net = MLP(self.main_fc_dim_list, "main_mlp_net", non_linearity_last=True)
        self.q_mlp = MLP([256, 64, action_shape], "q_mlp")
        self.feature_split_shape = Config.FEATURE_SPLIT_SHAPE

    # Forward inference
    # 前向推理
    def forward(self, feature):

        treasure_feature, buff_feature, end_feature, dist_feature, global_feature, visited_feature = feature.split(
            self.feature_split_shape, dim=1
        )
        treasure_feature = treasure_feature.reshape(-1, 13, 6)
        buff_feature = buff_feature.reshape(-1, 1, 8)
        end_feature = end_feature.reshape(-1, 1, 6)

        treasure_out = self.treasure_mlp(treasure_feature)
        buff_out = self.buff_mlp(buff_feature)
        end_out = self.end_mlp(end_feature)

        dist_out = self.dist_mlp(dist_feature)
        global_out = self.global_mlp(global_feature)
        visited_out = self.visited_mlp(visited_feature)

        treasure_out, _ = treasure_out.max(dim=1)
        buff_out = buff_out.reshape(-1, 64)
        end_out = end_out.reshape(-1, 64)

        combined_features = torch.cat([treasure_out, buff_out, end_out, dist_out, global_out, visited_out], dim=1)
        fc_mlp_out = self.main_mlp_net(combined_features)
        logits = self.q_mlp(fc_mlp_out)

        return logits


#######################
## Utility functions ##
#######################
def make_fc_layer(in_features: int, out_features: int):
    """Wrapper function to create and initialize a linear layer

    Args:
        in_features (int): ``in_features``
        out_features (int): ``out_features``

    Returns:
        nn.Linear: the initialized linear layer
    """
    fc_layer = nn.Linear(in_features, out_features)

    # initialize weight and bias
    # 初始化权重与偏移量
    # nn.init.xavier_uniform_(fc_layer.weight)
    nn.init.orthogonal(fc_layer.weight)
    nn.init.zeros_(fc_layer.bias)

    return fc_layer


############################
## Building-block classes ##
############################
class MLP(nn.Module):
    """
    A simple multi-layer perceptron
    一个简单的多层感知器
    """

    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        """
        Create a MLP object
        Args:
            fc_feat_dim_list (List[int]): ``in_features`` of the first linear layer followed by
                ``out_features`` of each linear layer
            name (str): human-friendly name, serving as prefix of each comprising layers
            non_linearity (nn.Module, optional): the activation function to use. Defaults to nn.ReLU.
            non_linearity_last (bool, optional): whether to append a activation function in the end.
                Defaults to False.

        创建一个 MLP 对象
        参数：
            fc_feat_dim_list (List[int])：第一个线性层的“in_features”，以及每个线性层的“out_features”
            name (str)：人性化的名称，作为每个组成层的前缀
            non_linearity (nn.Module，可选)：要使用的激活函数。默认为 nn.ReLU。
            non_linearity_last (bool，可选)：是否在末尾附加激活函数。默认为 False。
        """
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            # no relu for the last fc layer of the mlp unless required
            # 除非有需要，否则 mlp 的最后一个 fc 层不使用 relu
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
