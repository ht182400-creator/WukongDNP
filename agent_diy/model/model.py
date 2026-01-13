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


# class Model(nn.Module):
#     def __init__(self, state_shape, action_shape=0, softmax=False):
#         super().__init__()

#         # User-defined network
#         # 用户自定义网络

# class Model(nn.Module):
#     def __init__(self, state_shape, action_shape=0, softmax=False):
#         super().__init__()
#         # 增加网络容量
#         self.treasure_mlp = MLP([6, 128, 128], "treasure_mlp")
#         self.buff_mlp = MLP([8, 128, 128], "buff_mlp")
#         self.end_mlp = MLP([6, 128, 128], "end_mlp")
#         self.dist_mlp = MLP([16 + 8 + 8, 128, 128], "dist_mlp")
#         self.global_mlp = MLP([9, 128, 128], "global_mlp")
#         self.visited_mlp = MLP([16 * 16, 128, 128], "visited_mlp")
        
#         # 增加主网络层数
#         self.main_fc_dim_list = [128 * 6, 1024, 512, 256]
#         self.main_mlp_net = MLP(self.main_fc_dim_list, "main_mlp_net", non_linearity_last=True)
#         self.q_mlp = MLP([256, 128, action_shape], "q_mlp")

class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        # 增加网络容量
        self.treasure_mlp = MLP([6, 128, 128], "treasure_mlp")
        self.buff_mlp = MLP([8, 128, 128], "buff_mlp")
        self.end_mlp = MLP([6, 128, 128], "end_mlp")
        self.dist_mlp = MLP([16 + 8 + 8, 128, 128], "dist_mlp")
        self.global_mlp = MLP([9, 128, 128], "global_mlp")
        self.visited_mlp = MLP([16 * 16, 128, 128], "visited_mlp")
        
        # 增加主网络深度
        self.main_fc_dim_list = [128 * 6, 512, 512, 256]
        self.main_mlp_net = MLP(self.main_fc_dim_list, "main_mlp_net", non_linearity_last=True)
        
        # Dueling DQN 结构
        self.value_stream = MLP([256, 128, 1], "value_stream")
        self.advantage_stream = MLP([256, 128, action_shape], "advantage_stream")
        
        self.feature_split_shape = Config.FEATURE_SPLIT_SHAPE

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
        buff_out = buff_out.reshape(-1, 128)
        end_out = end_out.reshape(-1, 128)

        combined_features = torch.cat([treasure_out, buff_out, end_out, dist_out, global_out, visited_out], dim=1)
        fc_mlp_out = self.main_mlp_net(combined_features)
        
        # Dueling DQN
        value = self.value_stream(fc_mlp_out)
        advantage = self.advantage_stream(fc_mlp_out)
        
        # 组合价值和优势
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values