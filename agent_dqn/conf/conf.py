#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration, including dimension settings, algorithm parameter settings.
# The last few configurations in the file are for the Kaiwu platform to use and should not be changed.
# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:

    # features
    # 特征
    FEATURES = [
        6 * 13,
        8,
        6,
        16 + 8 + 8,
        9,
        16 * 16,
    ]

    FEATURE_SPLIT_SHAPE = FEATURES

    # Size of observation
    # observation的维度
    DIM_OF_OBSERVATION = 389

    # Dimension of movement action direction
    # 移动动作方向的维度
    DIM_OF_ACTION_DIRECTION = 8

    # Dimension of flash action direction
    # 空间穿梭动作方向的维度
    DIM_OF_TALENT = 8

    # Input dimension of reverb sample on learner. Note that different algorithms have different dimensions.
    # **Note**, this item must be configured correctly and should be aligned with the NumpyData2SampleData function data in definition.py
    # Otherwise the sample dimension error may be reported
    # learner上reverb样本的输入维度
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = 2 * (DIM_OF_OBSERVATION + DIM_OF_ACTION_DIRECTION + DIM_OF_TALENT) + 4

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    # 提高GAMMA以更重视长期奖励，因为任务较长(2000步)且奖励更丰富
    # Increased GAMMA to value long-term rewards more, as the task is long (2000 steps) and rewards are richer
    GAMMA = 0.95  # 从0.9提高到0.95，更重视长期奖励

    # epsilon
    # 探索参数：epsilon-greedy策略
    EPSILON_MIN = 0.1  # 最小探索率，保持一定探索
    EPSILON_MAX = 1.0  # 初始探索率，完全随机探索
    # epsilon衰减率：随着预测次数增加，探索率逐渐降低
    # 考虑到奖励机制优化后需要更多探索来学习收集宝箱和优化路径
    EPSILON_DECAY = 2e-6  # 从1e-6提高到2e-6，稍微加快探索收敛速度

    # Initial learning rate
    # 初始的学习率
    # 根据训练日志分析：训练不稳定（Q值从正变负，价值损失大幅上升），降低学习率以提高稳定性
    START_LR = 5e-5  # 从1e-4降低到5e-5，提高训练稳定性，减少价值损失波动
