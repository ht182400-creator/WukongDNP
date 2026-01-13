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
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    reset_wrapper,
)
from kaiwu_agent.utils.common_func import attached
from agent_dqn.algorithm.algorithm import Algorithm
from agent_dqn.feature.definition import ObsData
from agent_dqn.feature.preprocessor import Preprocessor
from arena_proto.back_to_the_realm_v2.custom_pb2 import (
    RelativeDirection,
)


def one_hot_encoding(grid_pos):
    """
    This function performs one_hot_encoding on the grid position features transmitted by proto and returns a vector of length 256.
        - The first 128 dimensions are the one-hot features of the x-axis
        - The last 128 dimensions are the one-hot features of the z-axis

    此函数将proto传输的网格位置特征进行one_hot_encoding处理, 返回一个长度为256的向量
        - 前128维是x轴的one-hot特征
        - 后128维是z轴的one-hot特征
    """
    one_hot_pos_x, one_hot_pos_z = np.zeros(128).tolist(), np.zeros(128).tolist()
    one_hot_pos_x[grid_pos.x], one_hot_pos_z[grid_pos.z] = 1, 1

    return one_hot_pos_x + one_hot_pos_z


def read_relative_position(rel_pos):
    """
    This function unpacks and processes the relative position features transmitted by proto, and returns a vector of length 9.
        - The first 8 dimensions are one-hot direction features
        - The last dimension is the distance feature

    此函数将proto传输的相对位置特征进行拆包并处理, 返回一个长度为9的向量
        - 前8维是one-hot的方向特征
        - 最后一维是距离特征
    """
    direction = [0] * 8
    if rel_pos.direction != RelativeDirection.RELATIVE_DIRECTION_NONE:
        direction[rel_pos.direction - 1] = 1

    grid_distance = 1 if rel_pos.grid_distance < 0 else rel_pos.grid_distance / (128 * 128)
    feature = direction + [grid_distance]
    return feature


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.agent_type = agent_type
        self.logger = logger
        self.algorithm = Algorithm(device, logger, monitor)
        self.feature_manager = Preprocessor()
        self.last_action = -1

    def reset(self):
        self.last_action = -1

    @predict_wrapper
    def predict(self, list_obs_data):
        return self.algorithm.predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, observation):
        obs_data, _ = self.observation_process(observation["obs"], observation["extra_info"])
        act_data = self.algorithm.predict_detail([obs_data], exploit_flag=True)
        act = self.action_process(act_data[0])
        return act

    @learn_wrapper
    def learn(self, list_sample_data):
        self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.algorithm.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.algorithm.model.load_state_dict(torch.load(model_file_path, map_location=self.algorithm.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def observation_process(self, obs, extra_info):
        (feature_vec, legal_action, reward_list) = self.feature_manager.feature_process(
            [obs, extra_info], self.last_action
        )
        return ObsData(feature=feature_vec, legal_act=legal_action), reward_list

    def action_process(self, act_data):
        result = act_data.move_dir
        result += act_data.use_talent * 8
        self.last_action = result
        return result
