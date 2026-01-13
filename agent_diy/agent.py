#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
    reset_wrapper,
)
from agent_diy.model.model import Model
from kaiwu_agent.utils.common_func import attached
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        pass

    @exploit_wrapper
    def exploit(self, list_obs_data):
        pass

    @learn_wrapper
    def learn(self, list_sample_data):
        pass

    # @save_model_wrapper
    # def save_model(self, path=None, id="1"):
    #     pass
    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        try:
            # 确保路径存在
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                
            model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

            # 将模型的状态字典拷贝到CPU
            model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.algorithm.model.state_dict().items()}
            torch.save(model_state_dict_cpu, model_file_path)

            self.logger.info(f"save model {model_file_path} successfully")
        except Exception as e:
            self.logger.error(f"save model failed: {str(e)}")
            raise


    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        pass

    def observation_process(self, obs, extra_info):
        pass

    def action_process(self, act_data):
        pass
