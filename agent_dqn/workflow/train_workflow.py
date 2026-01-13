#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
import os
from kaiwu_agent.utils.common_func import Frame, attached

from tools.train_env_conf_validate import read_usr_conf
from agent_dqn.feature.definition import (
    sample_process,
)
from tools.metrics_utils import get_training_metrics


@attached
def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    try:
        env, agent = envs[0], agents[0]
        episode_num_every_epoch = 100000
        last_save_model_time = 0

        # Read and validate configuration file
        # 配置文件读取和校验
        usr_conf = read_usr_conf("agent_dqn/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error(f"usr_conf is None, please check agent_dqn/conf/train_env_conf.toml")
            return

        # Get max_step from configuration file
        # 从配置文件读取最大步数
        map_name = usr_conf.get("env_conf", {}).get("map_name", "map_cherry")
        max_step = usr_conf.get("env_conf", {}).get(map_name, {}).get("max_step", 2000)
        logger.info(f"Loaded max_step from config: {max_step}")
        
        # Set max_step to preprocessor
        # 设置最大步数到preprocessor
        agent.feature_manager.max_step = max_step

        while True:
            for g_data in run_episodes(episode_num_every_epoch, env, agent, usr_conf, logger, monitor, max_step):
                agent.learn(g_data)
                g_data.clear()

                # save model file
                # 保存model文件
                now = time.time()
                if now - last_save_model_time >= 1800:
                    agent.save_model()
                    last_save_model_time = now

    except Exception as e:
        raise RuntimeError(f"workflow error")


def run_episodes(n_episode, env, agent, usr_conf, logger, monitor, max_step=2000):
    try:
        last_put_monitor_time = 0
        for episode in range(n_episode):
            collector = list()

            # Retrieving training metrics
            # 获取训练中的指标
            if episode % 50 == 0:
                training_metrics = get_training_metrics()
                if training_metrics:
                    logger.info(f"training_metrics is {training_metrics}")

            # Reset the task and get the initial state
            # 重置任务, 并获取初始状态
            obs, extra_info = env.reset(usr_conf=usr_conf)
            if extra_info["result_code"] < 0:
                logger.error(
                    f"env.reset result_code is {extra_info['result_code']}, result_message is {extra_info['result_message']}"
                )
                raise RuntimeError(extra_info["result_message"])
            elif extra_info["result_code"] > 0:
                continue

            # At the start of each game, support loading the latest model file
            # The call will load the latest model from a remote training node
            # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
            agent.reset()
            agent.load_model(id="latest")

            # Feature processing
            # 特征处理
            obs_data, _ = agent.observation_process(obs, extra_info)

            done = False
            step = 0
            diy_1 = 0
            diy_2 = 0
            diy_3 = 0
            diy_4 = 0
            diy_5 = 0

            max_step_no = int(os.environ.get("max_step_no", "0"))

            while not done:
                # Agent performs inference, gets the predicted action for the next frame
                # Agent 进行推理, 获取下一帧的预测动作
                act_data = agent.predict(list_obs_data=[obs_data])

                # Unpack ActData into action
                # ActData 解包成动作
                act = agent.action_process(act_data[0])

                # Interact with the environment, execute actions, get the next state
                # 与环境交互, 执行动作, 获取下一步的状态
                step_no, _obs, terminated, truncated, _extra_info = env.step(act)
                if _extra_info["result_code"] != 0:
                    logger.warning(
                        f"_extra_info.result_code is {_extra_info['result_code']}, \
                        _extra_info.result_message is {_extra_info['result_message']}"
                    )
                    break

                step += 1

                # Feature processing
                # 特征处理
                _obs_data, reward_list = agent.observation_process(_obs, _extra_info)
                reward = sum(reward_list)

                # Determine task over, and update the number of victories
                # 判断任务结束, 并更新胜利次数
                game_info = _extra_info["game_info"]
                treasure_score = game_info.get('treasure_score', 0)  # 宝箱分数（每个宝箱100分）
                treasure_count = treasure_score // 100  # 转换为宝箱数量（每个宝箱100分）
                
                if truncated:
                    # 超时惩罚：根据收集的宝箱数量给予部分奖励，但仍有惩罚
                    # 超时意味着没有完成任务，给予较大惩罚，但收集的宝箱仍有一定价值
                    treasure_bonus = treasure_count * 12  # 每个宝箱12分奖励（部分补偿）
                    time_penalty = -15  # 超时基础惩罚
                    reward = time_penalty + treasure_bonus
                    logger.info(
                        f"Game truncated! step_no:{step_no} end:{game_info['end_pos']['x']},{game_info['end_pos']['z']} "
                        f"score:{game_info['total_score']} treasure:{treasure_score} reward:{reward:.2f}"
                    )
                elif terminated:
                    # 成功到达终点：基础奖励 + 宝箱奖励（主要）+ 步数效率奖励（次要）
                    # 策略调整：优先收集宝箱（至少4-5个），在max_step步内到达终点即可，不需要尽量少步数
                    # 基础奖励：成功到达终点的基础奖励，保持适中
                    base_reward = 25 * min(agent.feature_manager.step_no_norm + 0.2, 1)  # 从35降低到25
                    
                    # 宝箱奖励：每个宝箱额外奖励，大幅提高权重，优先收集宝箱
                    # 收集4个以上宝箱时给予额外奖励
                    treasure_bonus = treasure_count * 25  # 每个宝箱25分奖励，大幅提高宝箱奖励
                    if treasure_count >= 4:
                        treasure_bonus += 50  # 收集4个以上宝箱时额外奖励50分
                    if treasure_count >= 5:
                        treasure_bonus += 50  # 收集5个以上宝箱时额外奖励50分（总共100分）
                    
                    # 步数效率奖励：步数越少奖励越大，但权重很小，因为不需要尽量少步数
                    # 只要在max_step步内完成即可，步数效率是次要目标
                    step_efficiency = max(0, (max_step - step_no) / max_step)
                    step_bonus = step_efficiency ** 1.5 * 5  # 从25大幅降低到5，步数效率是次要目标
                    
                    # 综合奖励：宝箱奖励是主要激励，基础奖励确保到达终点，步数效率奖励很小
                    reward = base_reward + treasure_bonus + step_bonus
                    logger.info(
                        f"Game terminated! step_no:{step_no} end:{game_info['end_pos']['x']},{game_info['end_pos']['z']} "
                        f"score:{game_info['total_score']} treasure:{treasure_score} (count:{treasure_count}) "
                        f"reward:{reward:.2f} (base:{base_reward:.2f} treasure:{treasure_bonus:.2f} step:{step_bonus:.2f})"
                    )
                done = terminated or truncated or (max_step_no > 0 and step >= max_step_no)

                # Construct task frames to prepare for sample construction
                # 构造任务帧，为构造样本做准备
                frame = Frame(
                    obs=obs_data.feature,
                    _obs=_obs_data.feature,
                    obs_legal=obs_data.legal_act,
                    _obs_legal=_obs_data.legal_act,
                    act=act,
                    rew=reward,
                    done=done,
                    ret=reward,
                )

                collector.append(frame)

                # If the task is over, the sample is processed and sent to training
                # 如果任务结束，则进行样本处理，将样本送去训练
                if done:
                    if monitor and time.time() - last_put_monitor_time >= 60:
                        monitor_data = {
                            "diy_1": diy_1,
                            "diy_2": diy_2,
                            "diy_3": diy_3,
                            "diy_4": diy_4,
                            "diy_5": diy_5,
                        }
                        monitor.put_data({os.getpid(): monitor_data})
                        last_put_monitor_time = time.time()

                    if len(collector) > 0:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Status update
                # 状态更新
                obs_data = _obs_data
                extra_info = _extra_info
    except Exception as e:
        logger.error(f"run_episodes error")
        raise RuntimeError(f"run_episodes error")
