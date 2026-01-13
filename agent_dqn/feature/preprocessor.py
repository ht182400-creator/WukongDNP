#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import numpy as np


def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 8
        self.skill_action_num = 8
        self.grid_map_size = 128
        self.view_size = 51
        self.visit_view_mask = np.ones((11, 11))

        self.reset()

    def reset(self):
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = (0, 0)

        self.last_pos_norm = np.array((0, 0))
        self.last_action = -1
        self.bad_move_ids = set()
        self.map_data = np.zeros((2, self.grid_map_size, self.grid_map_size)).astype(int)
        self.map_data[0] = np.ones_like(self.map_data[0])

        self.visited_norm = 0
        self.last_visited_norm = 0
        self.speed_up = False
        self.buff_found = False
        self.buff_status = 0
        self.buff_pos = (0, 0)
        self.buff_cooldown = 0

        self.feature_treasures_pos = [self._get_pos_feature(False, (0, 0), (0, 0))] * 8
        self.last_treasure_count = 0
        self.last_buff_count = 0
        self.ray_dist = [0] * 8
        
        # Collision detection variables
        # 碰撞检测变量
        self.last_pos = (-1, -1)
        self.collision_detected = False
        self.last_end_dist = None  # 用于计算距离终点的改进
        self.end_dist_improvement = 0  # 距离改进值

    def update_map(self, map_data, view, pos):
        view_size = view.shape[0]
        l = max(0, pos[0] - view_size // 2)
        r = min(self.grid_map_size, pos[0] + view_size // 2 + 1)
        u = max(0, pos[1] - view_size // 2)
        d = min(self.grid_map_size, pos[1] + view_size // 2 + 1)

        view_x_start = l - (pos[0] - view_size // 2)
        view_x_end = view_x_start + (r - l)
        view_y_start = u - (pos[1] - view_size // 2)
        view_y_end = view_y_start + (d - u)

        cropped_view = view[view_x_start:view_x_end, view_y_start:view_y_end]
        before = np.sum(map_data[l:r, u:d])
        map_data[l:r, u:d] = cropped_view
        after = np.sum(map_data[l:r, u:d])
        return after - before

    def _get_pos_feature(self, found, cur_pos, target_pos):
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        dist_norm = norm(dist, 1.41 * 128) if found else 1
        target_pos_norm = norm(target_pos, 128, -128)
        feature = np.array(
            (
                found,
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
                target_pos_norm[0],
                target_pos_norm[1],
                dist_norm,
            ),
        )
        return feature

    def pb2struct(self, frame_state, last_action):
        obs, env_info = frame_state
        hero = obs["frame_state"]["heroes"][0]

        self.last_action = last_action
        self.last_pos_norm = self.cur_pos_norm

        self.step_no = obs["frame_state"]["step_no"]
        self.step_no_norm = norm(self.step_no, 2000)

        # Collision detection: check if position changed significantly
        # 碰撞检测：检查位置是否显著改变（移动距离小于500视为撞墙）
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        if self.last_pos != (-1, -1):
            dist = ((self.last_pos[0] - self.cur_pos[0]) ** 2 + 
                   (self.last_pos[1] - self.cur_pos[1]) ** 2) ** 0.5
            self.collision_detected = dist <= 500
        else:
            self.collision_detected = False
        self.last_pos = self.cur_pos

        # End point information
        # 终点信息
        is_end_pos_found = False
        self.end_pos = self.cur_pos
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] == 4 and organ["status"] != -1:
                is_end_pos_found = True
                self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])

        self.feature_end_pos = self._get_pos_feature(is_end_pos_found, self.cur_pos, self.end_pos)
        
        # Calculate distance to end point for distance reward
        # 计算到终点的距离，用于距离奖励
        if is_end_pos_found:
            cur_end_dist = self.feature_end_pos[-1]  # normalized distance
            if self.last_end_dist is not None:
                self.end_dist_improvement = self.last_end_dist - cur_end_dist  # 距离减少为正
            else:
                self.end_dist_improvement = 0
            self.last_end_dist = cur_end_dist
        else:
            self.end_dist_improvement = 0
            self.last_end_dist = None

        # Agent information
        # 智能体信息
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)
        self.move_usable, self.skill_usable = True, hero["talent"]["status"] == 1
        self.speed_up = hero["speed_up"]
        self.skill_cd = hero["talent"]["cooldown"]

        # Map information
        # 地图信息
        obstacle_view = np.array([x["values"] for x in obs["map_info"]])
        self.update_map(self.map_data[0], obstacle_view, self.cur_pos)
        self.update_map(self.map_data[1], self.visit_view_mask, self.cur_pos)
        self.last_visited_norm = self.visited_norm
        self.visited_norm = self.map_data[1].mean()

        self.buff_found = False
        self.buff_status = 0
        self.buff_cooldown = 0
        self.buff_pos = self.cur_pos

        # Buff information
        # buff信息
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] == 2 and organ["status"] != -1:
                self.buff_found = True
                self.buff_status = organ["status"] == 1
                self.buff_cooldown = organ["cooldown"]
                self.buff_pos = (organ["pos"]["x"], organ["pos"]["z"])

        self.feature_buff_pos = self._get_pos_feature(self.buff_found, self.cur_pos, self.buff_pos)

        # Global information
        # 全局信息
        self.feature_treasures_pos = [
            self._get_pos_feature(True, self.cur_pos, (x["pos"]["x"], x["pos"]["z"]))
            for x in obs["frame_state"]["organs"]
            if x["sub_type"] == 1 and x["status"] == 1
        ] + [self._get_pos_feature(False, self.cur_pos, self.cur_pos)] * 13

        self.feature_treasures_pos = self.feature_treasures_pos[:13]
        self.feature_treasures_pos = sorted(self.feature_treasures_pos, key=lambda x: x[-1])

        self.treasure_get = obs["score_info"]["treasure_collected_count"] - self.last_treasure_count
        self.last_treasure_count = obs["score_info"]["treasure_collected_count"]

        self.buff_get = obs["score_info"]["buff_count"] - self.last_buff_count
        self.last_buff_count = obs["score_info"]["buff_count"]

    def feature_process(self, frame_state, last_action):
        self.pb2struct(frame_state, last_action)

        # Ray information
        # 射线信息
        map_data = self.map_data[0]
        delta = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
        max_ray = 10
        self.ray_dist = []
        for dx, dy in delta:
            x, y = self.cur_pos
            cnt = -1
            while (cnt == -1) or (
                x >= 0
                and x < self.grid_map_size
                and y >= 0
                and y < self.grid_map_size
                and int(map_data[x][y]) != 0
                and cnt < max_ray
            ):
                x += dx
                y += dy
                cnt += 1
            self.ray_dist.append(cnt)

        # Flash information
        # 空间穿梭信息
        flash_dist = []
        max_flash_dist = 8000 // 500
        for dx, dy in delta:
            ex, ey = self.cur_pos
            ex += dx * max_flash_dist
            ey += dy * max_flash_dist
            cnt = max_flash_dist
            while (cnt == -1) or (
                ex >= 0
                and ex < self.grid_map_size
                and ey >= 0
                and ey < self.grid_map_size
                and int(map_data[ex][ey]) == 0
                and cnt > 0
            ):
                ex -= dx
                ey -= dy
                cnt -= 1
            flash_dist.append(cnt)

        # Legal action
        # 合法动作
        legal_action = self.get_legal_action()

        # Feature
        # 特征
        feature = np.concatenate(
            self.feature_treasures_pos
            + [
                self.feature_buff_pos,
                np.array(
                    (
                        self.buff_status,
                        norm(self.buff_cooldown, 100),
                    ),
                ),
                self.feature_end_pos,
                np.array(legal_action),
                norm(self.ray_dist, max_ray),
                norm(flash_dist, max_flash_dist),
                self.cur_pos_norm,
                np.array(
                    (
                        self.speed_up,
                        self.move_usable,
                        self.skill_usable,
                        norm(self.skill_cd, 100),
                        self.step_no_norm,
                        self.visited_norm,
                        norm(self.last_treasure_count, 13),
                    ),
                ),
                self.map_data[1].reshape(16, 8, 16, 8).mean(axis=(1, 3)).flatten(),
            ],
        )

        return (
            feature,
            legal_action,
            self.reward_process(),
        )

    def get_legal_action(self):
        # if last_action is move and current position is the same as last position, add this action to bad_move_ids
        # 如果上一步的动作是移动，且当前位置与上一步位置相同，则将该动作加入到bad_move_ids中
        if (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
            and self.last_action > -1
        ):
            self.bad_move_ids.add(self.last_action)
        else:
            self.bad_move_ids = set()

        legal_action = [self.move_usable] * self.move_action_num + [self.skill_usable] * self.skill_action_num
        for move_id in self.bad_move_ids:
            legal_action[move_id] = 0

        if self.move_usable not in legal_action:
            self.bad_move_ids = set()
            return [self.move_usable] * self.move_action_num + [self.skill_usable] * self.skill_action_num

        return legal_action

    def reward_process(self):
        # Reward weight - optimized for treasure collection within 2000 steps
        # 奖励权重 - 针对在2000步内尽可能多收集宝箱并到达终点进行优化
        # 策略调整：优先收集宝箱（至少4-5个），在2000步内到达终点即可，不需要尽量少步数
        step_weight = 10  # 探索新区域的奖励
        buff_weight = 0
        treasure_weight = 70  # 大幅提高宝箱奖励权重（从45到70），优先收集宝箱
        end_weight = 3.5  # 保持终点奖励权重，确保能到达终点，但不需要快速到达
        collision_weight = -8  # 碰撞惩罚权重，减少碰撞
        step_penalty_weight = -0.005  # 大幅降低步数惩罚（从-0.025到-0.005），因为不需要尽量少步数
        distance_reward_weight = 1.5  # 距离奖励权重，保持适中，引导向终点移动

        step_no_norm = min(self.step_no_norm + 0.2, 1)

        # Step reward: exploration reward
        # 每步奖励：探索新区域的奖励
        step_reward = step_weight * (self.visited_norm - self.last_visited_norm)

        # Buff reward
        # Buff奖励
        buff_reward = buff_weight * self.buff_get

        # Treasure reward: strongly encourage treasure collection
        # 宝藏奖励：强烈鼓励收集宝箱，优先收集宝箱（至少4-5个）
        # 早期收集奖励更大，但整个过程中都给予较高奖励
        treasure_reward = treasure_weight * self.treasure_get * (1 - step_no_norm * 0.3)

        # End reward: reward for getting closer to end point
        # 终点奖励：接近终点时给予奖励，确保能在2000步内到达终点
        # 不需要快速到达，只要在2000步内到达即可
        end_dist = self.feature_end_pos[-1] if len(self.feature_end_pos) > 0 else 1.0
        end_reward = end_weight * (1 - end_dist) * (0.3 + 0.7 * step_no_norm)  # 步数越多时奖励越大，鼓励在2000步内到达

        # Distance improvement reward: reward for getting closer to end point
        # 距离改进奖励：接近终点时给予奖励，提供即时反馈
        distance_reward = distance_reward_weight * self.end_dist_improvement

        # Collision penalty: penalize collisions to reduce wall hits
        # 碰撞惩罚：惩罚碰撞以减少撞墙
        collision_penalty = collision_weight if self.collision_detected else 0

        # Step penalty: minimal penalty per step
        # 步数惩罚：每步只有很小的负奖励，因为不需要尽量少步数，只要在2000步内到达即可
        step_penalty = step_penalty_weight

        return [step_reward, treasure_reward, end_reward, buff_reward, distance_reward, collision_penalty, step_penalty]
