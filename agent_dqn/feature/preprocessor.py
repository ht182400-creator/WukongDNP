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

        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        # End point information
        # 终点信息
        is_end_pos_found = False
        self.end_pos = self.cur_pos
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] == 4 and organ["status"] != -1:
                is_end_pos_found = True
                self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])

        self.feature_end_pos = self._get_pos_feature(is_end_pos_found, self.cur_pos, self.end_pos)

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
        # Reward weight
        # 奖励权重
        step_weight = 10
        buff_weight = 0
        treasure_weight = 0
        end_weight = 0.1

        step_no_norm = min(self.step_no_norm + 0.2, 1)

        # Step reward
        # 每步奖励
        step_reward = step_weight * (self.visited_norm - self.last_visited_norm)

        # Buff reward
        # Buff奖励
        buff_reward = buff_weight * self.buff_get

        # Treasure reward
        # 宝藏奖励
        treasure_reward = treasure_weight * self.treasure_get * (1 - step_no_norm)

        # End reward
        # 终点奖励
        end_reward = end_weight * (1 - self.feature_end_pos[-1]) * step_no_norm

        return [step_reward, treasure_reward, end_reward, buff_reward]
