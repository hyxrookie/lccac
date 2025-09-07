import random

import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R


class EnergyReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.v1 = 2
        self.v2 = 1

    def get_reward(self, task, env, agent_id):

        ego_obs_list = np.array(env._jsbsims[agent_id].get_property_values(task.state_var))
        enm_obs_list = np.array(env._jsbsims[agent_id].enemies[0].get_property_values(task.state_var))
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], 123.4, 26.0, 0.0)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], 123.4, 26.0, 0.0)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        AO, TA, R_dis, side_flage = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        R_ego_min = task.min_missile_attack_distance
        R_ego_max = task.max_missile_attack_distance
        ego_v = ego_obs_list[12]
        ego_high = ego_obs_list[2]
        SE = self.cal_SE(ego_v, ego_high)
        dealt_SE = SE - env.Pre_SE
        remaining = 1 if task.remaining_missiles[agent_id] > 0 else 0

        if abs(dealt_SE) > 3000:
            dealt_SE = 0

        if remaining:
            if R_dis < R_ego_min:
                energy = math.exp(-abs(SE - self.cal_SE(238, 4500)) / self.cal_SE(238, 4500))
            elif R_ego_min < R_dis <= 0.5 * R_ego_max:
                energy = math.exp(-abs(SE - self.cal_SE(270, 6000)) / self.cal_SE(270, 6000))
            elif 0.5 * R_ego_max < R_dis <= 0.8 * R_ego_max:
                energy = math.exp(-abs(SE - self.cal_SE(300, 7500)) / self.cal_SE(300, 7500))
            elif 0.8 * R_ego_max < R_dis <= 1.5 * R_ego_max:
                energy = math.exp(-abs(SE - self.cal_SE(320, 9000)) / self.cal_SE(320, 9000))
            else:
                energy = math.exp(-abs(SE - self.cal_SE(340, 10500)) / self.cal_SE(340, 10500))
            if energy < 0.8:
                reward = max(dealt_SE / 10, -1) * self.v1
            else:
                reward = max(dealt_SE / 10, -1) * self.v2
        else:
            energy = math.exp(-abs(SE - self.cal_SE(340, 12000)) / self.cal_SE(340, 12000))
            reward = energy + max(dealt_SE / 10, -1)

        # if agent_id == "A0100":
        #     env.worksheet.write(env.current_step, 7, SE)
        #     env.worksheet.write(env.current_step, 8, dealt_SE)
        #     env.worksheet.write(env.current_step, 9, energy)
        #     env.worksheet.write(env.current_step, 10, reward)

        return reward

    def cal_SE(self, v, h):
        return (v ** 2) / 19.62 + h
