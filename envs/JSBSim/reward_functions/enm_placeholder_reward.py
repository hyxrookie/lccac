import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R


class EnmPlaceholderReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.R_max_Angle_threshold = math.radians(70)
        self.AO_threshold = math.radians(90)

    def get_reward(self, task, env, agent_id):

        ego_obs_list = np.array(env._jsbsims[agent_id].get_property_values(task.state_var))
        enm_obs_list = np.array(env._jsbsims[agent_id].enemies[0].get_property_values(task.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], 123.4, 26.0, 0.0)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], 123.4, 26.0, 0.0)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        AO, TA, R_dis, side_flage = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        R_ego_min = task.min_missile_attack_distance
        R_ego_max = task.max_missile_attack_distance
        enm_AO = math.radians(180) - TA
        enm_AO_previous = math.pi - env.previous_TA
        enm_v = enm_obs_list[12]
        ego_v = ego_obs_list[12]
        enm_previous_v = env.previous_V
        enm_previous_dis = env.previous_distance
        reward = 0
        if R_ego_min <= R_dis <= 1.2 * R_ego_max:
            if enm_AO < self.R_max_Angle_threshold:
                if abs(enm_AO) <= abs(enm_AO_previous):
                    reward = -2
                else:
                    reward = 1.5
            elif abs(enm_AO) > self.R_max_Angle_threshold:
                if abs(enm_AO) < abs(enm_AO_previous):
                    reward = -1
                else:
                    reward = 0.5
        elif R_dis < R_ego_min:
            if abs(enm_AO) < self.AO_threshold:
                if abs(AO) <= self.AO_threshold:
                    if enm_v < ego_v:
                        reward = -0.5
                    else:
                        reward = 0.2
                elif abs(AO) > self.AO_threshold:
                    if R_dis <= enm_previous_dis:
                        reward = 0.2
                    elif R_dis > enm_previous_v:
                        reward = -1

        return reward
