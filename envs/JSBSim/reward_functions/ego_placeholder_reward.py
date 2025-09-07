import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU,get_AO_TA_R


class EgoPlaceholderReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.R_max_Angle_threshold = math.radians(60)
        self.AO_threshold = math.radians(90)


    def get_reward(self, task, env, agent_id):

        ego_obs_list = np.array(env._jsbsims[agent_id].get_property_values(task.state_var))
        enm_obs_list = np.array(env._jsbsims[agent_id].enemies[0].get_property_values(task.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], 123.4, 26.0, 0.0)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], 123.4, 26.0, 0.0)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        AO, TA, R_dis, side_flage = get_AO_TA_R(ego_feature,enm_feature,return_side=True)
        R_ego_min = task.min_missile_attack_distance
        R_ego_max = task.max_missile_attack_distance
        enm_AO = math.radians(180)-TA
        ego_v = ego_obs_list[12]
        enm_v = enm_obs_list[12]
        ego_previous_v = env.previous_V
        # print(f'count:{env.current_step},previous:{ego_previous_v},now:{ego_v}')
        ego_previous_dis = env.previous_distance
        reward = 0
        if R_ego_min <= R_dis <= 1.1*R_ego_max:
            if abs(AO) < self.R_max_Angle_threshold:
                if abs(AO) <= abs(env.previous_AO):
                    reward = 1.5
                else:
                    reward = -0.3
            elif abs(AO) > self.R_max_Angle_threshold:
                if abs(AO) < abs(env.previous_AO):
                    reward = 0.75
                else:
                    reward = -1.5
        elif R_dis > 1.1*R_ego_max:
            if R_dis > ego_previous_dis:
                reward = -4
            else:
                reward = 1
        else:
            if R_dis <= ego_previous_dis:
                if abs(AO) <= self.AO_threshold:
                    if abs(enm_AO) > self.AO_threshold:
                        reward = 0
                    else:
                        if ego_v <= enm_v:
                            if ego_v <= ego_previous_v:
                                reward = 0.5
                            else:
                                reward = -0.5
                        else:
                            if ego_v <= ego_previous_v:
                                reward = 0.3
                            else:
                                reward = -1
                else:
                    if abs(enm_AO) <= self.AO_threshold:
                        reward = 0.2
                    else:
                        reward = 0
            else:
                if abs(AO) <= self.AO_threshold < abs(enm_AO):
                    reward = 0.5
                if abs(AO) > self.AO_threshold >= abs(enm_AO):
                    reward = -1

        return reward
