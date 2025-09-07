import numpy as np
from .reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R



class QsDistanceReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)



    def get_reward(self, task, env, agent_id):
        ego_obs_list = np.array(env._jsbsims[agent_id].get_property_values(task.state_var))
        enm_obs_list = np.array(env._jsbsims[agent_id].enemies[0].get_property_values(task.state_var))
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], 123.4, 26.0, 0.0)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], 123.4, 26.0, 0.0)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        AO, TA, R_dis, side_flage = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        dealt_dis = R_dis-env.previous_distance
        env.previous_dis_max = 1

        reward = (dealt_dis)/env.previous_dis_max
        env.height_dealt_max = max(env.height_dealt_max,abs(dealt_dis))


        return reward*0.3






