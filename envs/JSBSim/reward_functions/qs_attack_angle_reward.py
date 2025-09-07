import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R
import math




class QsAttackAngelReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        reward = 0
        last_time_AO = env.previous_AO
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            reward = (AO -last_time_AO)/math.pi
        return reward*0.3

