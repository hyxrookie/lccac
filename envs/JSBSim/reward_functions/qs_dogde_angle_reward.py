import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R
import math




class QsDodgeAngelReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        reward = 0

        enm_AO_previous = math.pi - env.previous_TA
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            enm_AO = math.radians(180) - TA
            if enm_AO_previous > enm_AO:
                reward = -1
            elif enm_AO_previous == enm_AO:
                reward = 0
            else:
                reward = 1

        return reward*0.2

