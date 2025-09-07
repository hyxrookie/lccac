import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R
import math


class DwAngelReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.R_max_Angle_threshold = math.radians(60)

    def get_reward(self, task, env, agent_id):
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        reward = 0
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            if AO <= self.R_max_Angle_threshold and TA >= math.pi / 2:
                reward = 1
            elif TA <= math.pi / 2:
                reward = 0
            else:
                reward = -1
        return self._process(0.5*reward, agent_id)
