import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R
import math


class ScAngelReward(BaseRewardFunction):
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
            if AO <= self.R_max_Angle_threshold:
                reward = (abs(AO-self.R_max_Angle_threshold)/2*self.R_max_Angle_threshold)*math.exp((TA-math.pi)/math.pi)
            else:
                reward = -(abs(AO-self.R_max_Angle_threshold)/2*self.R_max_Angle_threshold)*math.exp((TA-math.pi)/math.pi)
        return self._process(reward, agent_id)
