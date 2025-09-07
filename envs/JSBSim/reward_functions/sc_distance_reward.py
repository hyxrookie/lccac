import numpy as np
from .reward_function_base import BaseRewardFunction
import math
from ..utils.utils import get_AO_TA_R

class ScDistanceReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)


    def get_reward(self, task, env, agent_id):
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        reward = 0
        Dmkmax=80000

        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            AO, TA, D = get_AO_TA_R(ego_feature, enm_feature)

            # info = self.check_relative_information(task,D)
            if(D<Dmkmax):
                reward = 1
            else:
                reward = math.exp(-(Dmkmax-D)/Dmkmax)

        return self._process(reward,agent_id)



