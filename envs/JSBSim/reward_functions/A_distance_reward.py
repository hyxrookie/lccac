import numpy as np
from .reward_function_base import BaseRewardFunction
import math
from ..utils.utils import get_AO_TA_R

class DistanceReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)


    def get_reward(self, task, env, agent_id):
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        reward = 0
        Drmax=100000
        Dmmax=25000
        Dmmin=8000
        Dmkmax=20000
        Dmkmin=10000

        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            AO, TA, D = get_AO_TA_R(ego_feature, enm_feature)

            # info = self.check_relative_information(task,D)
            if(Drmax<D):
                reward = 0.2 * math.exp(-(D-Drmax)/Drmax);
            elif(Dmmax<D<=Drmax):
                reward = 0.5*math.exp(-(D-Dmmax)/(Drmax-Dmmax))
            elif (Dmkmax<D<=Dmmax) and (Dmkmin<D<=Dmkmax):
                reward = pow(2,-(D-Dmkmax)/(Dmmax-Dmkmax))
            elif(Dmmin<=D<=Dmkmax):
                reward = pow(2,-(D-Dmkmin)/(Dmmin-Dmkmin))
            else:
                reward = 0

        return self._process(reward,agent_id)



