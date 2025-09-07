import numpy as np
from .reward_function_base import BaseRewardFunction
import math
from ..core.catalog import Catalog as c



class HeightReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)



    def get_reward(self, task, env, agent_id):
        ego_z = env.agents[agent_id].get_position()[-1]
        enm_z = env.agents[agent_id].enemies[0].get_position()[-1]
        ideal_height_range = 6000
        reward = 0
        if ideal_height_range <= ego_z:
            reward = math.exp(-(ego_z-ideal_height_range)/ideal_height_range)
        elif enm_z <= ego_z < ideal_height_range:
            reward = math.exp((ego_z-ideal_height_range)/enm_z)
        elif 0.6*enm_z <= ego_z < enm_z:
            reward = ego_z/enm_z - 0.5
        else:
            reward = 0.1

        return self._process(reward,agent_id)






