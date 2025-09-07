import numpy as np
from .reward_function_base import BaseRewardFunction
import math
from ..core.catalog import Catalog as c



class ScHeightReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        ego_z = env.agents[agent_id].get_position()[-1]
        enm_z = env.agents[agent_id].enemies[0].get_position()[-1]
        ideal_height_range = 2000
        deal_heigth = abs(ego_z-enm_z)
        reward = 0
        if ideal_height_range <= deal_heigth:
            reward = math.exp(-(deal_heigth - ideal_height_range)/ideal_height_range)
        else:
            reward = math.exp((deal_heigth - ideal_height_range)/ideal_height_range)

        return self._process(reward,agent_id)






