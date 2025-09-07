import numpy as np
from .reward_function_base import BaseRewardFunction
import math
from ..core.catalog import Catalog as c



class QsHeightReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)



    def get_reward(self, task, env, agent_id):
        ego_z = env.agents[agent_id].get_position()[-1]
        enm_z = env.agents[agent_id].enemies[0].get_position()[-1]
        dealt_height = abs(ego_z - enm_z)
        height_dealt = env.previous_height_dealt - dealt_height

        reward = (height_dealt)/env.height_dealt_max
        env.height_dealt_max = max(env.height_dealt_max,height_dealt)


        return reward*0.2






