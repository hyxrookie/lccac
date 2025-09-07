import numpy as np
from .reward_function_base import BaseRewardFunction
import math
from ..core.catalog import Catalog as c


class DwHeightReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        ego_z = env.agents[agent_id].get_position()[-1]
        ideal_height_range = [4500, 7500]
        reward = 0
        if ideal_height_range[0] <= ego_z <= ideal_height_range[1]:
            reward = 1
        else:
            reward = -1

        return self._process(0.2*reward, agent_id)
