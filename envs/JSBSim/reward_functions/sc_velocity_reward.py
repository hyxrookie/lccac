import numpy as np
from .reward_function_base import BaseRewardFunction
import math
from ..core.catalog import Catalog as c

change = 0.3048

class ScVelocityReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        ego_real_speed = env.agents[agent_id].get_property_value(c.velocities_vt_fps)
        V_0 = 270
        reward = math.exp(-(ego_real_speed-V_0)/V_0)
        return self._process(reward,agent_id)




