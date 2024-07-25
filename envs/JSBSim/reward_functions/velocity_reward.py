import numpy as np
from .reward_function_base import BaseRewardFunction
import math
from ..core.catalog import Catalog as c

change = 0.3048

class VelocityReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)



    def get_reward(self, task, env, agent_id):
        ego_real_speed = env.agents[agent_id].get_property_value(c.velocities_vt_fps) * change
        enm_real_speed = env.agents[agent_id].enemies[0].get_property_value(c.velocities_vt_fps) * change
        V_0 = task.optimal_air_combat_speed
        if V_0 > 1.5 * enm_real_speed:
            if ego_real_speed > V_0:
                T_v = math.exp(-(ego_real_speed - V_0)/V_0)
            elif 1.5 * enm_real_speed < ego_real_speed <= V_0:
                T_v = 1
            elif 0.6 * enm_real_speed < ego_real_speed <= 1.5 * enm_real_speed:
                T_v = ego_real_speed / enm_real_speed - 0.5
            elif ego_real_speed < 0.6 * enm_real_speed:
                T_v = 0.1
        else:
            if V_0 < ego_real_speed:
                T_v = math.exp(-(ego_real_speed - V_0)/V_0)
            elif 0.6 * enm_real_speed < ego_real_speed <= V_0:
                T_v = 0.4*(ego_real_speed/enm_real_speed + ego_real_speed/V_0)
            elif ego_real_speed < 0.6 * enm_real_speed:
                T_v = 0.1

        if 0.2*V_0 <= ego_real_speed <= 2*V_0:
            return self._process(T_v, agent_id)
        elif 0.2*V_0 >= ego_real_speed or ego_real_speed >= 2*V_0:
            return self._process(-T_v, agent_id)




