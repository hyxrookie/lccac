import numpy as np
from .reward_function_base import BaseRewardFunction
import math
from ..core.catalog import Catalog as c



class HeightReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)



    def get_reward(self, task, env, agent_id):
        ego_z = env.agents[agent_id].get_position()[-1] / 1000
        enm_z = env.agents[agent_id].enemies[0].get_position()[-1] / 1000
        opt_height = task.optimal_air_combat_height
        Th=0
        if ego_z >= opt_height:
            Th = math.exp(-(ego_z-opt_height)/opt_height)
        elif  enm_z <= ego_z < opt_height:
            Th = math.exp((ego_z-opt_height)/enm_z)
        elif 0.6*ego_z <= enm_z < enm_z:
            Th = ego_z/enm_z - 0.5
        elif ego_z < 0.6*enm_z:
            Th = 0.1

        if ego_z > 1.5*opt_height:
            return self._process(-Th, agent_id)
        elif 0.4*opt_height < ego_z <= 1.5*opt_height:
            return self._process(Th, agent_id)
        else:
            return self._process((5*ego_z - opt_height) / (9*opt_height) * Th, agent_id)




