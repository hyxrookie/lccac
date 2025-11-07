import numpy as np

from .reward_function_base import BaseRewardFunction
from ..utils.utils import LLA2NEU
from ..core.catalog import Catalog as c

class ScEventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """
    def __init__(self, config):
        super().__init__(config)
        self.missile_counts = 10


    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        ego_z = env.agents[agent_id].get_position()[-1]
        ego_real_speed = env.agents[agent_id].get_property_value(c.velocities_vt_fps)
        reward = 0
        if ego_z < 200:
            reward -= 10
        if ego_z>20000:
            reward -= 10
        if ego_real_speed > 300:
            reward -= 10
        if ego_real_speed <50:
            reward -= 10

        return self._process(reward,agent_id)
