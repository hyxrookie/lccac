import numpy as np

from .reward_function_base import BaseRewardFunction
from ..utils.utils import LLA2NEU


class DwEventDrivenReward(BaseRewardFunction):
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
        # ego_obs_list = np.array(env._jsbsims[agent_id].get_property_values(task.state_var))
        reward = 0
        if env.agents[agent_id].is_shotdown:
            reward -= 50
        elif env.agents[agent_id].is_crash:
            reward -= 100

        for missile in env.agents[agent_id].launch_missiles:
            if missile.is_success:
                reward += 100
        return self._process(reward,agent_id)
