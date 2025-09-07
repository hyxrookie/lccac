import numpy as np

from .reward_function_base import BaseRewardFunction
from ..utils.utils import LLA2NEU


class ScMissileLockedReward(BaseRewardFunction):
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
        enm_id = env._jsbsims[agent_id].enemies[0].uid
        reward = 0
        if np.sum(task.lock_duration[agent_id]) >= task.lock_duration[agent_id].maxlen and task.min_missile_attack_distance <= task.R_dis <= task.max_missile_attack_distance:
            reward += 10
        if np.sum(task.lock_duration[enm_id]) >= task.lock_duration[enm_id].maxlen and task.min_missile_attack_distance <= task.R_dis <= task.max_missile_attack_distance:
            reward -= 10


        # if agent_id == "A0100" and env.current_step != 0:
        #     env.worksheet.write(env.current_step, 14, reward)

        return reward
