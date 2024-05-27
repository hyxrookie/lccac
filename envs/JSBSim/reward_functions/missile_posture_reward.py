import numpy as np
from .reward_function_base import BaseRewardFunction


class MissilePostureReward(BaseRewardFunction):
    """
    MissilePostureReward
    Use the velocity attenuation
    """
    def __init__(self, config):
        super().__init__(config)
        self.previous_missile_v = None

    def reset(self, task, env):
        self.previous_missile_v = None
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        Reward is velocity attenuation of the missile

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:#如果接收到导弹警告
            missile_v = missile_sim.get_velocity()
            aircraft_v = env.agents[agent_id].get_velocity()
            if self.previous_missile_v is None:
                self.previous_missile_v = missile_v
            v_decrease = (np.linalg.norm(self.previous_missile_v) - np.linalg.norm(missile_v)) / 340 * self.reward_scale #导弹的速度衰减率
            angle = np.dot(missile_v, aircraft_v) / (np.linalg.norm(missile_v) * np.linalg.norm(aircraft_v)) #angle是导弹速度向量与飞机（我方？）速度矢量的夹角θ的余弦
            #dot是 NumPy 库中的一个函数，用于计算两个向量的点积
            if angle < 0:#夹角【90.180】
                reward = angle / (max(v_decrease, 0) + 1)
            else:#夹角【0，90】
                reward = angle * max(v_decrease, 0)
        else:#如果没有导弹警告
            self.previous_missile_v = None
            reward = 0
        self.reward_trajectory[agent_id].append([reward])
        return reward
