import numpy as np
from .reward_function_base import BaseRewardFunction


class AltitudeReward(BaseRewardFunction):
    """
    AltitudeReward
    Punish if current fighter doesn't satisfy some constraints. Typically negative.
    - Punishment of velocity when lower than safe altitude   (range: [-1, 0])
    - Punishment of altitude when lower than danger altitude (range: [-1, 0])
    """
    def __init__(self, config):
        ##调用父类的初始化方法，将配置参数传递给父类，以便父类进行一些必要的初始化操作。
        super().__init__(config)
        #从配置文件中获取安全高度参数，如果配置文件中没有设置，则默认值为4.0千米。这个值用于确定何时施加速度惩罚。
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)         # km
        #从配置文件中获取危险高度参数，如果配置文件中没有设置，则默认值为3.5千米。这个值用于确定何时施加高度惩罚。
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)     # km
        #从配置文件中获取速度惩罚系数参数，如果配置文件中没有设置，则默认值为0.2。这个系数用于调整速度惩罚的程度。
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)     # mh
        #设置奖励项的名称，用于在记录奖励信息时标识各项奖励的来源。这里包括了总的奖励项以及两个子项（速度惩罚项和高度惩罚项）。
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        ego_z = env.agents[agent_id].get_position()[-1] / 1000    # unit: km
        ego_vz = env.agents[agent_id].get_velocity()[-1] / 340    # unit: mh
        Pv = 0.
        if ego_z <= self.safe_altitude:
            Pv = -np.clip(ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude, 0., 1.)
        PH = 0.
        if ego_z <= self.danger_altitude:
            PH = np.clip(ego_z / self.danger_altitude, 0., 1.) - 1. - 1.
        new_reward = Pv + PH
        return self._process(new_reward, agent_id, (Pv, PH))
