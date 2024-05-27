import numpy as np
from wandb import agent
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class PostureReward(BaseRewardFunction):
    """
    PostureReward = Orientation * Range
    - Orientation: Encourage pointing at enemy fighter, punish when is pointed at.
    - Range: Encourage getting closer to enemy fighter, punish if too far away.

    NOTE:
    - Only support one-to-one environments.
    """
    def __init__(self, config):
        super().__init__(config)
        self.orientation_version = getattr(self.config, f'{self.__class__.__name__}_orientation_version', 'v2')
        #从配置文件 config 中读取 PostureReward_orientation_version 参数，如果没有设置，默认为 'v2'。
        self.range_version = getattr(self.config, f'{self.__class__.__name__}_range_version', 'v3')
        #从配置文件 config 中读取 PostureReward_range_version 参数，如果没有设置，默认为 'v3'。
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 3.0)
        #从配置文件 config 中读取 PostureReward_target_dist 参数，如果没有设置，默认为 3.0。
        self.orientation_fn = self.get_orientation_function(self.orientation_version) #根据 orientation_version 初始化方向奖励函数，注意，self.orientation_fn接受AT，TA两个参数，并返回计算结果
        self.range_fn = self.get_range_funtion(self.range_version) #根据 range_version 初始化范围奖励函数。
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]
        #生成奖励项名称列表，便于标识和记录不同奖励项，reward_item_names 列表包含 PostureReward、PostureReward_orn、PostureReward_range 三个元素，分别对应总奖励、方向奖励和范围奖励。

    def get_reward(self, task, env, agent_id):
        """
        Reward is a complex function of AO, TA and R in the last timestep.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        new_reward = 0
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                    enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            orientation_reward = self.orientation_fn(AO, TA)
            range_reward = self.range_fn(R / 1000)
            new_reward += orientation_reward * range_reward
        return self._process(new_reward, agent_id, (orientation_reward, range_reward))#处理奖励和内部变量

    def get_orientation_function(self, version):#根据版本号返回不同的函数
        if version == 'v0':
            #lambda关键字用于创建匿名函数，接受的参数是AO，TA，并最后返回这个函数
            return lambda AO, TA: (1. - np.tanh(9 * (AO - np.pi / 9))) / 3. + 1 / 3. \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        elif version == 'v1':
            return lambda AO, TA: (1. - np.tanh(2 * (AO - np.pi / 2))) / 2. \
                * (np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi) + 0.5
        elif version == 'v2':
            return lambda AO, TA: 1 / (50 * AO / np.pi + 2) + 1 / 2 \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        else:
            raise NotImplementedError(f"Unknown orientation function version: {version}")

    def get_range_funtion(self, version):
        if version == 'v0':
            return lambda R: np.exp(-(R - self.target_dist) ** 2 * 0.004) / (1. + np.exp(-(R - self.target_dist + 2) * 2))
        #np.exp(X)计算e^x
        elif version == 'v1':
            return lambda R: np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                     (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1)
        elif version == 'v2':
            return lambda R: max(np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                         (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1), np.sign(7 - R))
        #np.sign(X)返回X计算结果的符号值1，-1，0
        elif version == 'v3':
            return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)
        else:
            raise NotImplementedError(f"Unknown range function version: {version}")
