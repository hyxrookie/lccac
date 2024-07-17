import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict


class BaseRewardFunction(ABC):
    """
    Base RewardFunction class
    Reward-specific reset and get_reward methods are implemented in subclasses
    """
    def __init__(self, config):
        self.config = config
        # inner variables
        self.reward_scale = getattr(self.config, f'{self.__class__.__name__}_scale', 1.0)
        #  使用 getattr 函数尝试从 self.config 对象中获取名为 self.__class__._scale 的属性值。
        #  如果 self.config 中存在这个属性，则返回该属性的值；如果不存在，则返回默认值 1.0。
        self.is_potential = getattr(self.config, f'{self.__class__.__name__}_potential', False)
        self.pre_rewards = defaultdict(float)
        self.reward_trajectory = defaultdict(list)
        self.reward_item_names = [self.__class__.__name__]

    def reset(self, task, env):
        """Perform reward function-specific reset after episode reset.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance
        """
        if self.is_potential:
            self.pre_rewards.clear()
            for agent_id in env.agents.keys():
                self.pre_rewards[agent_id] = self.get_reward(task, env, agent_id)
        self.reward_trajectory.clear()

    @abstractmethod
    def get_reward(self, task, env, agent_id):
        """Compute the reward at the current timestep.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        raise NotImplementedError

    def _process(self, new_reward, agent_id, render_items=()):
        """Process reward and inner variables.

        Args:
            new_reward (float)
            agent_id (str)
            render_items (tuple, optional): Must set if `len(reward_item_names)>1`. Defaults to None.

        Returns:
            [type]: [description]
        """
        reward = new_reward * self.reward_scale #将新的奖励值乘以 self.reward_scale，得到经过缩放的奖励值 reward。
        if self.is_potential: #进行潜力奖励调整
            reward, self.pre_rewards[agent_id] = reward - self.pre_rewards[agent_id], reward
            #将 reward 减去之前保存的该代理者的上一个奖励值 self.pre_rewards[agent_id]，并将结果赋值给 reward。
            #更新 self.pre_rewards[agent_id] 为当前的 reward。
        self.reward_trajectory[agent_id].append([reward, *render_items])
        # 将奖励值reward以及其他需要渲染的项render_items添加到
        # self.reward_trajectory[agent_id]列表中。这个列表用于记录代理者的奖励轨迹。
        return reward

    def get_reward_trajectory(self):
        """Get all the reward history of current episode.py

        Returns:
            (dict): {reward_name(str): reward_trajectory(np.array)}
        """
        return dict(zip(self.reward_item_names, np.array(self.reward_trajectory.values()).transpose(2, 0, 1)))
