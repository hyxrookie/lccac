import numpy as np
from wandb import agent
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class BxyPostureReward(BaseRewardFunction):
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
        self.range_version = getattr(self.config, f'{self.__class__.__name__}_range_version', 'v3')
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 3.0)

        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        self.range_fn = self.get_range_funtion(self.range_version)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]

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
        return self._process(new_reward, agent_id, (orientation_reward, range_reward))

    def get_orientation_function(self, version):

        # 定义惩罚因子
        def apply_penalty(AO, TA, reward):
            penalty_factor = 1.0  # 默认没有惩罚
            if AO > np.pi / 2 and TA > np.pi / 2:
                penalty_factor = 0.5  # 惩罚因子，可以根据需要调整
            return reward * penalty_factor

        if version == 'v0':
            return lambda AO, TA: (1. - np.tanh(9 * (AO - np.pi / 9))) / 3. + 1 / 3. \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        elif version == 'v1':
            return lambda AO, TA: (1. - np.tanh(2 * (AO - np.pi / 2))) / 2. \
                * (np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi) + 0.5
        elif version == 'v2':
            return lambda AO, TA: 1 / (50 * AO / np.pi + 2) + 1 / 2 \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5

        # def part1(AO):
        #     return ((1 - np.tanh(9 * (AO - np.pi / 9))) / 3 + 1 / 3) * (1 - AO / np.pi)
        #
        # # Part 2: min((arctanh(1 - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0) + 0.5
        # def part2(TA):
        #     values = (np.arctanh(1 - np.maximum(2 * TA / np.pi, 1e-4)) / (np.pi / 4))
        #     values = np.minimum(values, 0)  # Ensure values are within the range
        #     return values + 1

        # todo``
        elif version == 'v3':
            def weight_calculator(AO, TA):
                if AO <= np.pi / 3 and TA <= np.pi / 3:  # 追击敌机
                    return 1.25, 1
                elif AO > 5 * np.pi / 6 and TA > 5 * np.pi / 6:  # 被追击
                    return 1.25, 1
                # elif (AO > np.pi / 2 and TA <= np.pi / 2) or (AO <= np.pi / 2 and TA > np.pi / 2):  # 对头
                #     return 1.25, 1
                else:  # 背离
                    return 1, 1

            def AO_reward(AO):
                return (1. - np.tanh(2 * (AO - np.pi / 2))) / 2.   # v1 版本的AO
                # return ((1. - np.tanh(9 * (AO - np.pi / 9))) / 3. + 1 / 3.) * (1 - AO / np.pi)

            def TA_reward(TA):
                return (np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi) + 0.5
                # return min((np.arctanh(1. - np.maximum(2 * TA / np.pi, 1e-4))) / (np.pi / 4), 0.) + 1

            return lambda AO, TA: (
                    AO_reward(AO) * weight_calculator(AO, TA)[0] * TA_reward(TA) * weight_calculator(AO, TA)[1] + 0.5
            )

            # return lambda AO, TA: ((1. - np.tanh(9 * (AO - np.pi / 9))) / 3. + 1 / 3.) * (1 - AO / np.pi) \
            #     + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (np.pi / 4), 0.) + 1

        elif version == 'v4':
            return lambda AO, TA: apply_penalty(AO, TA,
                                                (1. - np.tanh(2 * (AO - np.pi / 2))) / 2. *
                                                (np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi) + 0.5
                                                )

        elif version == 'v5':
            # 定义影响系数
            alpha1 = 2
            alpha2 = 0.5
            return lambda AO, TA: np.exp(-np.abs(alpha1 * AO) - np.abs(alpha2 * TA))

        elif version == 'v6':
            return lambda AO, TA: 1 - (np.abs(AO) + np.abs(TA)) / (2 * np.pi)

        elif version == 'v7':
            return lambda AO, TA: (
                        (180 - TA) * (1 - np.exp((AO - 90) / 10)) / (180 * (1 + np.exp((AO - 90) / 10))) - TA / (
                            180 * (1 + np.exp(-(AO - 90) / 10))))

        else:
            raise NotImplementedError(f"Unknown orientation function version: {version}")

    def get_range_funtion(self, version):
        if version == 'v0':
            return lambda R: np.exp(-(R - self.target_dist) ** 2 * 0.004) / (1. + np.exp(-(R - self.target_dist + 2) * 2))
        elif version == 'v1':
            return lambda R: np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                     (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1)
        elif version == 'v2':
            return lambda R: max(np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                         (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1), np.sign(7 - R))
        elif version == 'v3':
            return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)

        elif version == 'v4':
            return lambda R: np.exp(-1 * (R - 3)) if R > 3 else 1
        else:
            raise NotImplementedError(f"Unknown range function version: {version}")

