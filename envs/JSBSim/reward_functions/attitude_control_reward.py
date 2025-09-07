import numpy as np

from .reward_function_base import BaseRewardFunction
from ..utils.utils import LLA2NEU, get_AO_TA_R


class AttitudeControlReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """
    def __init__(self, config):
        super().__init__(config)
        self.roll = 0
        self.roll_count = [0 for _ in range(20)]
        self.pitch_count = [0 for _ in range(10)]
        self.pitch = 0

        self.thredroll = 0.7
        self.ar = 0.2


    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward

        """
        ego_obs_list = np.array(env._jsbsims[agent_id].get_property_values(task.state_var))
        enm_obs_list = np.array(env._jsbsims[agent_id].enemies[0].get_property_values(task.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], 123.4, 26.0, 0.0)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], 123.4, 26.0, 0.0)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        AO, TA, R_dis, side_flage = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        rollSelf = ego_obs_list[3]
        pitchSelf = ego_obs_list[4]
        reward = 0
        rolldi = self.ar*rollSelf + (1-self.ar)*env.previous_roll
        pitchdi = self.ar*pitchSelf + (1 - self.ar)*env.previous_pitch
        # if rolldi - self.thredroll < rollSelf < rolldi + self.thredroll:
        #     reward = 1
        # else:
        #     reward = -0.5

        # height = ego_obs_list[2]
        # #俯仰姿态奖励
        # # print(f'俯仰角:{PitchSelf}')
        # if PitchSelf < -0.4 and R_dis > task.max_missile_attack_distance:
        #     self.pitch_count[self.pitch] = 1
        #     # print(f'俯仰队列 {self.pitch_count}')
        #     self.pitch += 1
        #     self.pitch %= 10
        # if all(i == 1 for i in self.pitch_count):
        #     reward -= 300
        #     self.pitch = 0
        #     self.pitch_count = [0 for _ in range(10)]
        #
        #滚转姿态奖励
        if R_dis > task.max_missile_attack_distance and abs(rollSelf) > 1.74:
            self.roll_count[self.roll] = 1
            self.roll += 1
            self.roll %= 20
        if all(item == 1 for item in self.roll_count):
            reward -= 200
            self.roll = 0
            self.roll_count = [0 for _ in range(20)]

        return self._process(reward, agent_id)
