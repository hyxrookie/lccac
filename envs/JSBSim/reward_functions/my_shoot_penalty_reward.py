import math

import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class MyShootPenaltyReward(BaseRewardFunction):
    """
    ShootPenaltyReward
    when launching a missile, give -10 reward for penalty,
    to avoid launching all missiles at once
    """
    theta_mkmax = math.pi/6;
    theta_mmax = math.pi/4;
    D_mkmax = 10000;#瞎编的
    D_mkmin = 6000;
    D_mmax = 14000;
    D_mmin = 4000;

    def __init__(self, config):
        super().__init__(config)

    def reset(self, task, env):
        self.pre_remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        # self.pre_remaining_missiles，记录每个代理的剩余导弹数量。agent.num_missiles表示代理当前拥有的导弹数量。
        return super().reset(task, env)

    def check_relative_information(self, theta, D):#根据相对位置和角度返回不同的值
        if theta > self.theta_mmax or D > self.D_mmax or D < self.D_mmin:
            return 1
        elif self.theta_mkmax < theta <= self.theta_mmax and self.D_mmin<= D <= self.D_mmax:
            return 2
        elif theta <= self.theta_mkmax and self.D_mkmax <= D <= self.D_mmax:
            return 3
        elif theta <= self.theta_mkmax and self.D_mkmin <= D <= self.D_mkmax:
            return 4
        else:#错误情况
            return -1

    def get_reward_base_on_relative_information(self , info,theta,D):
        reward=0;
        if info==1:
            reward=-5
        elif info==2:
            reward=math.exp(-(theta-self.theta_mkmax)/self.theta_mkmax)+\
                   math.exp( - abs(2*D-(self.D_mmax+self.D_mmin))/(self.D_mmin+self.D_mmax) )-2
        elif info==3:
            reward=math.exp(-theta/self.theta_mkmax)+\
                   math.exp(- abs(D-self.D_mkmax)/self.D_mkmax)-2
        elif info==4:
            reward=math.exp(- theta/self.theta_mkmax)+\
                   math.exp(- abs ( 2*D -(self.D_mkmax+self.D_mkmin) ) / (self.D_mkmax+self.D_mkmin)) + 4
        else:#错误情况
            reward=0;
        return reward;

    def get_reward(self, task, env, agent_id):
        reward = 0
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            theta, TA, D = get_AO_TA_R(ego_feature, enm_feature)  # 目标方位角θ，且θ为正，没有负值。D是两机间的距离
            if task.remaining_missiles[agent_id] == self.pre_remaining_missiles[
                agent_id] - 1:  # 如果当前代理的剩余导弹数量比前一次记录的减少了 1，则表示该代理刚刚发射了一枚导弹。
                info = self.check_relative_information(theta,D)
                reward=self.get_reward_base_on_relative_information(info,theta,D)

        self.pre_remaining_missiles[agent_id] = task.remaining_missiles[agent_id]
        # 如果当前代理的剩余导弹数量比前一次记录的减少了 1，则表示该代理刚刚发射了一枚导弹。此时，给予一个 -10 的奖励（即惩罚），以避免代理一次性发射所有导弹。
        return self._process(reward, agent_id)
