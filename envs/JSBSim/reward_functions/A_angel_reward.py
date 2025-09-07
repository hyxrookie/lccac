import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R
import math




class AngelReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        reward = 0
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            ow,qw = self.get_ow_qw(AO)
            reward += pow(self.get_TO(AO), ow)*pow(self.get_Tq(AO), qw)
        return self._process(reward,agent_id)

    def get_ow_qw(self, o):
        if 60 < abs(o) <= 180:
            ow = 0.4
            qw = 1-ow
        elif 35 < abs(o) <= 60:
            ow = 0.5
            qw = 1-ow
        elif 20 < abs(o) <= 35:
            ow = 0.66
            qw = 1-ow
        elif 0 < abs(o) <= 20:
            ow = 0.75
            qw = 1-ow
        return ow, qw

    def get_TO(self,attack_angel):
        msa = math.radians(60)
        maa = math.radians(35)
        cea = math.radians(20)
        AO = math.radians(attack_angel)
        if msa < abs(AO) <= math.pi:
            TO = 0.1 - (abs(AO)-msa) / 10*(math.pi - msa)
        elif maa < abs(AO) <= msa:
            TO = 0.3 - (abs(AO)-maa) / 10*(msa - maa)
        elif cea < abs(AO) <= maa:
            TO = 0.8 - (abs(AO)-cea) / 10*(maa - cea)
        elif 0 < abs(AO) <= cea:
            TO = 1 - abs(AO)/5*cea
        return TO

    def get_Tq(self,GO_angel):
        TA = math.radians(GO_angel)
        if math.pi / 3 < abs(TA) <= math.pi:
            Tq = math.exp(-(abs(TA)-math.pi)/2*math.pi)
        elif 0 < abs(TA) <= math.pi / 3:
            Tq = math.exp(-(math.pi/2-abs(TA))/(math.pi/3))
        return Tq

