import random

import gymnasium
import torch
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Any, Tuple
from gymnasium import spaces
from algorithms.ppo.ppo_actor import PPOActor
from ..core.simulatior import AircraftSimulator, BaseSimulator
from ..tasks.task_base import BaseTask
from ..utils.utils import parse_config,get_AO_TA_R,LLA2NEU
from ..core.catalog import Catalog as c
import xlwt
class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True

class BaseEnv(gymnasium.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    An BaseEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.
    """
    metadata = {"render.modes": ["human", "txt"]}

    def __init__(self, config_name: str):
        # basic args
        self.missile_agent = None
        self.Pre_SE = 0
        self.swaprun = 0
        self.height_dealt_max = 1
        self.previous_pitch = 0
        self.previous_roll = 0
        self.previous_side_flage = 0
        self.previous_TA = 0
        self.current_step = 0
        self.ego_sum_SE = 0
        self.enm_sum_SE = 0
        self.EAWR = {"A0100":0,"B0100":0}
        self.previous_height_dealt = 0
        self.previous_dis_max = 1
        self.previous_distance = 0
        self.previous_AO = 0
        self.previous_V = 0
        self.config = parse_config(config_name)
        self.max_steps = getattr(self.config, 'max_steps', 100)  # type: int
        self.sim_freq = getattr(self.config, 'sim_freq', 60)  # type: int
        self.agent_interaction_steps = getattr(self.config, 'agent_interaction_steps', 12)  # type: int
        self.center_lon, self.center_lat, self.center_alt = \
            getattr(self.config, 'battle_field_center', (123.4, 26.0, 0.0))
        self._create_records = False
        self.args = Args()
        self.load()
        # self.wk = xlwt.Workbook(encoding='utf8')
        # self.worksheet = self.wk.add_sheet('Sheet2')
        # self.worksheet.write(0, 0, 'step')
        # self.worksheet.write(0, 1, '距离')
        # self.worksheet.write(0, 2, '距离变化')
        # self.worksheet.write(0, 3, '角度变化')
        # self.worksheet.write(0, 4, '方位角')
        # self.worksheet.write(0, 5, '偏航角优势奖励')
        # self.worksheet.write(0, 6, '攻击窗口奖励')
        # self.worksheet.write(0, 7, '比能量')
        # self.worksheet.write(0, 8, '比能量变化')
        # self.worksheet.write(0, 9, '理想比能量')
        # self.worksheet.write(0, 10, '能量奖励')
        # self.worksheet.write(0, 11, '敌机AO')
        # self.worksheet.write(0, 12, '敌机AO变化')
        # self.worksheet.write(0, 13, '躲避敌机攻击窗口奖励')
        # self.worksheet.write(0, 14, '事件奖励')
        # self.worksheet.write(0, 0, 'step')
        # self.worksheet.write(0, 1, 'ego altitude')
        # self.worksheet.write(0, 2, 'ego_roll_sin')
        # self.worksheet.write(0, 3, 'ego_roll_cos')
        # self.worksheet.write(0, 4, 'ego_pitch_sin')
        # self.worksheet.write(0, 5, 'ego_pitch_cos')
        # self.worksheet.write(0, 6, 'ego v_n_x')
        # self.worksheet.write(0, 7, 'ego v_e_y')
        # self.worksheet.write(0, 8, 'ego v_d_z')
        # self.worksheet.write(0, 9, 'ego vc')
        # self.worksheet.write(0, 10, 'v body')
        # self.worksheet.write(0, 11, 'height')
        # self.worksheet.write(0, 12, 'AO')
        # self.worksheet.write(0, 13, 'TA')
        # self.worksheet.write(0, 14, 'R')
        # self.worksheet.write(0, 15, 'side_flag')
        # self.worksheet.write(0, 16, 'action 1')
        # self.worksheet.write(0, 17, 'action 2')
        # self.worksheet.write(0, 18, 'action 3')
        # self.worksheet.write(0, 19, 'action 4')


    @property
    def num_agents(self) -> int:
        return self.task.num_agents

    @property
    def observation_space(self) -> gymnasium.Space:
        return self.task.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        return self.task.action_space

    @property
    def agents(self) -> Dict[str, AircraftSimulator]:
        return self._jsbsims

    @property
    def time_interval(self) -> int:
        return self.agent_interaction_steps / self.sim_freq

    def load(self):
        self.load_task()
        # self.load_missile_agent()
        self.load_simulator()
        self.seed()

    def load_task(self):
        self.task = BaseTask(self.config)


    def load_missile_agent(self):
        self.missile_agent = PPOActor(self.args, spaces.Box(low=-10, high=10., shape=(21,)), spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.Discrete(2)]), device=torch.device("cuda"))
        self.missile_agent.load_state_dict(torch.load(r"D:\2023\lc\lcCAC\cac\scripts\results\SingleCombat\1v1\ShootMissile\Selfplay\ppo\missile\all" + f"/actor_499.pt"))
        self.missile_agent.eval()
    def load_simulator(self):
        self._jsbsims = {}     # type: Dict[str, AircraftSimulator]
        for uid, config in self.config.aircraft_configs.items():
            self._jsbsims[uid] = AircraftSimulator(
                uid=uid,
                color=config.get("color", "Red"),
                model=config.get("model", "f16"),
                init_state=config.get("init_state"),
                origin=getattr(self.config, 'battle_field_center', (123.4, 26.0, 0.0)),
                sim_freq=self.sim_freq,
                num_missiles=config.get("missile", 1))
        # Different teams have different uid[0]
        _default_team_uid = list(self._jsbsims.keys())[0][0]
        self.ego_ids = [uid for uid in self._jsbsims.keys() if uid[0] == _default_team_uid]
        self.enm_ids = [uid for uid in self._jsbsims.keys() if uid[0] != _default_team_uid]

        # Link jsbsims
        for key, sim in self._jsbsims.items():
            for k, s in self._jsbsims.items():
                if k == key:
                    pass
                elif k[0] == key[0]:
                    sim.partners.append(s)
                else:
                    sim.enemies.append(s)

        self._tempsims = {}    # type: Dict[str, BaseSimulator]

    def add_temp_simulator(self, sim: BaseSimulator):
        self._tempsims[sim.uid] = sim

    def combat_distance(self,plane_name:str):
        ego_position = self._jsbsims[plane_name].get_position() / 1000
        distance = 0
        for sim in self._jsbsims[plane_name].enemies:
            enm_position = sim.get_position() / 1000
            distance += np.linalg.norm(ego_position-enm_position)
        return distance

    def reset(self) -> np.ndarray:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (np.ndarray): initial observation
        """
        # reset sim
        # self.wk.save('reward_anlysis.xlsx')
        self.previous_V = 0
        self.previous_distance = 0
        self.previous_AO = 0
        self.current_step = 0
        self.previous_dis_max = 1
        self.previous_height_dealt = 0
        self.height_dealt_max = 1
        self.previous_TA = 0
        self.previous_side_flage = 0
        self.Pre_SE = 0
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()
        # reset task
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (np.ndarray): the agents' actions, allow opponent's action input

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}
        # apply actions
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].action_pre = a_action
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        temp_EAWR = self.task.step(self)
        for key,value in temp_EAWR.items():
            self.EAWR[key] = self.EAWR[key]+1 if value else self.EAWR[key]

        obs = self.get_obs()

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]

        ego_obs_list = np.array(self._jsbsims['A0100'].get_property_values(self.task.state_var))
        enm_obs_list = np.array(self._jsbsims['A0100'].enemies[0].get_property_values(self.task.state_var))


        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], 123.4, 26.0, 0.0)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], 123.4, 26.0, 0.0)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])

        self.previous_height_dealt = abs(ego_obs_list[2]-enm_obs_list[2])
        self.previous_V = ego_obs_list[12]
        self.previous_roll = ego_obs_list[3]
        self.previous_pitch = ego_obs_list[4]
        self.Pre_SE = (ego_obs_list[12]**2)/19.62 + ego_obs_list[2]

        self.ego_sum_SE += self.Pre_SE
        self.enm_sum_SE += (enm_obs_list[12]**2)/19.62 + enm_obs_list[2]
        self.previous_AO, self.previous_TA, self.previous_distance, self.previous_side_flage = get_AO_TA_R(ego_feature,
                                                                                                           enm_feature,
                                                                                                           return_side=True)
        return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def get_obs(self):
        """Returns all agent observations in a list.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        return dict([(agent_id, self.task.get_obs(self, agent_id)) for agent_id in self.agents.keys()])

    def get_state(self):
        """Returns the global state.

        NOTE: This functon should not be used during decentralised execution.
        """
        state = np.hstack([self.task.get_obs(self, agent_id) for agent_id in self.agents.keys()])
        return dict([(agent_id, state.copy()) for agent_id in self.agents.keys()])

    def close(self):
        """Cleans up this environment's objects

        NOTE: Environments automatically close() when garbage collected or when the
        program exits.
        """
        for sim in self._jsbsims.values():
            sim.close()
        for sim in self._tempsims.values():
            sim.close()
        self._jsbsims.clear()
        self._tempsims.clear()

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some

        environments do not support rendering at all.) By convention,

        if mode is:

        - human: print on the terminal
        - txt: output to txt.acmi files

        Note:

            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        :param mode: str, the mode to render with
        """
        if mode == "txt":
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.current_step * self.time_interval
                f.write(f"#{timestamp:.2f}\n")
                for sim in self._jsbsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
                for sim in self._tempsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
        # TODO: real time rendering [Use FlightGear, etc.]
        else:
            raise NotImplementedError

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _pack(self, data: Dict[str, Any]) -> np.ndarray:
        """Pack seperated key-value dict into grouped np.ndarray"""
        ego_data = np.array([data[uid] for uid in self.ego_ids])
        enm_data = np.array([data[uid] for uid in self.enm_ids])
        if enm_data.shape[0] > 0:
            data = np.concatenate((ego_data, enm_data))  # type: np.ndarray
        else:
            data = ego_data  # type: np.ndarray
        try:
            assert np.isnan(data).sum() == 0
        except AssertionError:
            import pdb
            pdb.set_trace()
        # only return data that belongs to RL agents
        return data[:self.num_agents, ...]

    def _unpack(self, data: np.ndarray) -> Dict[str, Any]:
        """Unpack grouped np.ndarray into seperated key-value dict"""
        assert isinstance(data, (np.ndarray, list, tuple)) and len(data) == self.num_agents
        # unpack data in the same order to packing process
        unpack_data = dict(zip((self.ego_ids + self.enm_ids)[:self.num_agents], data))
        # fill in None for other not-RL agents
        for agent_id in (self.ego_ids + self.enm_ids)[self.num_agents:]:
            unpack_data[agent_id] = None
        return unpack_data
