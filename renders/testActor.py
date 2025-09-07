import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
from gymnasium import spaces
import pandas as pd


def _t2n(x):
    return x.detach().cpu().numpy()


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


obs_data_dir = r"D:\2023\lc\lcCAC\cac\obs\obs1.csv"

obs_data = pd.read_csv(obs_data_dir)

obs = obs_data.loc[:, '1':"15"]

ego_policy_index = 170
ego_run_dir = r"D:\2023\lc\lcCAC\cac\scripts\results\SingleCombat\1v1\NoWeapon\Selfplay\ppo\v2\wandb\run-20241015_213409-gl0gyvdz\files"
env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
args = Args()
ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
ego_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))

ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((1, 1))
ego_obs = np.zeros((1, 15),dtype=np.float32)

max_rows = 999
count = 0
while True:
    for i in range(15):
        ego_obs[0][i] = obs.iloc[count][i]
    # print(f'ego_obs {ego_obs}')
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
    print(f'count {count} ego_actions {ego_actions}')
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    count += 1
    if count == max_rows:
        break