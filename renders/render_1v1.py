import random
import os
import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
import time
from gymnasium import spaces
logging.basicConfig(level=logging.DEBUG)

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


def _t2n(x):
    return x.detach().cpu().numpy()

num_agents = 2
render = True
ego_policy_index = random.randint(1040,1240)
enm_policy_index = random.randint(1040,1240)
# ego_m_policy_index = 545



episode_rewards = 0
ego_run_dir = r"D:\2023\lc\lcCAC\cac\scripts\results\SingleCombat\1v1\NoWeapon\HierarchySelfplay\ppo\v2\my_last"
# ego_run_dir = r"D:\2023\lc\lcCAC\cac\scripts\results\SingleCombat\1v1\ShootMissile\HierarchySelfplay\ppo\v1\wandb\run-20240927_165046-i2ar26ds\files"
# enm_run_dir = r"D:/CAC/CloseAirCombat/scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/Ronew_self_orient_add_range_dyn_reward_pfsp/wandb/run-20250317_125922-l2vkwjgy/files"
enm_run_dir = r"D:\2023\lc\lcCAC\cac\scripts\results\SingleCombat\1v1\NoWeapon\HierarchySelfplay\ppo\v2\no_energy"
# ego_m_run_dir = r"D:\2023\lc\lcCAC\cac\scripts\results\SingleCombat\1v1\ShootMissile\Selfplay\ppo\missile"


experiment_name = ego_run_dir.split('\\')[-4]
file_name = ego_run_dir.split('\\')[-2]


env = SingleCombatEnv("1v1/NoWeapon/HierarchySelfplay")
env.seed(1)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
# ego_m_policy=PPOActor(args, spaces.Box(low=-10, high=10., shape=(21,)), spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.Discrete(2)]), device=torch.device("cuda"))#这里得改
# enm_policy = PPOActor(args, env.observation_space, spaces.MultiDiscrete([41, 41, 41, 30]), device=torch.device("cuda"))

enm_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))
# ego_m_policy.load_state_dict(torch.load(ego_m_run_dir+f"/actor_{ego_m_policy_index}.pt"))

ego_spend_time = 0
ego_SE = 0
ego_EAWR = 0
ego_enm_launch_num = 0
ego_enm_dodge_num = 0
ego_launch_num = 0
ego_hit_num = 0

enm_spend_time = 0
enm_SE = 0
enm_EAWR = 0
enm_ego_launch_num = 0
enm_ego_dodge_num = 0
enm_launch_num = 0
enm_hit_num = 0
win = 0


start = 0
end = 1000
Result_statistics = {}
print("Start render")
timestamp = time.time()
egoname = ego_run_dir.split('\\')[-1]
enmname = enm_run_dir.split('\\')[-1]
new_dir = f"{egoname}_vs_{enmname}_{timestamp}"
os.makedirs(new_dir)

while start != end:
    ego_policy_index = random.randint(1040, 1240)
    enm_policy_index = random.randint(1040, 1240)
    ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
    enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))
    print(f'这是第{start}回合')
    start += 1
    episode_rewards = 0
    obs = env.reset()
    if render:
        env.render(mode='txt',
                   filepath=f'./{new_dir}/{experiment_name}_task_{ego_policy_index}vs{enm_policy_index}_in_({file_name})_version_{start}.txt.acmi')
    ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((num_agents // 2, 1))
    enm_obs =  obs[num_agents // 2:, :]
    ego_obs =  obs[:num_agents // 2, :]

    # print(ego_obs_m)

    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
    while True:
        ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
        #导弹

        # print(f'ego_actions {ego_actions}')
        ego_actions = _t2n(ego_actions)
        ego_rnn_states = _t2n(ego_rnn_states)
        enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)

        enm_actions = _t2n(enm_actions)
        # enm_actions = np.append(enm_actions, 0)
        # enm_actions = enm_actions[np.newaxis, :]
        # print(ego_actions)
        # print(enm_actions)
        enm_rnn_states = _t2n(enm_rnn_states)
        actions = np.concatenate((ego_actions, enm_actions), axis=0)
        # Obser reward and next obs

        obs, rewards, dones, infos = env.step(actions)

        rewards = rewards[:num_agents // 2, ...]

        episode_rewards += rewards
        if render:
            env.render(mode='txt',
                       filepath=f'./{new_dir}/{experiment_name}_task_{ego_policy_index}vs{enm_policy_index}_in_({file_name})_version_{start}.txt.acmi')
        if dones.all():
            for sim in env.agents.values():
                if sim.uid.__eq__("A0100"):
                    ego_spend_time += infos["current_step"]
                    ego_SE = env.ego_sum_SE / infos["current_step"]
                    ego_EAWR = env.EAWR[sim.uid] / infos["current_step"]
                    for missile in sim.enemies[0].launch_missiles:
                        ego_enm_launch_num += 1
                        if missile.is_miss:
                            ego_enm_dodge_num += 1
                    for missile in sim.launch_missiles:
                        ego_launch_num += 1
                        if missile.is_success:
                            ego_hit_num += 1
                    if sim.is_alive and sim.enemies[0].is_alive == False:
                        win += 1

                if sim.uid.__eq__("B0100"):
                    enm_spend_time += infos["current_step"]
                    enm_SE = env.enm_sum_SE / infos["current_step"]
                    enm_EAWR = env.EAWR[sim.uid] / infos["current_step"]
                    for missile in sim.enemies[0].launch_missiles:
                        enm_ego_launch_num += 1
                        if missile.is_miss:
                            enm_ego_dodge_num += 1
                    for missile in sim.launch_missiles:
                        enm_launch_num += 1
                        if missile.is_success == 1:
                            enm_hit_num += 1
                if start == end:
                    if sim.uid.__eq__("A0100"):
                        print({sim.uid: {
                            "avg_combat_time": ego_spend_time/end,
                            "ego_avg_SE": ego_SE/end,
                            "ego_avg_EAWR": ego_EAWR/end,
                            "ego_dogde_missile_rate": ego_enm_dodge_num/ego_enm_launch_num ,
                            "ego_hit_missile_rate": ego_hit_num/ego_launch_num ,
                            "win_rate": win/end
                        }})
                    else:
                        print({sim.uid: {
                            "enm_avg_SE": enm_SE/end,
                            "enm_avg_EAWR": enm_EAWR/end,
                            "enm_dogde_missile_rate": enm_ego_dodge_num/enm_ego_launch_num,
                            "enm_hit_missile_rate": enm_hit_num/enm_launch_num
                        }})
            print(infos)
            env._create_records = False
            break
        bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
        print(f"step:{env.current_step}, bloods:{bloods}")
        # if env.current_step == 500:
        #     break
        enm_obs = obs[num_agents // 2:, ...]
        ego_obs = obs[:num_agents // 2, :]


    # env.wk.save('reward_analysis.xlsx')
    print(episode_rewards)