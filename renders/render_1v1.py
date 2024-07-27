import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
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
ego_policy_index = '4'
enm_policy_index = '4'
episode_rewards = 0
ego_run_dir = r"D:\lcCAC\cac\scripts\results\SingleCombat\1v1\ShootMissile\HierarchySelfplay\ppo\v1\wandb\run-20240726_170502-exr3533w\files"
enm_run_dir = r"D:\lcCAC\cac\scripts\results\SingleCombat\1v1\ShootMissile\HierarchySelfplay\ppo\v1\wandb\run-20240726_170502-exr3533w\files"
experiment_name = ego_run_dir.split('\\')[-4]
file_name = ego_run_dir.split('\\')[-2]

env = SingleCombatEnv("1v1/ShootMissile/HierarchySelfplay")
env.seed(0)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
ego_policy.eval()
enm_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))

ego_spend_time = 0
ego_lunch_missile_number = 0
ego_distance = 0
ego_missile_hit_count = 0
ego_clash_count = 0
ego_shot_down_count = 0
ego_been_shot_down_count = 0

enm_spend_time = 0
enm_lunch_missile_number = 0
enm_distance = 0
enm_missile_hit_count = 0
enm_clash_count = 0
enm_shot_down_count = 0
enm_been_shot_down_count = 0

start = 0
end = 10
Result_statistics = {}
print("Start render")
while start != end:
    start += 1
    episode_rewards = 0
    obs = env.reset()
    if render:
        env.render(mode='txt',
                   filepath=f'{experiment_name}_task_{ego_policy_index}vs{enm_policy_index}_in_({file_name})_version_{start}.txt.acmi')
    ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((num_agents // 2, 1))
    enm_obs =  obs[num_agents // 2:, :]
    ego_obs =  obs[:num_agents // 2, :]
    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
    while True:
        ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
        ego_actions = _t2n(ego_actions)
        ego_rnn_states = _t2n(ego_rnn_states)
        enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
        enm_actions = _t2n(enm_actions)
        enm_rnn_states = _t2n(enm_rnn_states)
        actions = np.concatenate((ego_actions, enm_actions), axis=0)
        # Obser reward and next obs
        obs, rewards, dones, infos = env.step(actions)
        for sim in env.agents.values():
            if sim.uid.__eq__("A0100"):
                ego_distance += env.combat_distance(sim.uid)
            else:
                break
        rewards = rewards[:num_agents // 2, ...]

        episode_rewards += rewards
        if render:
            env.render(mode='txt',
                       filepath=f'{experiment_name}_task_{ego_policy_index}vs{enm_policy_index}_in_({file_name})_version_{start}.txt.acmi')
        if dones.all():
            for sim in env.agents.values():
                if sim.uid.__eq__("A0100"):
                    ego_spend_time += infos["current_step"]
                    ego_lunch_missile_number +=len(sim.launch_missiles)
                    for missile in sim.launch_missiles:
                        ego_missile_hit_count += int(missile.is_success)
                    ego_clash_count += int(sim.is_crash)
                    ego_shot_down_count += int(sim.enemies[0].is_shotdown)
                    ego_been_shot_down_count += int(sim.is_shotdown)
                if sim.uid.__eq__("B0100"):
                    enm_spend_time += infos["current_step"]
                    enm_lunch_missile_number += len(sim.launch_missiles)
                    for missile in sim.launch_missiles:
                        enm_missile_hit_count += int(missile.is_success)
                    enm_clash_count += int(sim.is_crash)
                    enm_shot_down_count += int(sim.enemies[0].is_shotdown)
                    enm_been_shot_down_count += int(sim.is_shotdown)
                if start == end:
                    if sim.uid.__eq__("A0100"):
                        print({sim.uid: {
                            "avg_combat_time": ego_spend_time/end,
                            "missile_hits_rate": ego_missile_hit_count/ego_lunch_missile_number,
                            "avg_distance": ego_distance / ego_spend_time,
                            "shoot_down_rate": ego_shot_down_count/end,
                            "survival_rate": (end-ego_clash_count-ego_been_shot_down_count)/end
                        }})
                    else:
                        print({sim.uid: {
                            "avg_combat_time": enm_spend_time / end,
                            "missile_hits_rate": enm_missile_hit_count / enm_lunch_missile_number,
                            "avg_distance": ego_distance / ego_spend_time,
                            "shoot_down_rate": enm_shot_down_count / end,
                            "survival_rate": (end - enm_clash_count - enm_been_shot_down_count) / end
                        }})
            print(infos)
            env._create_records = False
            break
        bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
        print(f"step:{env.current_step}, bloods:{bloods}")
        enm_obs =  obs[num_agents // 2:, ...]
        ego_obs =  obs[:num_agents // 2, ...]




    print(episode_rewards)