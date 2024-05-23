import torch
import torch.nn as nn
from typing import Union, List
from .ppo_policy import PPOPolicy
from ..utils.buffer import ReplayBuffer
from ..utils.utils import check, get_gard_norm


# "ppo_trainer.py":
#    - 这个文件包含了PPO算法的训练器（Trainer）类，名为`PPOTrainer`。
#    - `PPOTrainer`类负责实现PPO算法的训练过程。
#    - 它定义了PPO算法中的各种配置参数，如PPO迭代次数、剪切参数、值函数损失的系数等。
#    - `PPOTrainer`类管理了PPO策略（Policy）对象、回放缓冲区等实用工具。
#    - 其主要功能是通过与环境交互、采集数据、计算损失函数并更新策略网络和值函数网络来进行PPO算法的训练。
class PPOTrainer():
    def __init__(self, args, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        # ppo config
        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # rnn configs
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length

    def ppo_update(self, policy: PPOPolicy, sample):#核心部分。每次更新策略时都会调用这个方法。

        obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps 调用策略的evaluate_actions方法计算动作的概率、值函数预测和策略熵
        values, action_log_probs, dist_entropy = policy.evaluate_actions(obs_batch,
                                                                         rnn_states_actor_batch,
                                                                         rnn_states_critic_batch,
                                                                         actions_batch,
                                                                         masks_batch)

        # Obtain the loss function 损失函数计算
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
        policy_loss = -policy_loss.mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2)
        value_loss = value_loss.mean()

        policy_entropy_loss = -dist_entropy.mean()

        loss = policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef

        # Optimize the loss function 反向传播和优化
        policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.optimizer.step()

        return policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm

    def train(self, policy: PPOPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]):
       #PPO算法的训练主循环。它负责从回放缓冲区中获取训练数据样本，并调用ppo_update方法进行参数更新
       #初始化一个字典train_info，用于存储训练过程中的各种损失和梯度信息。
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):#进行多次PPO迭代（次数由ppo_epoch决定）
            if self.use_recurrent_policy:
                #如果使用循环策略（recurrent policy），则从缓冲区中生成数据样本。
                # 这里使用了ReplayBuffer类的recurrent_generator方法，按照num_mini_batch和data_chunk_length生成小批量数据。
                data_generator = ReplayBuffer.recurrent_generator(buffer, self.num_mini_batch, self.data_chunk_length)
            else:
                raise NotImplementedError

            for sample in data_generator:#对每一个生成的数据样本，调用ppo_update方法进行PPO更新。ppo_update方法返回各种损失值和梯度信息。

                policy_loss, value_loss, policy_entropy_loss, ratio, \
                    actor_grad_norm, critic_grad_norm = self.ppo_update(policy, sample)
                #将每次更新返回的损失值和梯度信息累加到train_info字典中。
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()
        #计算总的更新次数，即ppo_epoch和num_mini_batch的乘积。
        num_updates = self.ppo_epoch * self.num_mini_batch
        #将累加的损失值和梯度信息除以总的更新次数，以获得平均值。
        for k in train_info.keys():
            train_info[k] /= num_updates
        #返回包含平均损失和梯度信息的字典train_info。
        return train_info

# 1. `PPOTrainer`类的构造函数`__init__`接受参数`args`和`device`，其中`args`包含了PPO算法的配置参数，
# `device`表示所使用的设备（默认为CPU）。
# 2. `PPOTrainer`类中定义了PPO算法的各种配置参数，包括PPO迭代次数、剪切参数、值函数损失的系数等。
# 3. `ppo_update`方法是PPO算法的核心更新函数。它接受一个`PPOPolicy`对象和采样数据`sample`作为输入。
# `sample`包含了观测数据、动作数据、回报数据等。
# 4. 在`ppo_update`方法中，首先将采样数据转换为PyTorch张量，并将其移动到指定的设备上。
# 5. 然后，通过调用`policy.evaluate_actions`方法计算当前策略网络在给定观测数据和动作数据下的动作对数概率、值函数和熵。
# 6. 接下来，根据PPO算法的损失函数公式，计算策略损失（policy_loss）、值函数损失（value_loss）和策略熵损失（policy_entropy_loss）。
# 7. 将这些损失函数组合成总体损失函数，并根据总体损失函数进行梯度更新。
# 8. 如果设置了`use_max_grad_norm`参数为True，则对梯度进行裁剪以防止梯度爆炸。
# 9. 最后，返回策略损失、值函数损失、策略熵损失、动作概率比率、演员梯度范数和评论家梯度范数等信息。

# 10. `train`方法用于执行整个PPO算法的训练过程。它接受一个`PPOPolicy`对象和一个回放缓冲区（`ReplayBuffer`）作为输入。
# 11. 在`train`方法中，首先初始化训练信息（`train_info`）的各个字段。
# 12. 然后，根据配置参数进行PPO算法的多次迭代。
# 13. 如果使用循环策略（`use_recurrent_policy`为True），则调用`ReplayBuffer.recurrent_generator`方法生成循环数据的生成器。
# 14. 对于每个生成的数据样本，调用`ppo_update`方法进行PPO算法的更新，并累加各项训练信息。
# 15. 最后，将训练信息除以总更新次数，得到平均值，并返回该训练信息。
# 总体而言，这个文件定义了PPO算法的训练器类`PPOTrainer`，包括PPO算法的配置参数、核心更新函数和训练过程。
# 通过与环境交互、采集数据、计算损失函数并更新策略网络和值函数网络来进行PPO算法的训练。