import torch
import numpy as np
from typing import Union, List
from abc import ABC, abstractmethod
from .utils import get_shape_from_space


class Buffer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def insert(self, **kwargs):
        pass

    @abstractmethod
    def after_update(self):
        pass

    @abstractmethod
    def clear(self):
        pass


class ReplayBuffer(Buffer):

    @staticmethod
    def _flatten(T: int, N: int, x: np.ndarray):#将输入的多维数组 x 进行重塑，以便将前两个维度 T 和 N 合并成一个维度
        return x.reshape(T * N, *x.shape[2:])

    @staticmethod
    def _cast(x: np.ndarray):#对输入的多维数组 x 进行转置和重塑，使其形状便于处理。
        return x.transpose(1, 2, 0, *range(3, x.ndim)).reshape(-1, *x.shape[3:])

    def __init__(self, args, num_agents, obs_space, act_space):
        #初始化 ReplayBuffer 实例，并为各种数据（观测、动作、奖励等）分配存储空间。
        #虽然每一个线程都有敌我两架飞机，但是缓冲区只存储A号飞机的数据，B号飞机不会存储
        # buffer config
        self.buffer_size = args.buffer_size #设置缓冲区的大小，表示最多能存储多少步的经验数据
        self.n_rollout_threads = args.n_rollout_threads #设置并行环境的数量，每个环境独立生成经验数据。
        self.num_agents = num_agents #设置代理的数量，表示在多代理环境中有多少个代理
        self.gamma = args.gamma #设置折扣因子，用于计算回报时对未来奖励的折扣。
        self.use_proper_time_limits = args.use_proper_time_limits #标志是否使用正确的时间限制处理终止状态。？？
        self.use_gae = args.use_gae #标志是否使用广义优势估计（GAE）来计算优势函数。
        self.gae_lambda = args.gae_lambda #设置GAE的lambda参数，用于控制优势估计的偏差与方差。
        # rnn config
        self.recurrent_hidden_size = args.recurrent_hidden_size #设置RNN的隐藏层大小。
        self.recurrent_hidden_layers = args.recurrent_hidden_layers #设置RNN隐藏层的数量。

        #从观测空间和动作空间中获取其形状，以便为这些数据分配正确大小的存储空间。
        obs_shape = get_shape_from_space(obs_space)
        act_shape = get_shape_from_space(act_space)

        # (o_0, a_0, r_0, d_1, o_1, ... , d_T, o_T)
        #初始化存储观测值的数组。形状为 (self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *obs_shape)，其中+1是为了存储最新的观测值。
        self.obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32)
        #初始化存储动作的数组。形状为 (self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape)。
        self.actions = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        #初始化存储奖励的数组。形状为 (self.buffer_size, self.n_rollout_threads, self.num_agents, 1)。
        self.rewards = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # NOTE: masks[t] = 1 - dones[t-1], which represents whether obs[t] is a terminal state
        #初始化存储掩码的数组。掩码用于指示某个状态是否为终止状态。形状为 (self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1)。
        self.masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # NOTE: bad_masks[t] = 'bad_transition' in info[t-1], which indicates whether obs[t] a true terminal state or time limit end state
        #初始化存储坏掩码的数组。坏掩码用于指示状态是否因时间限制而终止。形状为 (self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1)。
        self.bad_masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # pi(a)
        #初始化存储动作对数概率的数组。形状为 (self.buffer_size, self.n_rollout_threads, self.num_agents, 1)。
        self.action_log_probs = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # V(o), R(o) while advantage = returns - value_preds
        #初始化存储价值预测的数组。形状为 (self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1)。
        self.value_preds = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        #初始化存储回报的数组。形状为 (self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1)。
        self.returns = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # rnn
        #初始化存储RNN状态的数组，分别用于演员和评论家。
        # 形状为 (self.buffer_size + 1, self.n_rollout_threads, self.num_agents, self.recurrent_hidden_layers, self.recurrent_hidden_size)。
        self.rnn_states_actor = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents,
                                          self.recurrent_hidden_layers, self.recurrent_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)

        #初始化步数计数器，用于记录当前存储数据的位置。
        self.step = 0

    @property
    def advantages(self) -> np.ndarray:
        advantages = self.returns[:-1] - self.value_preds[:-1]  # type: np.ndarray
        return (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def insert(self,
               obs: np.ndarray,
               actions: np.ndarray,
               rewards: np.ndarray,
               masks: np.ndarray,
               action_log_probs: np.ndarray,
               value_preds: np.ndarray,
               rnn_states_actor: np.ndarray,
               rnn_states_critic: np.ndarray,
               bad_masks: Union[np.ndarray, None] = None,
               **kwargs):
        """Insert numpy data.
        Args:
            obs:                o_{t+1}
            actions:            a_{t}
            rewards:            r_{t}
            masks:              mask[t+1] = 1 - done_{t}
            action_log_probs:   log_prob(a_{t})
            value_preds:        value(o_{t})
            rnn_states_actor:   ha_{t+1}
            rnn_states_critic:  hc_{t+1}
            obs (np.ndarray)：时间步 t+1 时的观测值。
            actions (np.ndarray)：时间步 t 时的动作。
            rewards (np.ndarray)：时间步 t 时的奖励。
            masks (np.ndarray)：时间步 t+1 时的掩码，用于指示是否是终止状态，通常为 1 - done[t]。
            action_log_probs (np.ndarray)：时间步 t 时动作的对数概率。
            value_preds (np.ndarray)：时间步 t 时的价值预测。
            rnn_states_actor (np.ndarray)：时间步 t+1 时演员的 RNN 状态。
            rnn_states_critic (np.ndarray)：时间步 t+1 时评论家的 RNN 状态。
            bad_masks (np.ndarray, optional)：时间步 t+1 时的坏掩码，用于指示状态是否因时间限制而终止。
        """
        self.obs[self.step + 1] = obs.copy() #将时间步 t+1 时的观测值插入到 obs 数组中的当前位置 self.step + 1。使用 copy() 确保数据是独立的副本，避免原始数据被修改。
        self.actions[self.step] = actions.copy() #将时间步 t 时的动作插入到 actions 数组中的当前位置 self.step。
        self.rewards[self.step] = rewards.copy() #将时间步 t 时的奖励插入到 rewards 数组中的当前位置 self.step。
        self.masks[self.step + 1] = masks.copy() #时间步 t+1 时的掩码插入到 masks 数组中的当前位置 self.step + 1。
        self.action_log_probs[self.step] = action_log_probs.copy() #将时间步 t 时动作的对数概率插入到 action_log_probs 数组中的当前位置 self.step。
        self.value_preds[self.step] = value_preds.copy() #将时间步 t 时的价值预测插入到 value_preds 数组中的当前位置 self.step。
        self.rnn_states_actor[self.step + 1] = rnn_states_actor.copy() #将时间步 t+1 时演员的 RNN 状态插入到 rnn_states_actor 数组中的当前位置 self.step + 1。
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy() #将时间步 t+1 时评论家的 RNN 状态插入到 rnn_states_critic 数组中的当前位置 self.step + 1。
        if bad_masks is not None:#如果提供了坏掩码，将时间步 t+1 时的坏掩码插入到 bad_masks 数组中的当前位置 self.step + 1。
            self.bad_masks[self.step + 1] = bad_masks.copy()

        #更新步数计数器。通过取模操作确保步数在缓冲区大小内循环，防止步数超出缓冲区大小。
        self.step = (self.step + 1) % self.buffer_size

    def after_update(self):
        #在模型更新后，将最后一个时间步的数据复制到第一个索引位置，以便为下一轮数据采集做好准备。
        """Copy last timestep data to first index. Called after update to model."""
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.rnn_states_actor[0] = self.rnn_states_actor[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()

    def clear(self):#清除缓冲区中的所有数据，并重置步数。
        self.step = 0
        self.obs = np.zeros_like(self.obs, dtype=np.float32)
        self.actions = np.zeros_like(self.actions, dtype=np.float32)
        self.rewards = np.zeros_like(self.rewards, dtype=np.float32)
        self.masks = np.ones_like(self.masks, dtype=np.float32)
        self.bad_masks = np.ones_like(self.bad_masks, dtype=np.float32)
        self.action_log_probs = np.zeros_like(self.action_log_probs, dtype=np.float32)
        self.value_preds = np.zeros_like(self.value_preds, dtype=np.float32)
        self.returns = np.zeros_like(self.returns, dtype=np.float32)
        self.rnn_states_actor = np.zeros_like(self.rnn_states_critic)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)

    def compute_returns(self, next_value: np.ndarray):
        """
        计算回报（returns），可以使用折扣奖励或广义优势估计（GAE）的方法。
        Compute returns either as discounted sum of rewards, or using GAE.

        Args:
            next_value(np.ndarray): value predictions for the step after the last episode step.
        """
        if self.use_proper_time_limits:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    td_delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = td_delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) \
                        * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    td_delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = td_delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    @staticmethod
    def recurrent_generator(buffer: Union[Buffer, List[Buffer]], num_mini_batch: int, data_chunk_length: int):
        """
        生成用于训练的迷你批次数据，特别是为RNN训练生成分块的数据。
        A recurrent generator that yields training data for chunked RNN training arranged in mini batches.
        This generator shuffles the data by sequences.

        Args:
            buffers (Buffer or List[Buffer])
            num_mini_batch (int): number of minibatches to split the batch into.
            data_chunk_length (int): length of sequence chunks with which to train RNN.

        Returns:
            (obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch)
        """
        buffer = [buffer] if isinstance(buffer, ReplayBuffer) else buffer  # type: List[ReplayBuffer]
        n_rollout_threads = buffer[0].n_rollout_threads
        buffer_size = buffer[0].buffer_size
        num_agents = buffer[0].num_agents
        assert all([b.n_rollout_threads == n_rollout_threads for b in buffer]) \
            and all([b.buffer_size == buffer_size for b in buffer]) \
            and all([b.num_agents == num_agents for b in buffer]) \
            and all([isinstance(b, ReplayBuffer) for b in buffer]), \
            "Input buffers must has the same type and shape"
        buffer_size = buffer_size * len(buffer)

        assert n_rollout_threads * buffer_size >= data_chunk_length, (
            "PPO requires the number of processes ({}) * buffer size ({}) * num_agents ({})"
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, buffer_size, num_agents, data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = np.vstack([ReplayBuffer._cast(buf.obs[:-1]) for buf in buffer])
        actions = np.vstack([ReplayBuffer._cast(buf.actions) for buf in buffer])
        masks = np.vstack([ReplayBuffer._cast(buf.masks[:-1]) for buf in buffer])
        old_action_log_probs = np.vstack([ReplayBuffer._cast(buf.action_log_probs) for buf in buffer])
        advantages = np.vstack([ReplayBuffer._cast(buf.advantages) for buf in buffer])
        returns = np.vstack([ReplayBuffer._cast(buf.returns[:-1]) for buf in buffer])
        value_preds = np.vstack([ReplayBuffer._cast(buf.value_preds[:-1]) for buf in buffer])
        rnn_states_actor = np.vstack([ReplayBuffer._cast(buf.rnn_states_actor[:-1]) for buf in buffer])
        rnn_states_critic = np.vstack([ReplayBuffer._cast(buf.rnn_states_critic[:-1]) for buf in buffer])

        # Get mini-batch size and shuffle chunk data
        data_chunks = n_rollout_threads * buffer_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            obs_batch = []
            actions_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            advantages_batch = []
            returns_batch = []
            value_preds_batch = []
            rnn_states_actor_batch = []
            rnn_states_critic_batch = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1, N, Dim] => [T, N, Dim] => [N, T, Dim] => [N * T, Dim] => [L, Dim]
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(old_action_log_probs[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                returns_batch.append(returns[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                # size [T+1, N, Dim] => [T, N, Dim] => [N, T, Dim] => [N * T, Dim] => [1, Dim]
                rnn_states_actor_batch.append(rnn_states_actor[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            advantages_batch = np.stack(advantages_batch, axis=1)
            returns_batch = np.stack(returns_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *buffer[0].rnn_states_actor.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *buffer[0].rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = ReplayBuffer._flatten(L, N, obs_batch)
            actions_batch = ReplayBuffer._flatten(L, N, actions_batch)
            masks_batch = ReplayBuffer._flatten(L, N, masks_batch)
            old_action_log_probs_batch = ReplayBuffer._flatten(L, N, old_action_log_probs_batch)
            advantages_batch = ReplayBuffer._flatten(L, N, advantages_batch)
            returns_batch = ReplayBuffer._flatten(L, N, returns_batch)
            value_preds_batch = ReplayBuffer._flatten(L, N, value_preds_batch)

            yield obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch


class SharedReplayBuffer(ReplayBuffer):

    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        # env config
        self.num_agents = num_agents
        self.n_rollout_threads = args.n_rollout_threads
        # buffer config
        self.gamma = args.gamma
        self.buffer_size = args.buffer_size
        self.use_proper_time_limits = args.use_proper_time_limits
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        # rnn config
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers

        obs_shape = get_shape_from_space(obs_space)
        share_obs_shape = get_shape_from_space(share_obs_space)
        act_shape = get_shape_from_space(act_space)

        # (o_0, s_0, a_0, r_0, d_0, ..., o_T, s_T)
        self.obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32)
        self.share_obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *share_obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # NOTE: masks[t] = 1 - dones[t-1], which represents whether obs[t] is a terminal state .... same for all agents
        self.masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        # NOTE: active_masks[t, :, i] represents whether agent[i] is alive in obs[t] .... differ in different agents
        self.active_masks = np.ones_like(self.masks)
        # pi(a)
        self.action_log_probs = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        # V(o), R(o) while advantage = returns - value_preds
        self.value_preds = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # rnn
        self.rnn_states_actor = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents,
                                          self.recurrent_hidden_layers, self.recurrent_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)

        self.step = 0

    def insert(self,
               obs: np.ndarray,
               share_obs: np.ndarray,
               actions: np.ndarray,
               rewards: np.ndarray,
               masks: np.ndarray,
               action_log_probs: np.ndarray,
               value_preds: np.ndarray,
               rnn_states_actor: np.ndarray,
               rnn_states_critic: np.ndarray,
               bad_masks: Union[np.ndarray, None] = None,
               active_masks: Union[np.ndarray, None] = None,
               available_actions: Union[np.ndarray, None] = None):
        """Insert numpy data.
        Args:
            obs:                o_{t+1}
            share_obs:          s_{t+1}
            actions:            a_{t}
            rewards:            r_{t}
            masks:              1 - done_{t}
            action_log_probs:   log_prob(a_{t})
            value_preds:        value(o_{t})
            rnn_states_actor:   ha_{t+1}
            rnn_states_critic:  hc_{t+1}
            active_masks:       1 - agent_done_{t}
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            pass
        return super().insert(obs, actions, rewards, masks, action_log_probs, value_preds, rnn_states_actor, rnn_states_critic)

    def after_update(self):
        self.active_masks[0] = self.active_masks[-1].copy()
        self.share_obs[0] = self.share_obs[-1].copy()
        return super().after_update()

    def recurrent_generator(self, advantages: np.ndarray, num_mini_batch: int, data_chunk_length: int):
        """
        A recurrent generator that yields training data for chunked RNN training arranged in mini batches.
        This generator shuffles the data by sequences.

        Args:
            advantages (np.ndarray): advantage estimates.
            num_mini_batch (int): number of minibatches to split the batch into.
            data_chunk_length (int): length of sequence chunks with which to train RNN.

        Returns:
            (obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                rnn_states_actor_batch, rnn_states_critic_batch)
        """
        assert self.n_rollout_threads * self.buffer_size >= data_chunk_length, (
            "PPO requires the number of processes ({}) * buffer size ({}) "
            "to be greater than or equal to the number of data chunk length ({}).".format(
                self.n_rollout_threads, self.buffer_size, data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = self._cast(self.obs[:-1])
        share_obs = self._cast(self.share_obs[:-1])
        actions = self._cast(self.actions)
        masks = self._cast(self.masks[:-1])
        active_masks = self._cast(self.active_masks[:-1])
        old_action_log_probs = self._cast(self.action_log_probs)
        advantages = self._cast(advantages)
        returns = self._cast(self.returns[:-1])
        value_preds = self._cast(self.value_preds[:-1])
        rnn_states_actor = self._cast(self.rnn_states_actor[:-1])
        rnn_states_critic = self._cast(self.rnn_states_critic[:-1])

        # Get mini-batch size and shuffle chunk data
        data_chunks = self.n_rollout_threads * self.buffer_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            obs_batch = []
            share_obs_batch = []
            actions_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            advantages_batch = []
            returns_batch = []
            value_preds_batch = []
            rnn_states_actor_batch = []
            rnn_states_critic_batch = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1, N, M, Dim] => [T, N, M, Dim] => [N, M, T, Dim] => [N * M * T, Dim] => [L, Dim]
                obs_batch.append(obs[ind:ind + data_chunk_length])
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(old_action_log_probs[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                returns_batch.append(returns[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                # size [T+1, N, M, Dim] => [T, N, M, Dim] => [N, M, T, Dim] => [N * M * T, Dim] => [1, Dim]
                rnn_states_actor_batch.append(rnn_states_actor[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            advantages_batch = np.stack(advantages_batch, axis=1)
            returns_batch = np.stack(returns_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *self.rnn_states_actor.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = self._flatten(L, N, obs_batch)
            share_obs_batch = self._flatten(L, N, share_obs_batch)
            actions_batch = self._flatten(L, N, actions_batch)
            masks_batch = self._flatten(L, N, masks_batch)
            active_masks_batch = self._flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = self._flatten(L, N, old_action_log_probs_batch)
            advantages_batch = self._flatten(L, N, advantages_batch)
            returns_batch = self._flatten(L, N, returns_batch)
            value_preds_batch = self._flatten(L, N, value_preds_batch)

            yield obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                rnn_states_actor_batch, rnn_states_critic_batch
