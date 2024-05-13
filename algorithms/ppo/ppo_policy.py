import torch
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic


 # "ppo_policy.py":
 #   - 这个文件定义了PPO策略（Policy）类，名为`PPOPolicy`。
 #   - `PPOPolicy`类封装了策略网络（`PPOActor`）和值函数网络（`PPOCritic`）。
 #   - 它提供了一系列方法，用于生成动作、计算值函数、评估动作等。
 #   - `PPOPolicy`类还管理了PPO算法中的各种配置参数，如学习率、隐藏层大小等。
 #   - 其主要功能是根据当前状态生成动作，根据策略网络和值函数网络计算损失函数，并提供评估动作的方法。

class PPOPolicy:

# 1. `__init__(self, args, obs_space, act_space, device=torch.device("cpu"))`:
#    - 这是`PPOPolicy`类的构造函数，用于初始化PPO策略对象。
#    - 它接受参数`args`（包含配置参数的对象）、`obs_space`（观测空间）、`act_space`（动作空间）和`device`（设备，默认为CPU）。
#    - 在该函数中，它会初始化学习率、观测空间和动作空间等属性，并创建策略网络对象（`self.actor`）和值函数网络对象（`self.critic`）。
#    - 此外，它还创建了用于优化器的Adam优化器对象（`self.optimizer`），并将策略网络和值函数网络的参数添加到优化器中。
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):

        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = PPOCritic(args, self.obs_space, self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.lr)

# 2. `get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks)`:
#    - 这个函数用于根据给定的观测值和RNN状态获取动作。
#    - 它接受参数`obs`（观测值）、`rnn_states_actor`（策略网络的RNN状态）、`rnn_states_critic`（值函数网络的RNN状态）和`masks`（掩码）。
#    - 在函数内部，它调用策略网络的`actor`方法生成动作，并调用值函数网络的`critic`方法计算值函数。
#    - 它返回值函数、动作、动作的对数概率、策略网络的RNN状态和值函数网络的RNN状态。
    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

# 3. `get_values(self, obs, rnn_states_critic, masks)`:
#    - 这个函数用于根据给定的观测值和RNN状态获取值函数估计值。
#    - 它接受参数`obs`（观测值）、`rnn_states_critic`（值函数网络的RNN状态）和`masks`（掩码）。
#    - 在函数内部，它调用值函数网络的`critic`方法计算值函数。
#    - 它返回值函数估计值。
    def get_values(self, obs, rnn_states_critic, masks):
        """
        Returns:
            values
        """
        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values

# 4. `evaluate_actions(self, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None)`:
#    - 这个函数用于评估给定动作的值函数估计值、动作的对数概率和分布熵。
#    - 它接受参数`obs`（观测值）、`rnn_states_actor`（策略网络的RNN状态）、`rnn_states_critic`（值函数网络的RNN状态）、`action`（动作）、`masks`（掩码）和`active_masks`（活动掩码，默认为None）。
#    - 在函数内部，它调用策略网络的`evaluate_actions`方法计算动作的对数概率和分布熵，并调用值函数网络的`critic`方法计算值函数估计值。
#    - 它返回值函数估计值、动作的对数概率和分布熵。
    def evaluate_actions(self, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None):
        """
        Returns:
            values, action_log_probs, dist_entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

# 5. `act(self, obs, rnn_states_actor, masks, deterministic=False)`:
#    - 这个函数用于根据给定的观测值和RNN状态生成动作。
#    - 它接受参数`obs`（观测值）、`rnn_states_actor`（策略网络的RNN状态）、`masks`（掩码）和`deterministic`（是否使用确定性动作，默认为False）。
#    - 在函数内部，它调用策略网络的`actor`方法生成动作。
#    - 它返回动作和策略网络的RNN状态。
    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor

# 6. `prep_training(self)`:
#    - 这个函数用于准备策略网络和值函数网络进行训练。
#    - 在函数内部，它将策略网络和值函数网络设置为训练模式（`train()`）。
    def prep_training(self):
        self.actor.train()
        self.critic.train()

# 7. `prep_rollout(self)`:
#    - 这个函数用于准备策略网络进行推断（rollout）。
#    - 在函数内部，它将策略网络和值函数网络设置为评估模式（`eval()`）。
    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

# 8. `copy(self)`:
#    - 这个函数用于创建当前PPO策略的副本。
#    - 在函数内部，它创建一个新的`PPOPolicy`对象，并将当前对象的属性复制到新对象中。
#    - 它返回创建的副本对象。
    def copy(self):
        return PPOPolicy(self.args, self.obs_space, self.act_space, self.device)
