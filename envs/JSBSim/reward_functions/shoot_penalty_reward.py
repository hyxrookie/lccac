from .reward_function_base import BaseRewardFunction


class ShootPenaltyReward(BaseRewardFunction):
    """
    ShootPenaltyReward
    when launching a missile, give -10 reward for penalty, 
    to avoid launching all missiles at once 
    """
    def __init__(self, config):
        super().__init__(config)

    def reset(self, task, env):
        self.pre_remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        # self.pre_remaining_missiles，记录每个代理的剩余导弹数量。agent.num_missiles表示代理当前拥有的导弹数量。
        return super().reset(task, env)


    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        if task.remaining_missiles[agent_id] == self.pre_remaining_missiles[agent_id] - 1:
            reward -= 2
            #如果当前代理的剩余导弹数量比前一次记录的减少了 1，则表示该代理刚刚发射了一枚导弹。
            # 此时，给予一个 -10 的奖励（即惩罚），以避免代理一次性发射所有导弹。
        self.pre_remaining_missiles[agent_id] = task.remaining_missiles[agent_id]
        #如果当前代理的剩余导弹数量比前一次记录的减少了 1，则表示该代理刚刚发射了一枚导弹。此时，给予一个 -10 的奖励（即惩罚），以避免代理一次性发射所有导弹。
        return self._process(reward, agent_id)