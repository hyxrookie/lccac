from .reward_function_base import BaseRewardFunction


class QsEventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """
    def __init__(self, config):
        super().__init__(config)

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
        if env.agents[agent_id].is_shotdown:
            reward -= 50
        elif env.agents[agent_id].is_crash:
            reward -= 50
        for missile in env.agents[agent_id].launch_missiles:
            if missile.is_success:
                env.agents[agent_id].launch_missiles.remove(missile)
                reward += 50
        # for missile in env.agents[agent_id].launch_missiles:
        #     if missile.is_success:
        #         reward += 100
        return reward
