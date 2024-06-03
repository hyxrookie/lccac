import numpy as np
from .env_base import BaseEnv
from ..tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'hierarchical_singlecombat':
            self.task = HierarchicalSingleCombatTask(self.config)
        elif taskname == 'singlecombat_dodge_missile':
            self.task = SingleCombatDodgeMissileTask(self.config)
        elif taskname == 'singlecombat_shoot':
            self.task = SingleCombatShootMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_dodge_missile':
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_shoot':#1v1/ShootMissile/HierarchySelfplay
            self.task = HierarchicalSingleCombatShootTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> np.ndarray:#覆盖env_base中的reset方法
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        # switch side
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        # self.init_states[0].update({
        #     'ic_psi_true_deg': (self.np_random.uniform(270, 540))%360,
        #     'ic_h_sl_ft': self.np_random.uniform(17000, 23000),
        # })
        init_states = self.init_states.copy()
        self.np_random.shuffle(init_states)
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])
        self._tempsims.clear()

# 这段代码定义了一个名为 `SingleCombatEnv` 的类，作为一个一对一竞技场景的环境。下面是对该代码的作用的解释：
# 1. 继承和初始化：
#    - `SingleCombatEnv` 类继承自 `BaseEnv` 类，这是一个基础环境类。
#    - 在 `__init__` 方法中，通过调用 `super().__init__(config_name)` 来初始化基础环境。
# 2. 加载任务：
#    - `load_task` 方法根据配置文件中指定的任务名称，加载相应的任务对象。
#    - 根据任务名称，选择合适的任务类进行初始化，并将其赋值给 `self.task`。
# 3. 环境重置：
#    - `reset` 方法用于重置环境到初始状态，并返回初始观测。
#    - 在重置过程中，将当前步数重置为0，重置模拟器状态并调用任务的重置方法。
#    - 最后，通过调用 `get_obs` 方法获取观测，并将其打包后返回。
# 4. 模拟器重置：
#    - `reset_simulators` 方法用于重置模拟器的状态。
#    - 如果 `init_states` 是空的，则将模拟器的初始状态保存到 `init_states` 列表中。
#    - 然后，通过打乱 `init_states` 列表中的状态顺序，为每个模拟器重新加载状态。
#    - 这样可以在每次重置环境时，为不同的模拟器加载不同的初始状态。
# 总的来说，这段代码定义了一个针对一对一竞技场景的环境类 `SingleCombatEnv`，
# 其中包括加载任务、环境重置和模拟器重置等功能。这些功能用于初始化环境、控制环境的重置，并与具体的任务对象进行交互和管理。