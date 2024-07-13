import os
import logging
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from typing import Literal, Union, List

import jsbsim
from .catalog import Property, Catalog
from ..utils.utils import get_root_dir, LLA2NEU, NEU2LLA

TeamColors = Literal["Red", "Blue", "Green", "Violet", "Orange"]


class BaseSimulator(ABC):

    def __init__(self, uid: str, color: TeamColors, dt: float):
        """Constructor. Creates an instance of simulator, initialize all the available properties.

        Args:
            uid (str): 5-digits hexadecimal numbers for unique identification.
            color (TeamColors): use different color strings to represent diferent teams
            dt (float): simulation timestep. Default = `1 / 60`.
        """
        self.__uid = uid
        self.__color = color
        self.__dt = dt
        self.model = ""
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)
        logging.debug(f"{self.__class__.__name__}:{self.__uid} is created!")

    @property
    def uid(self) -> str:
        return self.__uid

    @property
    def color(self) -> str:
        return self.__color

    @property
    def dt(self) -> float:
        return self.__dt

    def get_geodetic(self):#返回仿真器的地理位置（经度、纬度、高度）
        """(lontitude, latitude, altitude), unit: °, m"""
        return self._geodetic

    def get_position(self):#返回仿真器的位置（北、东、上，单位：米）。
        """(north, east, up), unit: m"""
        return self._position

    def get_rpy(self):#返回仿真器的姿态（滚转、俯仰、偏航，单位：弧度）。
        """(roll, pitch, yaw), unit: rad"""
        return self._posture

    def get_velocity(self):#返回仿真器的速度
        """(v_north, v_east, v_up), unit: m/s"""
        return self._velocity

    def reload(self):#重置仿真器的地理位置、位置、姿态和速度。
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)

    @abstractmethod
    def run(self, **kwargs):
        pass

    def log(self):
        lon, lat, alt = self.get_geodetic()
        roll, pitch, yaw = self.get_rpy() * 180 / np.pi
        log_msg = f"{self.uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
        log_msg += f"Name={self.model.upper()},"
        log_msg += f"Color={self.color}"
        return log_msg

    @abstractmethod
    def close(self):
        pass

    def __del__(self):
        logging.debug(f"{self.__class__.__name__}:{self.uid} is deleted!")


class AircraftSimulator(BaseSimulator):
    """A class which wraps an instance of JSBSim and manages communication with it.
    """

    ALIVE = 0
    CRASH = 1       # low altitude / extreme state / overload
    SHOTDOWN = 2    # missile attack

    def __init__(self,
                 uid: str = "A0100",
                 color: TeamColors = "Red",
                 model: str = 'f16',
                 init_state: dict = {},
                 origin: tuple = (120.0, 60.0, 0.0),
                 sim_freq: int = 60, **kwargs):
        """Constructor. Creates an instance of JSBSim, loads an aircraft and sets initial conditions.

        Args:
            uid (str): 5-digits hexadecimal numbers for unique identification. Default = `"A0100"`.
            color (TeamColors): use different color strings to represent diferent teams
            model (str): name of aircraft to be loaded. Default = `"f16"`.
                model path: './data/aircraft_name/aircraft_name.xml'
            init_state (dict): dict mapping properties to their initial values. Input empty dict to use a default set of initial props.
            origin (tuple): origin point (longitude, latitude, altitude) of the Global Combat Field. Default = `(120.0, 60.0, 0.0)`
            sim_freq (int): JSBSim integration frequency. Default = `60`.
        """
        super().__init__(uid, color, 1 / sim_freq)
        self.model = model
        self.init_state = init_state
        self.lon0, self.lat0, self.alt0 = origin
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        for key, value in kwargs.items():
            if key == 'num_missiles':
                self.num_missiles = value  # type: int
                self.num_left_missiles = self.num_missiles  # type: int
        # fixed simulator links
        self.partners = []  # type: List[AircraftSimulator]
        self.enemies = []   # type: List[AircraftSimulator]
        # temp simulator links
        self.launch_missiles = []   # type: List[MissileSimulator]
        self.under_missiles = []    # type: List[MissileSimulator]
        # initialize simulator
        self.reload()

    @property
    def is_alive(self):
        return self.__status == AircraftSimulator.ALIVE

    @property
    def is_crash(self):
        return self.__status == AircraftSimulator.CRASH

    @property
    def is_shotdown(self):
        return self.__status == AircraftSimulator.SHOTDOWN

    def crash(self):
        self.__status = AircraftSimulator.CRASH

    def shotdown(self):
        self.__status = AircraftSimulator.SHOTDOWN

    def reload(self, new_state: Union[dict, None] = None, new_origin: Union[tuple, None] = None):
        """Reload aircraft simulator
        """
        super().reload()

        # reset temp simulator links
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        self.launch_missiles.clear()
        self.under_missiles.clear()
        self.num_left_missiles = self.num_missiles

        # load JSBSim FDM
        self.jsbsim_exec = jsbsim.FGFDMExec(os.path.join(get_root_dir(), 'data'))
        self.jsbsim_exec.set_debug_level(0)
        self.jsbsim_exec.load_model(self.model)
        Catalog.add_jsbsim_props(self.jsbsim_exec.query_property_catalog(""))
        self.jsbsim_exec.set_dt(self.dt)
        self.clear_defalut_condition()

        # assign new properties
        if new_state is not None:
            self.init_state = new_state
        if new_origin is not None:
            self.lon0, self.lat0, self.alt0 = new_origin
        for key, value in self.init_state.items():
            self.set_property_value(Catalog[key], value)
        success = self.jsbsim_exec.run_ic()
        if not success:
            raise RuntimeError("JSBSim failed to init simulation conditions.")

        # propulsion init running
        propulsion = self.jsbsim_exec.get_propulsion()
        n = propulsion.get_num_engines()
        for j in range(n):
            propulsion.get_engine(j).init_running()
        propulsion.get_steady_state()
        # update inner property
        self._update_properties()

    def clear_defalut_condition(self):
        default_condition = {
            Catalog.ic_long_gc_deg: 120.0,  # geodesic longitude [deg]
            Catalog.ic_lat_geod_deg: 60.0,  # geodesic latitude  [deg]
            Catalog.ic_h_sl_ft: 20000,      # altitude above mean sea level [ft]
            Catalog.ic_psi_true_deg: 0.0,   # initial (true) heading [deg] (0, 360)
            Catalog.ic_u_fps: 800.0,        # body frame x-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_v_fps: 0.0,          # body frame y-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_w_fps: 0.0,          # body frame z-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_p_rad_sec: 0.0,      # roll rate  [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_q_rad_sec: 0.0,      # pitch rate [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_r_rad_sec: 0.0,      # yaw rate   [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_roc_fpm: 0.0,        # initial rate of climb [ft/min]
            Catalog.ic_terrain_elevation_ft: 0,
        }
        for prop, value in default_condition.items():
            self.set_property_value(prop, value)

    def run(self):
        """Runs JSBSim simulation until the agent interacts and update custom properties.

        JSBSim monitors the simulation and detects whether it thinks it should
        end, e.g. because a simulation time was specified. False is returned
        if JSBSim termination criteria are met.

        Returns:
            (bool): False if sim has met JSBSim termination criteria else True.
        """
        if self.is_alive:
            if self.bloods <= 0:
                self.shotdown()
            result = self.jsbsim_exec.run()
            if not result:
                raise RuntimeError("JSBSim failed.")
            self._update_properties()
            return result
        else:
            return True

    def close(self):
        """ Closes the simulation and any plots. """
        if self.jsbsim_exec:
            self.jsbsim_exec = None
        self.partners = []
        self.enemies = []

    def _update_properties(self):
        # update position
        self._geodetic[:] = self.get_property_values([
            Catalog.position_long_gc_deg,
            Catalog.position_lat_geod_deg,
            Catalog.position_h_sl_m
        ])
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)
        # update posture
        self._posture[:] = self.get_property_values([
            Catalog.attitude_roll_rad,
            Catalog.attitude_pitch_rad,
            Catalog.attitude_heading_true_rad,
        ])
        # update velocity
        self._velocity[:] = self.get_property_values([
            Catalog.velocities_v_north_mps,
            Catalog.velocities_v_east_mps,
            Catalog.velocities_v_down_mps,
        ])

    def get_sim_time(self):
        """ Gets the simulation time from JSBSim, a float. """
        return self.jsbsim_exec.get_sim_time()

    def get_property_values(self, props):
        """Get the values of the specified properties

        :param props: list of Properties

        : return: NamedTupl e with properties name and their values
        """
        return [self.get_property_value(prop) for prop in props]

    def set_property_values(self, props, values):
        """Set the values of the specified properties

        :param props: list of Properties

        :param values: list of float
        """
        if not len(props) == len(values):
            raise ValueError("mismatch between properties and values size")
        for prop, value in zip(props, values):
            self.set_property_value(prop, value)

    def get_property_value(self, prop):
        """Get the value of the specified property from the JSBSim simulation

        :param prop: Property

        :return : float
        """
        if isinstance(prop, Property):
            if prop.access == "R":
                if prop.update:
                    prop.update(self)
            return self.jsbsim_exec.get_property_value(prop.name_jsbsim)
        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def set_property_value(self, prop, value):
        """Set the values of the specified property

        :param prop: Property

        :param value: float
        """
        # set value in property bounds
        if isinstance(prop, Property):
            if value < prop.min:
                value = prop.min
            elif value > prop.max:
                value = prop.max

            self.jsbsim_exec.set_property_value(prop.name_jsbsim, value)

            if "W" in prop.access:
                if prop.update:
                    prop.update(self)
        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def check_missile_warning(self):
        for missile in self.under_missiles:
            if missile.is_alive:
                return missile
        return None


class MissileSimulator(BaseSimulator):

    #四个不同的状态值，用于判断导弹状态，不可更改，类似于宏
    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator, uid: str, missile_model: str = "AIM-9L"):
        assert parent.dt == target.dt, "integration timestep must be same!"
        missile = MissileSimulator(uid, parent.color, missile_model, parent.dt)
        missile.launch(parent) #发射
        missile.target(target) #设置目标
        return missile

    def __init__(self,
                 uid="A0101",
                 color="Red",
                 model="AIM-9L",
                 dt=1 / 12):
        super().__init__(uid, color, dt)
        self.__status = MissileSimulator.INACTIVE
        self.model = model
        self.parent_aircraft = None  # type: AircraftSimulator
        self.target_aircraft = None  # type: AircraftSimulator
        self.render_explosion = False

        # missile parameters (for AIM-9L)
        self._g = 9.81      # gravitational acceleration 重力加速度，值为 9.81 m/s^2。
        self._t_max = 60    # time limitation of missile life 导弹的飞行时间限制，设为 60 秒
        self._t_thrust = 3  # time limitation of engine  导弹发动机的工作时间限制，设为 3 秒。
        self._Isp = 120     # average specific impulse 比冲，即火箭发动机有效利用燃料的能力，这里设为 120 秒。这是导弹发动机性能的关键参数。
        self._Length = 2.87 #导弹的长度和直径，分别为 2.87 米和 0.127 米。
        self._Diameter = 0.127
        self._cD = 0.4      # aerodynamic drag factor 导弹的气动阻力系数，设为 0.4。这个值影响导弹在空气中飞行时所受到的阻力。
        self._m0 = 84       # mass, unit: kg 导弹的初始质量，设为 84 千克
        self._dm = 6        # mass loss rate, unit: kg/s 导弹飞行过程中的质量损失率，设为每秒 6 千克。这通常是指导弹发动机燃烧燃料导致的质量减少。
        self._K = 3         # proportionality constant of proportional navigation 比例导航中的比例常数，设为 3。这个值决定了导弹制导系统的敏感度。
        self._nyz_max = 30  # max overload 导弹能够承受的最大过载，设为 30 g。
        self._Rc = 300      # radius of explosion, unit: m 导弹爆炸的有效半径，设为 300 米。
        self._v_min = 150   # minimun velocity, unit: m/s  导弹的最小飞行速度，设为 150 米/秒。如果导弹速度低于此值，可能无法正常工作。

    @property
    def is_alive(self):
        """Missile is still flying is_alive
        属性用于检查导弹是否仍在飞行中。如果导弹的状态 (self.__status) 等于 MissileSimulator.LAUNCHED
        （即导弹已发射但尚未击中或错过目标），则返回 True，表示导弹仍在飞行。
        这是一个布尔值属性，用于快速判断导弹是否还处于活跃状态。"""
        return self.__status == MissileSimulator.LAUNCHED

    @property
    def is_success(self):
        """Missile has hit the target 用于检查导弹是否成功击中目标。
        如果导弹的状态等于 MissileSimulator.HIT，即导弹已经击中目标，那么返回 True。
        这同样是一个布尔值属性，用于判断导弹的攻击是否成功。"""
        return self.__status == MissileSimulator.HIT

    @property
    def is_done(self):
        """Missile is already exploded
        is_done 属性用于检查导弹是否已经结束其任务，无论是成功击中目标还是错过目标。
        如果导弹的状态是 MissileSimulator.HIT（击中目标）或者 MissileSimulator.MISS（错过目标），
        则返回 True，表示导弹已经爆炸或失效。这个属性用于确定导弹是否已经完成了它的使命，无论是成功还是失败。
        """
        return self.__status == MissileSimulator.HIT \
            or self.__status == MissileSimulator.MISS

    @property
    def Isp(self):
        return self._Isp if self._t < self._t_thrust else 0

    @property
    def K(self):
        """Proportional Guidance Coefficient"""
        # return self._K
        return max(self._K * (self._t_max - self._t) / self._t_max, 0)

    @property
    def S(self):
        """Cross-Sectional area, unit m^2"""
        S0 = np.pi * (self._Diameter / 2)**2
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return S0

    @property
    def rho(self):
        """Air Density, unit: kg/m^3"""
        # approximate expression
        return 1.225 * np.exp(-self._geodetic[-1] / 9300)
        # exact expression (Reference: https://www.cnblogs.com/pathjh/p/9127352.html)
        rho0, T0, h = 1.225, 288.15, self._geodetic[-1]
        if h <= 11000:  # Troposphere
            T = T0 - 0.0065 * h
            return rho0 * (T / T0)**4.25588
        elif h <= 20000:  # Lower Stratosphere
            T = 216.65
            return 0.36392 * np.exp((11000 - h) / 6341.62)
        else:  # Upper Stratosphere
            T = 216.65 + 0.001 * (h - 20000)
            return 0.088035 * (T / 216.65)**(-35.1632)

    @property
    def target_distance(self) -> float:
        return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())

    def launch(self, parent: AircraftSimulator):#launch 方法用于初始化导弹发射时的状态，继承父飞机的动能参数。将导弹状态设置为已发射。
        # inherit kinetic parameters from parent aricraft
        self.parent_aircraft = parent
        self.parent_aircraft.launch_missiles.append(self)
        self._geodetic[:] = parent.get_geodetic()
        self._position[:] = parent.get_position()
        self._velocity[:] = parent.get_velocity()
        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0  # missile's roll remains zero
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0
        # init status
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self.__status = MissileSimulator.LAUNCHED
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(5 / self.dt))  # 5s of distance increment -- can't hit
        #deque具有一个最大长度maxlen，当新元素添加到队列中，如果队列已满，最早的元素会被自动移除。
        #最大长度是通过计算5秒内的时间步数来确定的，即5 / self.dt。
        #代码中的注释"5s of distance increment -- can't hit"意味着，如果在5秒内，导弹与目标的距离没有显著减少，
        # 即认为导弹无法击中目标。这是一种启发式规则，用于在一定时间后判断导弹是否可能击中目标。
        #self.dt：这是导弹仿真的时间步长

        self._left_t = int(1 / self.dt)  # remove missile 1s after its destroying

    def target(self, target: AircraftSimulator):#设置导弹的目标飞机。
        self.target_aircraft = target  # TODO: change target?
        self.target_aircraft.under_missiles.append(self)

    def run(self):#定义了导弹在仿真环境中每经过一个时间步长 dt 时的运行逻辑
        self._t += self.dt #导弹的飞行时间 _t 增加了当前的时间步长 dt，这反映了导弹从上一次更新以来所经过的时间。
        action, distance = self._guidance() #调用 _guidance 方法来计算导弹应该遵循的制导指令 action 和当前导弹与目标之间的距离 distance。
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance
        if distance < self._Rc and self.target_aircraft.is_alive:#如果导弹与目标的距离小于爆炸半径 _Rc 且目标飞机仍然存活，则认为导弹击中了目标。
            self.__status = MissileSimulator.HIT #设置导弹命中状态并击落目标：
            self.target_aircraft.shotdown()
        elif (self._t > self._t_max) or (np.linalg.norm(self.get_velocity()) < self._v_min) \
                or np.sum(self._distance_increment) >= self._distance_increment.maxlen or not self.target_aircraft.is_alive:
            #如果导弹的飞行时间超过最大飞行时间 _t_max，或者导弹的速度小于最小速度 _v_min，
            # 或者距离增加的次数达到 _distance_increment 队列的最大长度，
            # 或者目标飞机已经被击落（不存活），则认为导弹未命中。
            self.__status = MissileSimulator.MISS
        else:
            #如果导弹没有击中也没有未命中，调用 _state_trans 方法来进行状态转换，根据制导指令 action 更新导弹的位置、速度和姿态。
            self._state_trans(action)
    # run方法是导弹仿真的每次迭代都会执行的，它根据导弹的制导逻辑和当前状态来更新导弹的行为，
    # 直到导弹命中目标、未命中或飞行时间结束。通过这种方法，仿真可以模拟导弹的飞行路径和攻击结果。


    def log(self):
        if self.is_alive:
            log_msg = super().log()
        elif self.is_done and (not self.render_explosion):
            self.render_explosion = True
            # remove missile model
            log_msg = f"-{self.uid}\n"
            # add explosion
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc}"
        else:
            log_msg = None
        return log_msg

    def close(self):
        self.target_aircraft = None

    def _guidance(self):
        """
        实现了导弹的制导逻辑
        Guidance law, proportional navigation
        """
        x_m, y_m, z_m = self.get_position() #导弹的位置
        dx_m, dy_m, dz_m = self.get_velocity() #导弹的速度
        v_m = np.linalg.norm([dx_m, dy_m, dz_m]) #导弹速度大小
        theta_m = np.arcsin(dz_m / v_m) #导弹的飞行俯仰角，即导弹速度向量与水平面的夹角。
        x_t, y_t, z_t = self.target_aircraft.get_position()#目标位置
        dx_t, dy_t, dz_t = self.target_aircraft.get_velocity()#目标速度
        Rxy = np.linalg.norm([x_m - x_t, y_m - y_t])  # distance from missile to target project to X-Y plane 导弹到目标在水平面（X-Y平面）上的投影距离。
        Rxyz = np.linalg.norm([x_m - x_t, y_m - y_t, z_t - z_m])  # distance from missile to target 导弹到目标的直线距离。
        # calculate beta & eps, but no need actually...
        # beta = np.arctan2(y_m - y_t, x_m - x_t)  # relative yaw
        # eps = np.arctan2(z_m - z_t, np.linalg.norm([x_m - x_t, y_m - y_t]))  # relative pitch
        dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / Rxy**2
        deps = ((dz_t - dz_m) * Rxy**2 - (z_t - z_m) * (
            (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz**2 * Rxy)
        ny = self.K * v_m / self._g * np.cos(theta_m) * dbeta
        nz = self.K * v_m / self._g * deps + np.cos(theta_m)
        return np.clip([ny, nz], -self._nyz_max, self._nyz_max), Rxyz

    def _state_trans(self, action):
        """
        State transition function
        """
        # update position & geodetic
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = NEU2LLA(*self.get_position(), self.lon0, self.lat0, self.alt0)
        # update velocity & posture
        v = np.linalg.norm(self.get_velocity())
        theta, phi = self.get_rpy()[1:]
        T = self._g * self.Isp * self._dm
        D = 0.5 * self._cD * self.S * self.rho * v**2
        nx = (T - D) / (self._m * self._g)
        ny, nz = action

        dv = self._g * (nx - np.sin(theta))
        self._dphi = self._g / v * (ny / np.cos(theta))
        self._dtheta = self._g / v * (nz - np.cos(theta))

        v += self.dt * dv
        phi += self.dt * self._dphi
        theta += self.dt * self._dtheta
        self._velocity[:] = np.array([
            v * np.cos(theta) * np.cos(phi),
            v * np.cos(theta) * np.sin(phi),
            v * np.sin(theta)
        ])
        self._posture[:] = np.array([0, theta, phi])
        # update mass
        if self._t < self._t_thrust:
            self._m = self._m - self.dt * self._dm
