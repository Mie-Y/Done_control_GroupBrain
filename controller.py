import numpy as np
import math

class VectorFieldController:
    def __init__(self):
        # 基础增益（仍可调）
        self.k_att = 2.0
        self.k_rep = 10.0
        self.k_boundary = 2.0
        self.k_form = 0.4
        self.k_predict = 0.8

        # 任务/安全参数
        self.capture_radius = 800.0        # 任务要求：距离坦克 < 800
        self.target_capture_thresh = 800.0
        self.min_inter_uav = 300.0        # 无人机间最小允许距离
        self.boundary_margin = 400.0

        # 队形半径限制（确保无人机间距 >= min_inter_uav）
        # 等边三角形关系：edge = sqrt(3) * radius -> radius = edge / sqrt(3)
        self.min_formation_radius = self.min_inter_uav / math.sqrt(3) * 1.05
        self.max_formation_radius = 900.0

        # 预测相关
        self.tank_velocity_history = []
        self.prediction_horizon = 8

        # 紧急机制
        self.emergency = False
        self.repulsion_boost = 1.0
        self.min_safe_distance = 120.0  # 近距紧急阈值

        # 速度限制（会基于坦克速度动态放宽）
        self.base_max_speed = 200.0 #300.0
        self.max_speed_cap = 800.0

        # 阶段
        self.phase = "approach"

    def compute_control(self, uavs, tank, tank_velocity=None):
        """
        uavs: list of (x,y)
        tank: (x,y)
        tank_velocity: (vx,vy) or None
        returns list of velocity vectors (vx,vy) for each UAV
        """
        n = len(uavs)
        controls = [np.zeros(2) for _ in range(n)]

        # 更新历史，用于预测（外部也可以更新）
        if tank_velocity is not None:
            self.tank_velocity_history.append(np.array(tank_velocity))
            if len(self.tank_velocity_history) > 20:
                self.tank_velocity_history.pop(0)

        # 预测坦克位置（短期）
        predicted_tank = self._predict_tank(tank, tank_velocity)

        # 计算队形半径，确保满足至少 min_inter_uav
        formation_radius = self._formation_radius(uavs, tank)

        # 调整阶段与参数
        self._update_phase(uavs, tank)
        self._tune_gains_by_phase()

        # 对每架无人机计算潜力场力并得到速度向量
        for i, uav in enumerate(uavs):
            total_force = np.zeros(2)

            # 1. 吸引力（向预测位置/坦克靠近）
            total_force += self._attraction_force(uav, predicted_tank)

            # 2. 无人机间排斥（增强近距离排斥）
            for j, other in enumerate(uavs):
                if i == j:
                    continue
                total_force += self._repulsion_force(uav, other)

            # 3. 队形保持力（将无人机摆到队形位置）
            total_force += self._formation_force(i, uav, n, predicted_tank, formation_radius)

            # 4. 边界排斥
            total_force += self._boundary_force(uav)

            # 5. 若处于接近阶段，加入拦截/预测分量
            if self.phase == "approach" and tank_velocity is not None:
                total_force += self._intercept_force(uav, tank, tank_velocity, formation_radius)

            # 6. 稳定性（在围捕/维持阶段弱力吸引到目标圈中）
            if self.phase in ("surround", "maintain"):
                total_force += self._stability_force(uav, predicted_tank, formation_radius)

            # 放大排斥逻辑（若有连续违例）
            if self.repulsion_boost > 1.0:
                total_force *= self.repulsion_boost

            # 把力映射为速度（直接当做期望速度方向），并做最大速度限制
            desired_speed = np.linalg.norm(total_force)
            max_speed = self._get_max_speed(tank_velocity)
            if desired_speed < 1e-6:
                vel = np.zeros(2)
            else:
                scale = min(desired_speed, max_speed) / desired_speed
                vel = total_force * scale

            controls[i] = tuple(vel)

        return controls

    # ---------- 力的具体实现 ----------
    def _attraction_force(self, uav, target):
        v = np.array(target) - np.array(uav)
        d = np.linalg.norm(v)
        if d < 1e-6:
            return np.zeros(2)
        dir = v / d
        # 在 capture_radius 内弱化吸引，外面加强
        if d > self.capture_radius:
            mag = self.k_att * d * 1.1
        else:
            mag = self.k_att * d * 0.5
        return mag * dir

    def _repulsion_force(self, uav, other):
        vec = np.array(uav) - np.array(other)
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            # 随机方向推开
            dir = np.random.rand(2) - 0.5
            dir = dir / (np.linalg.norm(dir) + 1e-6)
            dist = 0.1
        else:
            dir = vec / dist

        # 作用范围：若在 3 * min_inter_uav 内开始感知
        sense_range = max(3.0 * self.min_inter_uav, 600.0)
        if dist < sense_range:
            # 非线性排斥：距离越近增益越大
            if dist < self.min_safe_distance:
                mag = self.k_rep * 30.0 * (1.0 / (dist + 1e-6) - 1.0 / (self.min_safe_distance + 1e-6))**2
            else:
                mag = self.k_rep * (1.0 / (dist + 1e-6) - 1.0 / sense_range)
            return mag * dir
        return np.zeros(2)

    def _formation_force(self, idx, uav, n, tank, radius):
        """把无人机摆成等间隔队形（按编号分配角度）"""
        angle = 2.0 * math.pi * idx / n
        target = np.array([tank[0] + radius * math.cos(angle),
                           tank[1] + radius * math.sin(angle)])
        v = target - np.array(uav)
        d = np.linalg.norm(v)
        if d < 1e-6:
            return np.zeros(2)
        dir = v / d
        mag = self.k_form * min(d, 200.0)
        return mag * dir

    def _boundary_force(self, uav):
        x, y = uav
        b = 2500.0
        f = np.zeros(2)
        margin = self.boundary_margin
        # x方向
        if x < -b + margin:
            dx = x + b
            f[0] += self.k_boundary * (1.0 / (abs(dx) + 1e-6) - 1.0 / margin)
        elif x > b - margin:
            dx = b - x
            f[0] -= self.k_boundary * (1.0 / (abs(dx) + 1e-6) - 1.0 / margin)
        # y方向
        if y < -b + margin:
            dy = y + b
            f[1] += self.k_boundary * (1.0 / (abs(dy) + 1e-6) - 1.0 / margin)
        elif y > b - margin:
            dy = b - y
            f[1] -= self.k_boundary * (1.0 / (abs(dy) + 1e-6) - 1.0 / margin)
        return f

    def _intercept_force(self, uav, tank, tank_velocity, formation_radius):
        """基于坦克速度预测拦截点并朝向该点"""
        if tank_velocity is None:
            return np.zeros(2)
        v_t = np.array(tank_velocity)
        speed_t = np.linalg.norm(v_t)
        if speed_t < 1e-6:
            return np.zeros(2)
        dir_to_tank = np.array(tank) - np.array(uav)
        dist = np.linalg.norm(dir_to_tank)
        # 估计时间窗（短）
        time_to_intercept = max(1.0, dist / (self.base_max_speed + 1e-6))
        pred_point = np.array(tank) + v_t * min(self.prediction_horizon, max(1, int(time_to_intercept)))
        # 对拦截点施加中等强度吸引
        vec = pred_point - np.array(uav)
        d = np.linalg.norm(vec)
        if d < 1e-6:
            return np.zeros(2)
        return (1.2 * self.k_predict * d) * (vec / d)

    def _stability_force(self, uav, tank, formation_radius):
        """在围捕/维持阶段弱力保持在目标环上"""
        dir_vec = np.array(tank) - np.array(uav)
        dist = np.linalg.norm(dir_vec)
        if dist < 1e-6:
            return np.zeros(2)
        desired = formation_radius * 0.95
        # 只有当差距不大时施加稳定力（避免与排斥冲突）
        if abs(dist - desired) < 250:
            return 0.4 * (dist - desired) * (dir_vec / dist)
        return np.zeros(2)

    # ---------- 其它工具方法 ----------
    def _predict_tank(self, tank, tank_velocity):
        """简单短期预测：有历史时用加权平均速度"""
        if tank_velocity is None:
            return tank
        if len(self.tank_velocity_history) >= 2:
            weights = np.linspace(0.2, 1.0, min(len(self.tank_velocity_history), 10))
            recent = np.stack(self.tank_velocity_history[-len(weights):])
            w = weights / np.sum(weights)
            avg_v = np.sum(recent * w[:, None], axis=0)
            horizon = self.prediction_horizon
            return (tank[0] + avg_v[0] * horizon, tank[1] + avg_v[1] * horizon)
        else:
            # 简单线性预测
            return (tank[0] + tank_velocity[0] * self.prediction_horizon,
                    tank[1] + tank_velocity[1] * self.prediction_horizon)

    def _formation_radius(self, uavs, tank):
        # avg distance to tank
        dists = [np.linalg.norm(np.array(u) - np.array(tank)) for u in uavs]
        avg = np.mean(dists) if len(dists) > 0 else (self.capture_radius + 200)
        # radius 应保证等边三角形边长 >= min_inter_uav
        min_rad = self.min_inter_uav / math.sqrt(3) * 1.05
        # scale radius with avg distance,限幅
        if avg < self.capture_radius:
            return max(min_rad, avg * 0.6)
        else:
            # 在capture_radius..2*capture_radius之间线性过渡
            r = min(self.max_formation_radius, max(min_rad, avg * 0.8))
            return r

    def _get_max_speed(self, tank_velocity):
        t_speed = np.linalg.norm(tank_velocity) if tank_velocity is not None else 0.0
        max_speed = self.base_max_speed
        # 若坦克速度大，允许无人机跟速提升（但受 cap 限制）
        if t_speed > 200:
            max_speed = min(self.max_speed_cap, self.base_max_speed * (1.0 + t_speed / 300.0))
        return max_speed

    def _update_phase(self, uavs, tank):
        dists = [np.linalg.norm(np.array(u) - np.array(tank)) for u in uavs]
        if all(d < self.capture_radius for d in dists) and self._check_uav_spacing(uavs):
            self.phase = "maintain"
        elif np.mean(dists) < self.capture_radius * 1.5:
            self.phase = "surround"
        else:
            self.phase = "approach"

    def _tune_gains_by_phase(self):
        if self.phase == "approach":
            self.k_att = 2.2
            self.k_rep = 12.0
            self.k_form = 0.25
        elif self.phase == "surround":
            self.k_att = 1.2
            self.k_rep = 14.0
            self.k_form = 0.5
        else:  # maintain
            self.k_att = 0.8
            self.k_rep = 18.0
            self.k_form = 0.6

    def _check_uav_spacing(self, uavs):
        for i in range(len(uavs)):
            for j in range(i+1, len(uavs)):
                if np.linalg.norm(np.array(uavs[i]) - np.array(uavs[j])) < self.min_inter_uav:
                    return False
        return True
