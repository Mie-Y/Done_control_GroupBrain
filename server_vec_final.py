import asyncio
import websockets
import json
import base64
import logging
from typing import List, Tuple, Dict, Any
from PIL import Image
import io
import time
import numpy as np
import math
from controller import VectorFieldController
from ultralytics import YOLO
import random

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 固定参数 / 相机参数（按照你给定的分辨率和 FOV）
IMG_W = 940
IMG_H = 540
FOV_H_DEG = 90.0  # 水平视场角（度）
FOV_H_RAD = math.radians(FOV_H_DEG)
# 焦距（像素）: f = (width/2) / tan(FOV/2)
f_px = (IMG_W / 2.0) / math.tan(FOV_H_RAD / 2.0)
c_x = IMG_W / 2.0
c_y = IMG_H / 2.0

# 高度
z_fixed = -2797.85  # 无人机高度固定为200
tank_height = -2910.1  # 坦克高度
# 最大允许的水平距离（防止角度逼近0导致的爆炸式距离）
MAX_HORIZ_DIST = 10000.0


def estimate_tank_from_detections(drones: List[Tuple[float, float, float, float, float]]):
    """
    基于每架无人机检测到的图像坐标 (u,v) 以及无人机位置与朝向估计坦克的地面坐标 (x,y)。
    drones 列表元素格式： (x_drone, y_drone, z_drone, yaw_rad, detection_object)
    其中 detection_object: (u_center, v_center, score)
    返回 (x_tank, y_tank) 或 None
    """

    # 准备每架检测到的单点估计
    estimates = []
    weights = []

    dz = z_fixed - tank_height
    if dz <= 0:
        logger.warning("Drone altitude <= tank altitude, geometry invalid.")
        return None

    for entry in drones:
        x_i, y_i, z_i, yaw_rad, detection = entry
        u_center, v_center, score = detection

        # 计算归一化的水平、垂直角度（以相机前轴为基准）
        a = math.atan2((u_center - c_x), f_px)    # 水平角（相机坐标系）
        b = math.atan2((v_center - c_y), f_px)    # 垂直角（下为正，如果摄像机坐标系如此约定）

        # 如果垂直角接近0，则水平距离很大 -> 跳过或限制
        tan_b = math.tan(abs(b))
        if tan_b < 1e-6:
            horiz_dist = MAX_HORIZ_DIST
        else:
            horiz_dist = dz / tan_b
            # clamp
            horiz_dist = min(horiz_dist, MAX_HORIZ_DIST)

        # 全局方位角 = 无人机朝向 + 相机观测到的水平偏移角
        global_bearing = yaw_rad/180*math.pi + a

        # 计算估计点
        x_est = x_i + horiz_dist * math.cos(global_bearing)
        y_est = y_i + horiz_dist * math.sin(global_bearing)


        estimates.append((x_est, y_est))
        # 使用置信度作为权重（置信度为 0-1）
        w = float(score) if score is not None else 1.0
        weights.append(max(w, 1e-6))

    if len(estimates) == 0:
        return None
    elif len(estimates) == 1:
        # 只有一架无人机有检测，直接返回单点估计（并打印警告）
        logger.info("Only one drone detected the tank. Using single-shot estimate.")
        return estimates[0]
    else:
        # 多架无人机：加权平均
        w_sum = sum(weights)
        x_avg = sum(e[0] * w for e, w in zip(estimates, weights)) / w_sum
        y_avg = sum(e[1] * w for e, w in zip(estimates, weights)) / w_sum
        return x_avg, y_avg


def read_detector(results, drone_locations, drone_angles):
    """读取检测结果并估计坦克位置"""
    drones_data = []

    # 检查每个无人机是否检测到坦克
    for i in range(3):
        if 0 in results[i][0].boxes.cls.tolist():
            # 获取检测框坐标
            box_idx = results[i][0].boxes.cls.tolist().index(0)
            x_l, y_l, x_r, y_r = results[i][0].boxes.xyxy[box_idx].tolist()

            # 计算检测框中心
            u_center = (x_l + x_r) / 2
            v_center = (y_l + y_r) / 2

            # 获取置信度
            score = results[i][0].boxes.conf[box_idx].tolist()

            # 添加到无人机数据列表（包含位置、角度和检测信息）
            drones_data.append([
                drone_locations[i][0],  # 无人机x坐标
                drone_locations[i][1],  # 无人机y坐标
                drone_angles[i],  # 无人机方向角度
                u_center,  # 检测框中心u坐标
                v_center,  # 检测框中心v坐标
                score  # 置信度
            ])

    # 如果至少有两个无人机检测到坦克，则估计坦克位置
    if len(drones_data) >= 2:
        tank_position = estimate_tank_position(drones_data)
        if tank_position is not None:
            x_tank, y_tank = tank_position
            print(f"坦克估计坐标: ({x_tank}, {y_tank})")
            return x_tank, y_tank
    return None


def read_detector1(results, drone_infos):
    """
    读取检测结果并估计坦克位置。
    results: [result1, result2, result3] 由 YOLO.predict 返回
    drone_infos: 每个元素为 (x, y, z, yaw_rad)
    """
    drones_with_detection = []

    for i in range(len(results)):
        res = results[i]
        info = drone_infos[i]
        [x_drone, y_drone, z_drone], yaw_rad = info

        try:
            # 判断类别 0 是否存在（假设坦克类别为 0）
            classes = res[0].boxes.cls.tolist()
            if 0 in classes:
                box_idx = classes.index(0)
                x_l, y_l, x_r, y_r = res[0].boxes.xyxy[box_idx].tolist()
                u_center = (x_l + x_r) / 2.0
                v_center = (y_l + y_r) / 2.0
                score = float(res[0].boxes.conf[box_idx].tolist())
                drones_with_detection.append((x_drone, y_drone, z_drone, yaw_rad, (u_center, v_center, score)))
        except Exception as e:
            logger.debug(f"No detection parsing for drone {i}: {e}")
            continue

    if len(drones_with_detection) >= 1:
        est = estimate_tank_from_detections(drones_with_detection)
        if est is not None:
            logger.info(f"Estimated tank XY from detectors: {est}")
            return est

    return None

class DroneWebSocketServer:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.clients = set()
        self.controller = VectorFieldController()
        self.tank_velocity_history = []
        self.detector = YOLO("F:/Platform/Server/detec/best.pt")

    def HandleRequestFunc(self, image_1, image_2, image_3, drone_1_location, drone_2_location, drone_3_location,
                          drone_1_angle, drone_2_angle, drone_3_angle, tank_location):
        """
        处理请求的核心函数

        参数:
        - image_1, image_2, image_3: PIL Image对象
        - drone_1_location, drone_2_location, drone_3_location: [x, y, z] 坐标列表
        - drone_1_angle, drone_2_angle, drone_3_angle: 无人机方向角度（度）
        - tank_location: [x, y, z] 坐标列表

        返回:
        - [[drone_1_newlocation, angle_1], [drone_2_newlocation, angle_2], [drone_3_newlocation, angle_3]]
        """
        # logger.info("Processing request with HandleRequestFunc")
        # logger.info(f"Drone locations: {drone_1_location}, {drone_2_location}, {drone_3_location}")
        # logger.info(f"Drone angles: {drone_1_angle}, {drone_2_angle}, {drone_3_angle}")
        # logger.info(f"Tank location: {tank_location}")
        # logger.info(f"Image sizes: {image_1.size if image_1 else None}, {image_2.size if image_2 else None}, {image_3.size if image_3 else None}")

        pathresult = []
        ###################################sunhongze###################################
        result1 = self.detector.predict(
            image_1,
            save=False)
        result2 = self.detector.predict(
            image_2,
            save=False)
        result3 = self.detector.predict(
            image_3,
            save=False)

        # drone_locations = [drone_1_location, drone_2_location, drone_3_location]
        # drone_angles = [math.atan((tank_location[1] - drone_1_location[1]) / (tank_location[0] - drone_1_location[0])) / (math.pi) * 180,
        #                 math.atan((tank_location[1] - drone_2_location[1]) / (tank_location[0] - drone_2_location[0])) / (math.pi) * 180,
        #                 math.atan((tank_location[1] - drone_3_location[1]) / (tank_location[0] - drone_3_location[0])) / (math.pi) * 180]
        # drone_angles = [drone_1_angle, drone_2_angle, drone_3_angle]
        # predict_tank_location = read_detector([result1, result2, result3], drone_locations, drone_angles)
        drone_infos = [[drone_1_location,drone_1_angle],[drone_2_location,drone_2_angle],[drone_3_location,drone_3_angle]]
        predict_tank_location = read_detector1([result1, result2, result3], drone_infos)

        # if predict_tank_location is None or ((predict_tank_location[0]-tank_location[0])**2+(predict_tank_location[1]-tank_location[1])**2)>1000:
        # predict_tank_location = None
        print('Real tank XY from platform is ', tank_location[0], tank_location[1])
        if predict_tank_location is None or math.sqrt(((predict_tank_location[0] - tank_location[0]) ** 2 + (predict_tank_location[1] - tank_location[1]) ** 2))>300:
            # 如果没有检测到坦克，使用原来的坦克位置
            result = self.path(drone_1_location, drone_2_location, drone_3_location, tank_location)
            # print('sunhongze!sunhongze!sunhongze!sunhongze!sunhongze!sunhongze')
        else:
            # 使用检测到的坦克位置
            x_tank, y_tank = predict_tank_location
            # 保持原来的z坐标
            new_tank_location = [x_tank, y_tank, tank_location[2]]
            result = self.path(drone_1_location, drone_2_location, drone_3_location, new_tank_location)
            print('sunhongze=sunhongze=sunhongze=sunhongze=sunhongze=sunhongze')
        ###################################sunhongze###################################
        # result = self.path(drone_1_location, drone_2_location, drone_3_location, tank_location)
        # boundingresult = self.bounding(image_1, image_2, image_3)
        logger.info(f"New drone positions and angles: {result}")

        return result

    def base64_to_image(self, base64_string: str) -> Image.Image:
        """将base64字符串转换为PIL Image对象"""
        try:
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return None

    def parse_drone_group_data(self, data: Dict[str, Any]) -> Tuple[List[Image.Image], List[List[float]], List[float]]:
        """解析无人机组数据"""
        images = [None, None, None]
        drone_locations = [None, None, None]
        drone_rotations = [None, None, None]
        tank_location = None

        try:
            # 解析坦克位置
            if 'tank' in data and 'position' in data['tank']:
                tank_pos = data['tank']['position']
                tank_location = [tank_pos['x'], tank_pos['y'], tank_pos['z']]

            # 解析无人机数据
            if 'drones' in data:
                for drone_data in data['drones']:
                    drone_id = drone_data.get('drone_id', 0)

                    if 0 <= drone_id < 3:  # 确保drone_id在有效范围内
                        # 解析位置
                        if 'position' in drone_data:
                            pos = drone_data['position']
                            drone_locations[drone_id] = [pos['x'], pos['y'], pos['z']]

                        # 解析角度
                        if 'angle' in drone_data:
                            drone_rotations[drone_id] = drone_data['angle']

                        # 解析图像
                        if 'image' in drone_data and 'data' in drone_data['image']:
                            base64_data = drone_data['image']['data']
                            images[drone_id] = self.base64_to_image(base64_data)

            # 填充缺失的数据为默认值
            for i in range(3):
                if drone_locations[i] is None:
                    drone_locations[i] = [0.0, 0.0, 0.0]
                if images[i] is None:
                    # 创建一个空的1x1像素图像作为占位符
                    images[i] = Image.new('RGB', (1, 1), color='black')
                if drone_rotations[i] is None:
                    drone_rotations[i] = 0.0

            if tank_location is None:
                tank_location = [0.0, 0.0, 0.0]

        except Exception as e:
            logger.error(f"Error parsing drone group data: {e}")
            # 返回默认值
            images = [Image.new('RGB', (1, 1), color='black') for _ in range(3)]
            drone_locations = [[0.0, 0.0, 0.0] for _ in range(3)]
            drone_rotations = [0.0 for _ in range(3)]
            tank_location = [0.0, 0.0, 0.0]

        return images, drone_locations, drone_rotations, tank_location

    async def handle_client(self, websocket):
        """处理客户端连接"""
        logger.info(f"New client connected from {websocket.remote_address}")
        self.clients.add(websocket)

        try:
            async for message in websocket:
                try:
                    # 解析JSON消息
                    data = json.loads(message)
                    message_type = data.get('type', '')

                    # logger.info(f"Received message type: {message_type}")

                    if message_type == 'drone_group_data':
                        # 解析无人机组数据
                        images, drone_locations, drone_angles, tank_location = self.parse_drone_group_data(data)
                        # print("Drone_1 location:",drone_locations[0], ", Drone_2 location:",drone_locations[1],", Drone_3 location:",  drone_locations[2],", Tank location:", tank_location)
                        # print("Drone_1 angle:",drone_angles[0], ", Drone_2 angle:",drone_angles[1],", Drone_3 angle:",  drone_angles[2])

                        # 调用处理函数
                        pathresult = self.HandleRequestFunc(
                            images[0], images[1], images[2],
                            drone_locations[0], drone_locations[1], drone_locations[2],
                            drone_angles[0], drone_angles[1], drone_angles[2],
                            tank_location
                        )
                        # for i in range(3):
                        #     print("Drone ", i,
                        #           f"location:[{pathresult[i][0][0]},{pathresult[i][0][1]},{pathresult[i][0][2]}] ,")
                        # 构建响应消息
                        # print("Drone_1 length : ",round(math.sqrt((pathresult[1][0][0] - tank_location[0])**2 + (pathresult[1][0][1] - tank_location[1])**2)))

                        response = {
                            "type": "drone_control_response",
                            "group_id": data.get('group_id', 0),
                            "timestamp": data.get('timestamp', 0),
                            "tank_location": {
                                "x": tank_location[0],
                                "y": tank_location[1],
                                "z": tank_location[2]
                            },
                            "drone_commands": [
                                {
                                    "drone_id": i,
                                    "new_position": {
                                        "x": pathresult[i][0][0],
                                        "y": pathresult[i][0][1],
                                        "z": pathresult[i][0][2]
                                    },
                                    "rotation_angle": pathresult[i][1]  # 绕z轴旋转角度（度）
                                }
                                for i in range(3)
                            ],
                            "drone_boundingbox": [
                                {
                                    "drone_id": j,
                                    "boundingbox": {
                                        # "x": boundingresult[j][0],
                                        # "y": boundingresult[j][1],
                                        # "w": boundingresult[j][2],
                                        # "h": boundingresult[j][3]
                                        "x": 0.5,
                                        "y": 0.5,
                                        "w": 0.5,
                                        "h": 0.5
                                    }
                                }
                                for j in range(3)
                            ]
                        }

                        # 发送响应
                        await websocket.send(json.dumps(response))
                        # logger.info(f"Sent response for group {data.get('group_id', 0)}")

                    else:
                        logger.warning(f"Unknown message type: {message_type}")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error in client handler: {e}")
        finally:
            self.clients.discard(websocket)

    async def start_server(self):
        """启动WebSocket服务器"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        start_server = websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )

        await start_server
        logger.info("WebSocket server started successfully")

    def run(self):
        """运行服务器"""
        asyncio.get_event_loop().run_until_complete(self.start_server())
        asyncio.get_event_loop().run_forever()

    ###################################sunhongze###################################
    def path(self, drone_1_location, drone_2_location, drone_3_location, tank_location):
        """
        输入与输出格式保持原样。
        主要流程：
        1) 估计坦克平面速度
        2) 调用 controller.compute_control 得到期望速度向量
        3) 通过 dt 计算新位置
        4) 后处理：保证 inter-UAV >=200、UAV->tank <800（尽量）、并验证可视性（若不可见尝试小幅移动以进入视野）
        """
        # 1. 组装数据
        uavs_2d = [
            (drone_1_location[0], drone_1_location[1]),
            (drone_2_location[0], drone_2_location[1]),
            (drone_3_location[0], drone_3_location[1])
        ]
        tank_2d = (tank_location[0], tank_location[1])

        # 2. 坦克速度估计（使用历史）
        tank_velocity = None
        current_time = time.time()
        if len(self.tank_velocity_history) > 0:
            last_pos, last_vel, last_t = self.tank_velocity_history[-1]
            dt = current_time - last_t if current_time - last_t > 1e-6 else 1e-6
            vx = (tank_2d[0] - last_pos[0]) / dt
            vy = (tank_2d[1] - last_pos[1]) / dt
            # 处理异常速度：如果瞬间估计 > 1500 则用历史平均
            speed = math.hypot(vx, vy)
            if speed > 1500 and len(self.tank_velocity_history) > 1:
                historical = [v for _, v, _ in self.tank_velocity_history if v is not None]
                if historical:
                    vx = float(np.mean([v[0] for v in historical]))
                    vy = float(np.mean([v[1] for v in historical]))
            tank_velocity = (vx, vy)
        # 更新历史
        self.tank_velocity_history.append((tank_2d, tank_velocity, current_time))
        if len(self.tank_velocity_history) > 40:
            self.tank_velocity_history.pop(0)

        # 3. 调用控制器得到速度向量
        controls = self.controller.compute_control(uavs_2d, tank_2d, tank_velocity)

        # 4. 基于 dt 计算新位置（并进行后处理）
        dt = 0.25  # 时间步长，可调
        new_results = []
        proposed_positions = []
        for i, (vx, vy) in enumerate(controls):
            current_location = [drone_1_location, drone_2_location, drone_3_location][i]
            new_x = current_location[0] + vx * dt
            new_y = current_location[1] + vy * dt
            new_z = current_location[2]

            # 边界限制
            boundary = 2500
            new_x = max(min(new_x, boundary - 100), -boundary + 100)
            new_y = max(min(new_y, boundary - 100), -boundary + 100)

            proposed_positions.append([new_x, new_y, new_z])

        # 5. 后处理：保证无人机间距 >=200（若违反，进行最小修正推开）
        #    同时，尝试让每架无人机能“看到”坦克（若无法，则以最小移动朝向使其进入横向 FOV）
        #    并尽量保证 UAV->tank 距离 < 800（如果当前距>800则优先靠近）
        final_positions = proposed_positions.copy()
        # 多轮修正（最多三轮）
        for _iter in range(3):
            changed = False
            # 检查间距
            for i in range(3):
                for j in range(i+1, 3):
                    pi = np.array(final_positions[i][:2])
                    pj = np.array(final_positions[j][:2])
                    dist = np.linalg.norm(pi - pj)
                    if dist < self.controller.min_inter_uav:
                        # 推开两架机（沿连线方向）
                        overlap = self.controller.min_inter_uav - dist + 1e-3
                        dir = (pi - pj)
                        if np.linalg.norm(dir) < 1e-6:
                            dir = np.random.rand(2) - 0.5
                        dir = dir / (np.linalg.norm(dir) + 1e-6)
                        # 将两机分别沿正反方向移动一半 overlap（并限幅）
                        move = dir * (overlap / 2.0)
                        final_positions[i][0] += move[0]
                        final_positions[i][1] += move[1]
                        final_positions[j][0] -= move[0]
                        final_positions[j][1] -= move[1]
                        changed = True

            # 检查到坦克距离与可视性（相机参数来自文件头）
            for i in range(3):
                px, py, pz = final_positions[i]
                # 计算水平角度差（无人机朝向朝向坦克）
                dx = tank_location[0] - px
                dy = tank_location[1] - py
                horiz_angle = math.atan2(dy, dx)  # rad

                # 计算相机竖直角（基于高度差）
                dz = z_fixed - tank_height
                horiz_dist = math.hypot(dx, dy)
                if horiz_dist < 1e-6:
                    vert_angle = 0.0
                else:
                    vert_angle = math.atan2(dz, horiz_dist)

                # 判断是否在水平FOV内（使用 FOV_H_RAD）
                # 先假设无人机朝向朝向坦克（server中会把角度设朝向坦克）
                # 因为我们没有 UAV yaw 这里只判断绝对角度是否能被相机看到（通过在平台上会有 yaw）
                # 简化：只判断水平角偏差在 +/- FOV_H_RAD/2 内
                # 若不在视野里，尝试小移使之进入视野（向坦克侧向靠拢）
                # 这里的 yaw 不确定，采用保守策略：如果相对角度太大则横向靠近坦克
                # 使用一个简单判定：若 horiz_dist > 0 并且 horiz_dist < 10000
                # 如果 UAV->tank 距离过大 (> capture_radius)，优先靠近
                to_tank_dist = math.hypot(dx, dy)
                if to_tank_dist > self.controller.target_capture_thresh + 100:
                    # 快速靠近：往坦克方向移动一个步长（但不超过max speed * dt）
                    max_step = self.controller._get_max_speed(tank_velocity) * dt
                    move_vec = np.array([dx, dy]) / (to_tank_dist + 1e-6) * min(max_step, to_tank_dist)
                    final_positions[i][0] += move_vec[0]
                    final_positions[i][1] += move_vec[1]
                    changed = True

                # 如果水平投影角度过大（即需要在视野边缘以外），尝试微改横向位置：
                # 视野约束：用简单扇形判断（若超出，就向坦克方向横向靠拢一点）
                # 这里做保守移动，避免大幅改变队形
                # 计算当前相对角度与无人机朝向（server 会把朝向对齐坦克，这里只做基于几何的微调整）
                # 不精确 yaw 时，只要横向偏移能缩小角度就行
                # 计算角偏（以坦克为原点）
                # 若需要更严谨的 yaw 校验，应把 drone_angles 作为输入
                # 直接保证：若 horiz_dist > 0 且 to_tank_dist * sin(delta_angle) > something -> 横向移动
                # 这里把阈值设为 capture_radius * tan(FOV/2)
                fov_half = FOV_H_RAD / 2.0
                # 允许的横向偏移 at distance = horiz_dist
                allowed_side = math.tan(fov_half) * max(1.0, 1.0) * (horiz_dist if horiz_dist>1 else 1)
                # 横向偏移估算：distance from line joining UAV->tank perpendicular component = 0 (we don't have yaw)
                # Simplify: if allowed_side < some small fraction of horiz_dist => try move closer
                if allowed_side < (horiz_dist * 0.05):
                    # 向着坦克小步靠拢
                    step = min(50.0, math.hypot(dx, dy) * 0.02)
                    if to_tank_dist > 1e-6:
                        final_positions[i][0] += dx / to_tank_dist * step
                        final_positions[i][1] += dy / to_tank_dist * step
                        changed = True

            if not changed:
                break

        # 6. 生成返回结果并计算旋转角（指向坦克）
        result = []
        for i in range(3):
            new_x, new_y, new_z = final_positions[i]
            dx = tank_location[0] - new_x
            dy = tank_location[1] - new_y
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                angle_deg = math.degrees(math.atan2(dy, dx))
            else:
                angle_deg = 0.0
            result.append([[new_x, new_y, new_z], angle_deg])

        return result

    ###################################sunhongze###################################
    def bounding(self, image_1, image_2, image_3):

        result = []
        for i in range(3):
            result.append([0.5, 0.5, 0.5, 0.5])

        return result


def main():
    # 创建并运行服务器
    server = DroneWebSocketServer(host='localhost', port=8080)

    try:
        logger.info("Starting drone WebSocket server...")
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


if __name__ == "__main__":
    main()