# 项目说明

本文档记录当前目录中各文件的作用、相互关系和维护注意事项，供后续协作者或智能体快速理解项目。

## 项目概览

这是一个 Python 无人机围捕/控制服务项目。项目通过 WebSocket 接收外部平台发送的无人机组数据，包括无人机位置、朝向、图像和坦克位置；服务端可使用 YOLO 对无人机图像进行坦克检测，估计坦克平面坐标，再调用向量场控制器生成每架无人机的新位置和朝向。

整体目标可以概括为：

1. 接收平台侧的无人机和坦克状态。
2. 从无人机图像中检测坦克并估计坦克位置。
3. 使用向量场算法计算无人机运动控制量。
4. 返回无人机下一步位置和绕 z 轴旋转角度。

## 当前文件结构

```text
EncirclementControllerv1/
├─ AGENTS.md
├─ controller.py
├─ server_vec_ori.py
├─ server_vec_final.py
├─ server_vec_final_2drone.py
└─ __pycache__/
   └─ controller.cpython-39.pyc
```

## 文件职责

### controller.py

核心控制算法文件，定义 `VectorFieldController`。

主要职责：

- 根据无人机二维位置和坦克二维位置生成每架无人机的速度向量。
- 使用吸引力驱动无人机接近坦克。
- 使用无人机间斥力避免机间距离过近。
- 使用队形保持力让无人机围绕坦克形成分布。
- 使用边界斥力避免无人机越出约定区域。
- 根据坦克历史速度进行短期位置预测。
- 根据距离阶段切换 `approach`、`surround`、`maintain` 三种控制阶段。

关键入口：

- `VectorFieldController.compute_control(uavs, tank, tank_velocity=None)`

关键参数：

- `capture_radius = 800.0`：任务要求的坦克捕获半径。
- `min_inter_uav = 300.0`：无人机之间的最小允许距离。
- `boundary_margin = 400.0`：边界斥力开始生效的边距。
- `base_max_speed = 200.0`：基础最大速度。
- `max_speed_cap = 800.0`：速度上限。

### server_vec_ori.py

原始三无人机 WebSocket 服务版本。

主要特点：

- 默认服务地址为 `localhost:8080`。
- 接收三架无人机的数据。
- 主要使用平台传入的坦克位置进行路径计算。
- YOLO 检测器初始化代码被注释掉，整体更像是不启用图像检测的基线版本。
- 仍然保留了检测相关函数，但主流程没有实际使用检测结果。

### server_vec_final.py

三无人机最终版服务文件。

主要特点：

- 默认服务地址为 `localhost:8080`。
- 接收三架无人机的数据。
- 初始化 YOLO 检测器：

```python
self.detector = YOLO("F:/Platform/Server/detec/best.pt")
```

- 对三张无人机图像分别执行检测。
- 使用 `read_detector1()` 和 `estimate_tank_from_detections()` 估计坦克 XY 坐标。
- 如果检测估计位置与平台坦克位置偏差不超过约 300，则使用检测估计位置；否则退回平台传入的坦克位置。
- 调用 `path()` 生成三架无人机的新位置和朝向。

### server_vec_final_2drone.py

二无人机适配版服务文件。

主要特点：

- 默认服务地址为 `localhost:8080`。
- 使用 `DRONE_COUNT = 2` 适配二机输入和输出。
- 结构基本继承 `server_vec_final.py`。
- 只对两张无人机图像执行 YOLO 检测。
- 调用二机版本的 `path()` 生成两架无人机的新位置和朝向。

### __pycache__

Python 自动生成的字节码缓存目录，不属于业务源码。通常不需要手动维护。

## 模块关系

三个服务文件都依赖 `controller.py`：

```text
server_vec_ori.py
server_vec_final.py
server_vec_final_2drone.py
        |
        v
controller.py
```

`controller.py` 不依赖服务文件，是底层控制算法模块。

运行时通常只启动一个服务文件，因为三个服务文件默认都绑定同一个端口 `localhost:8080`。

## 主要数据流

```text
外部平台客户端
  -> WebSocket 发送 drone_group_data
  -> 服务端解析 JSON
  -> 解码 base64 图像为 PIL Image
  -> 读取无人机位置、角度和坦克位置
  -> YOLO 检测坦克
  -> 根据检测框中心和无人机姿态估计坦克 XY
  -> path() 估计坦克速度并准备无人机二维坐标
  -> VectorFieldController.compute_control()
  -> 后处理无人机位置，包括边界限制、机间距修正和接近坦克约束
  -> 返回 drone_control_response
```

## WebSocket 消息约定

服务端主要处理消息类型：

```json
{
  "type": "drone_group_data"
}
```

响应消息类型：

```json
{
  "type": "drone_control_response"
}
```

响应中包含：

- `tank_location`：坦克位置。
- `drone_commands`：每架无人机的新位置和旋转角度。
- `drone_boundingbox`：检测框占位数据，目前多处返回固定值 `0.5`。

## 关键算法说明

### 坦克位置估计

服务文件中使用相机参数估计坦克地面位置：

- 图像宽高：`IMG_W = 940`，`IMG_H = 540`。
- 水平视场角：`FOV_H_DEG = 90.0`。
- 根据 FOV 计算像素焦距 `f_px`。
- 根据检测框中心点、无人机位置、无人机 yaw 和高度差估计坦克 XY。

### 路径计算

`path()` 的核心流程：

1. 组装无人机二维位置和坦克二维位置。
2. 根据历史坦克位置估计坦克平面速度。
3. 调用 `controller.compute_control()` 获取速度向量。
4. 使用固定时间步长 `dt = 0.25` 计算新位置。
5. 限制无人机位置不越过边界。
6. 修正过近的无人机间距。
7. 尽量让无人机接近坦克捕获半径。
8. 计算每架无人机指向坦克的旋转角度。

## 运行入口

三个服务文件都有 `main()` 和 `if __name__ == "__main__"` 入口。

常见运行方式：

```powershell
python server_vec_final.py
```

或二机版本：

```powershell
python server_vec_final_2drone.py
```

注意：不要同时启动多个服务文件，除非修改端口，否则会因为都使用 `localhost:8080` 而冲突。

## 依赖推断

当前目录没有发现 `requirements.txt`。从源码看，项目至少依赖：

- `numpy`
- `Pillow`
- `websockets`
- `ultralytics`

标准库使用：

- `asyncio`
- `json`
- `base64`
- `logging`
- `io`
- `time`
- `math`
- `random`
- `typing`

## 已观察到的维护注意事项

1. 当前目录不是 Git 仓库，`git status` 无法使用。
2. `server_vec_final.py` 和 `server_vec_ori.py` 中存在旧函数 `read_detector()`，其中调用了未定义的 `estimate_tank_position()`。主流程实际使用的是 `read_detector1()`，所以正常路径下暂时不会触发，但后续清理时应处理。
3. 三个服务文件之间有大量重复代码，后续可考虑抽取通用 WebSocket 解析、YOLO 检测、坦克估计和路径后处理逻辑。
4. YOLO 模型路径是硬编码绝对路径 `F:/Platform/Server/detec/best.pt`，如果部署环境不同，需要调整或改成配置项。
5. 多个文件默认使用同一端口 `8080`，同时运行会端口冲突。
6. `drone_boundingbox` 目前返回固定占位值，并未真正返回 YOLO 检测框。
7. 终端中曾出现中文注释乱码现象，但 PowerShell 能解析到源码中的中文内容。若后续维护，建议统一文件编码为 UTF-8。

## 推荐主线

如果需要三架无人机控制，优先查看和运行：

```text
server_vec_final.py
```

如果需要两架无人机控制，优先查看和运行：

```text
server_vec_final_2drone.py
```

如果需要理解控制算法，优先查看：

```text
controller.py
```
