# 不依赖 MoveIt 的分阶段抓取

当前实现使用已有 `HostArmIK` 和 `/wbc/target_joints`（含现有桥接），没有引入 MoveIt、自动碰撞规划或力控。每一段动作都需要操作员明确点击。收到点击位姿、生成候选、预览或 IK 预检都不会自动移动手臂。

## 界面操作

重启更新后的界面，在「手臂/手」→「分阶段抓取」子页可以看到抓取面板。「手动控制」「分阶段抓取」「视觉与演示」分别使用独立滚动区域；窗口较小时可滚动访问底部控件。

1. 在「分割」页点击彩色图，完成带真实深度和外参的估计。结果会自动送到抓取面板，但不会发送运动命令。
2. 填写抓取配置路径。使用模型候选，或通过「导入手型候选」导入当前观测对应的 JSON。
3. 如果没有灵巧手模型，先把手**手动调成希望的朝向**，再点击「按当前手朝向生成候选」。可以调整 base_link 下的 XYZ 接触偏移，单位 cm。此候选是人工设定，不是模型预测；目标以当前估计的物体原点为参考，PCA 的原点只代表可见表面几何中心。
4. 点击「预览抓取位置」。画面会标出 pregrasp、approach、lift 的 TCP 位置与坐标轴，数字区显示对应 XYZ 和旋转矩阵。确认手臂侧别、接触位置、接近方向与工作空间。
5. 确认本段手与路径周围无障碍，点击「张开手」。观察实际手部，再点「确认手已张开」。发送指令不等于动作完成。
6. 点击「预检下一段 IK」。成功后重新核对路径，勾选本段路径确认，再点击「执行本段」，到达预抓取位置。
7. 再次进行 IK 预检与执行，沿接近方向到达抓取位置。
8. 确认手周围无障碍，点击「闭合手」。观察物体是否确实被夹持，再点「确认已抓住物体」。这里是人工确认，没有力/触觉检测。
9. 对抬起段再次预检、确认、执行。系统核对实际关节及 TCP 反馈后才标记完成。

「停止 / 取消」会取消后续阶段；运动中改为保持当时实测关节位置，并等待连续新反馈确认静止，不会自动松手。这个软件停止不是硬件急停。反馈失效或停止未确认时保留控制锁，并提示使用机器人停止装置。正常结束保持 model 模式，不自动恢复 tracker/遥操作。

## 配置

将 `config/grasp_execution.example.json` 复制为 `config/grasp_execution.json`，或设置 `EAI_GRASP_PROFILE`。示例 `calibrated=false`，不会直接执行；示例工作空间、开合量和单位变换只是格式示例。

现在可直接在「手臂/手 → 抓取标定」完成这个过程：填写实际执行侧、六通道开合量、接触坐标系到 Link7 的平移和 RPY、以及实测工作空间；逐项复核后，点击「确认实测并启用抓取配置」。该按钮不会移动机器人，但会把 `calibrated=true` 写入当前配置。因此只有在实际测量完毕并复核后才勾选三个确认项。

「分阶段抓取」中的「启动 FK 反馈」仅启动关节状态到 TCP 的反馈节点。它不发布手臂动作。启动后等待约两秒，再点击「检查准备状态」。

必须根据实际机器人填写：

- `arm_side`：明确 left/right，不会自动改用另一只手臂。
- `hand_model`：与实际手型及候选的 `gripper` 完全一致。
- `base_frame`：当前链路使用 base_link，不能用相机坐标冒充。
- `tcp_link`：当前宿主机 IK 为 Link7_l / Link7_r，不是手指尖或掌心。
- `T_grasp_tcp`：`T_base_tcp = T_base_grasp @ T_grasp_tcp`。表示机器人 Link7 坐标系在抓取接触坐标系中的米制变换，需实测。不要未经测量填单位矩阵。
- `approach_axis_grasp`：抓取坐标系中的单位接近方向。预抓取点沿其反方向退开 `pregrasp_distance_m`，抬起沿 base_link +Z。
- `open_positions` / `close_positions`：沿现有 `HAND_JOINT_ORDER` 的 6 个 [0,1] 驱动量，分别是 thumb_rotation、thumb_bend、index、middle、ring、pinky（以当前驱动配置核对）。不同物体可能需要不同的闭合预设。本版本不会自动优化逐指接触。
- `urdf_path`：与 `host_fk_node.py` 使用的同一份 16 自由度 URDF。通过文件 SHA256 验证一致性。
- `planner_python`：能导入 numpy 与 pinocchio 的 Python。IK 在独立进程运行，不能指向一个没有 pinocchio 的环境。
- `workspace_m`：按实际安装测量的 TCP XYZ 范围。
- 速度、加速度、预抓取距离、抬起距离与过期时间均有范围检查。

相机标定仍按 `docs/click_pose.md` 配置。没有 `T_base_camera` 时，抓取面板拒绝执行；不会用默认 TF 别名猜测外参。

## 候选导入格式

必须绑定当前界面显示的 `observation_id`，避免把旧图候选用于新图：

```json
{
  "ok": true,
  "observation_id": "从当前点击结果复制",
  "gripper": "与实际手型配置一致",
  "candidates": [
    {
      "T_camera_grasp": [[1,0,0,0], [0,1,0,0], [0,0,1,0.5], [0,0,0,1]],
      "model_score": 0.8
    }
  ]
}
```

上面的位姿只是 JSON 格式示例。导入模型分数不是成功率；机器人将另行应用手的 `T_grasp_tcp` 偏移。带有自定义开合值的候选必须与当前已标定配置一致，否则拒绝执行。

## 实际检查范围

宿主机预检对 Cartesian 路径逐点求 IK，检查位置和旋转误差、URDF 关节限位、IK 解连续性、另一只手臂/腰部保持不动以及插值 TCP 工作空间。轨迹使用五次平滑插值，按配置限制关节速度和加速度。执行前检查模型与起点，执行中检查实测关节偏差、使能丢失和控制循环延迟。

**这些检查不包含机器人自碰撞、手与环境碰撞或物体接触稳定性。** 该模式是人工监督的分阶段操作，不是无人值守自动抓取。后续 MoveIt 可以替换预检/执行后端，本次没有安装或调用它。

## 反馈与启用

本次 `host_fk_node.py` 增加 `arm_received_at`、`waist_received_at` 和 `urdf_sha256`，区分实测反馈和命令。请在机器人静止、没有执行任务时重启该 FK 进程与界面。旧的 FK 状态文件会被新执行器拒绝；不能把时间戳手工改新来绕过检查。需要持续的手臂和腰部实测反馈。

运行中的 GUI/机器人服务不会被代码修改自动重启。本次未发送机器人动作，也未进行实机抓取验收。

测试：

```bash
/usr/bin/python3 eai/test_click_pose.py -v
/usr/bin/python3 eai/test_grasp_execution.py -v
QT_QPA_PLATFORM=offscreen /usr/bin/python3 eai/test_grasp_execution_ui.py -v
```

状态机、轨迹播放器和 Qt 测试使用合成数据/假机器人对象。它们验证控制逻辑，不能代替真实标定、真实 IK 环境验证或实机验收。预检请求和结果保存在 `.cache/grasp_execution/`。

## 本机已验证的 IK 环境

项目内 `.cache/grasp_ik_env/bin/python`（Python 3.10）已安装 Pinocchio 3.4.0，并固定 urdfdom 4.0.1、tinyxml2 10.0.0，避免新版动态库 ABI 不匹配。启动脚本和预检进程优先使用此环境的 cmeel Python 包与动态库，避免 ROS overlay 中旧 Pinocchio 抢先加载。未修改系统 Python。

已验证：该环境加载 ROS 的 rclpy、geometry_msgs、sensor_msgs，并成功构建当前 16 自由度 FK 模型；双臂离线 IK 路径测试通过。`genie_msgs` 仍需原有机器人 overlay；启动脚本通过 `EAI_GENIE_SETUP` 或已有路径加载。没有启动 FK 节点或发布机器人命令。

离线 IK 测试：`eai/.cache/grasp_ik_env/bin/python eai/test_host_grasp_plan.py -v`。
