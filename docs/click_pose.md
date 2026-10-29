# 点击物体位姿与抓取候选

## 当前实现

「分割」页新增「点击估计（仅展示）」和标定文件输入框。点击 ROS 彩色预览后，使用该预览对应的原分辨率彩色帧，与采集时间戳最近的原始深度配对。默认拒绝超过 80 ms 的失步和接收时间超过 0.8 秒的旧帧。点击坐标从预览映射回原图，计算不使用缩小的深度预览。

彩色话题启用时，后台订阅可发现的配对深度，不要求显示深度面板。新流程不会调用 `_store_segment_target`、不会回填 TCP，也不会触发旧的自动移动逻辑。旧按钮仍可记录位姿，但已移除自动移动左臂。分阶段抓取入口位于「手臂/手」，见 `docs/grasp_execution.md`。勿把旧「接触 TCP」与新物体姿态混为一谈。

输出包含：相机光学坐标下 `T_camera_object`（米、右手系）、分割有效深度比例、RGB-D 时间差、方法和局限。可配置 `T_base_camera` 输出机器人坐标。PCA 仅是可见表面几何预览，不能表示完整物体或唯一语义朝向。Any6D 无 mesh 的现有适配使用局部凸包，尚未实现官方参考图重建，不能视为完整 Any6D 精度。

结果快照叠加掩码和物体 XYZ 轴，明确标记不是 TCP。输入和结果保存至 `.cache/click_pose/<timestamp>/`：`rgb.png`、`inputs.npz`、`result.json`。快照不跟随实时物体运动；需要更新时重新点击。

## 必须提供真实标定

复制 `config/click_pose_calibration.example.json` 为 `config/click_pose_calibration.json`，或在界面选择其他 JSON 路径（也可设置 `EAI_CLICK_CALIBRATION`）。模板的零焦距和 false 状态故意不可用，不能当作真实标定。

- `K` 是原始采集分辨率对应的已去畸变彩色相机 3×3 内参。
- `width/height` 必须匹配原图，不是预览尺寸。
- 输入彩色必须已去畸变，深度必须注册到同一彩色相机。完成后才设置 `rectified` 和 `depth_registered_to_color` 为 true。相同分辨率不代表已经对齐。
- `optical_frame` 必须与两路图像的实际 frame_id 一致；不能只修改 frame_id 掩盖未完成的对齐。
- 支持 16UC1 毫米或 32FC1 米深度，不猜其他编码单位。
- `T_base_camera` 可省略/null。提供时必须是从相机光学坐标到 `base_frame` 的米制 4×4 刚体变换。不要把反方向的外参填进来。

目前不会自动执行去畸变或深度注册；这些必须由相机发布链路完成。没有上述数据时新流程输出具体错误，不用估算焦距或固定 0.45 米继续估计。

## 分割与物体位姿模型

沿用上方分割、位姿后端和 HTTP 配置：

- 几何分割 + PCA 可用于检查数据链路，仅标为预览。
- 选择 SAM3 点提示，使用现有 `sam3_segment_worker.py`；新点击总是传点击点，不使用概念文本匹配多个物体。需要独立 Python 环境与 checkpoint。
- FoundationPose 必须提供当前目标自身、尺寸正确的模型；不要复用默认盒子模型估计其他物体。
- Any6D 沿用现有 worker；无模型返回结果明确标为 partial_hull_estimate。
- 模型缺少环境/权重或返回错误时，显示错误，不自动回退成成功的 PCA 结果。

本次没有下载权重、安装模型环境、实现单图完整物体重建或完成神经模型真机验收。

官方依据：
- https://github.com/facebookresearch/sam3
- https://github.com/NVlabs/FoundationPose
- https://github.com/taeyeopl/Any6D

## 可选 GraspGen 候选

新增 `graspgen_worker.py` 调用官方 `GraspGenSampler.run_inference`。在官方 GraspGen 环境装好依赖及对应夹爪 checkpoint 后，在某个相机配置中添加：

```json
"graspgen": {
  "python": "/absolute/path/to/graspgen-env/bin/python",
  "config": "/absolute/path/to/checkpoint/gripper_config.yaml",
  "gripper": "EXACT_CHECKPOINT_GRIPPER_NAME"
}
```

输出 `T_camera_grasp` 是该模型定义的夹爪坐标系在相机系中的位姿，不是你的机器人 TCP。worker 检查请求手型是否与 checkpoint 一致，并按模型分数返回至多 20 个候选。模型分数不是实测成功率。候选未经过全场景碰撞、IK、TCP 标定与预抓取路径检查，因此 `execution_ready=false`。候选为空或模型失败也会明确显示。

当前六驱动灵巧手尚未做抓取模型适配、逐指关节求解或接触验证；新执行面板支持人工接触点候选及已标定的六通道开合预设。不能配置一个平行夹爪名称就当作完成了灵巧手抓取。官方接口：https://github.com/NVlabs/GraspGen/blob/main/grasp_gen/grasp_server.py

## 验证与启用

```bash
(cd eai && python3 -m unittest tests.test_click_pose -v)
python3 -m compileall -q eai/eai_app
```

修改不会热更新正在运行的 GUI。完成真实标定和模型配置后，重新启动界面，再点击头部彩色图验证。先用静止、有实际尺寸可量测的物体检查米制位置和多次点击稳定性；这些单元测试不等于实机抓取验收。
