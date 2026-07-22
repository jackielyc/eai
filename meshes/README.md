# 方案 A：无 CAD mesh → 照片重建 → FoundationPose

当没有厂家 CAD 模型时，用手机/相机绕物体拍多张照片，重建 mesh 后再做 6D 位姿估计。

## 快速开始

### Web 界面（推荐）

```bash
cd eai
bash run_cad_ui.sh --install-deps   # 首次
bash run_cad_ui.sh                  # 打开 http://127.0.0.1:7860
```

在浏览器里上传多视角照片或外部 mesh，填写物体名与真实尺寸（最大边长），点生成即可。

### 命令行

```bash
cd eai

# 1. 安装依赖（COLMAP + Python）
bash run_mesh_reconstruct.sh --install-deps

# 2a. 从照片重建
bash run_mesh_reconstruct.sh --images /path/to/photos --name my_object

# 2b. 或导入 Meshroom / RealityCapture 已导出的 mesh
bash run_mesh_reconstruct.sh --import-mesh ~/Downloads/model.obj --name my_object

# 3. 启动 FoundationPose
bash run_foundationpose.sh --host 0.0.0.0 --mesh meshes/my_object/reconstructed.obj
```

一键重建并启动 FP：

```bash
bash run_mesh_reconstruct.sh --images ~/photos/cup --name cup --start-fp
```

## 拍照建议

| 项目 | 建议 |
|------|------|
| 数量 | ≥12 张（最少 8 张） |
| 重叠 | 相邻视角约 60% 重叠 |
| 环绕 | 水平一圈 + 俯视/仰视若干张 |
| 背景 | 纹理丰富、避免纯白墙 |
| 物体 | 尽量静止；哑光、非透明最佳 |
| 对焦 | 清晰、曝光正常；关闭 HDR 有时更稳 |

照片目录示例：

```
~/photos/cup/
  IMG_001.jpg
  IMG_002.jpg
  ...
```

## 输出文件

```
meshes/<name>/
  reconstructed.obj   # 给 FoundationPose 用（已居中、缩放到米）
  manifest.json       # 尺度/来源记录
  colmap_work/        # 照片重建中间结果（仅 --images 流程）
```

## 尺度说明

默认 `--target-extent-m 0.15`：重建 mesh 的**最大边长**缩放到 0.15 m（15 cm）。

若真实物体更大/更小，重建后位姿的**绝对平移**会与真实尺寸成比例；可在 viewer 里用已知尺寸校正，或重建时指定：

```bash
bash run_mesh_reconstruct.sh --images ... --name bottle --target-extent-m 0.25
```

## Meshroom 用户

1. Meshroom 正常跑完 Photogrammetry
2. 导出 **TexturedMesh** 为 `.obj` 或 `.ply`
3. 导入后处理（无需 COLMAP）：

```bash
bash run_mesh_reconstruct.sh --import-mesh /path/to/Meshroom.obj --name my_object
```

## 故障排查

| 现象 | 处理 |
|------|------|
| `未找到 colmap` | `bash run_mesh_reconstruct.sh --install-deps` |
| mapper 失败 | 增加照片、提高重叠、改善光照 |
| 点云过少 | 物体太光滑/透明，加纹理贴纸或喷哑光漆 |
| FP 位姿飘 | 检查 `--target-extent-m` 是否与真实尺寸接近 |
