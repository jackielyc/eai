#!/usr/bin/env python3
"""
方案 A：多视角照片 / 外部 mesh → FoundationPose 可用 .obj

流程 1（照片重建，需 COLMAP）:
  python3 photogrammetry_reconstruct.py --images /path/to/photos --name my_object

流程 2（Meshroom / RealityCapture 等已导出 mesh）:
  python3 photogrammetry_reconstruct.py --import-mesh model.obj --name my_object

输出:
  meshes/<name>/reconstructed.obj
  meshes/<name>/manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger("mesh_reconstruct")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MESH_SUFFIXES = {".obj", ".ply", ".glb", ".stl", ".off"}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def find_colmap() -> Optional[str]:
    return shutil.which("colmap")


def list_images(images_dir: Path) -> List[Path]:
    files = [
        p
        for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return files


def stage_images(images: List[Path], staging_dir: Path) -> Path:
    """COLMAP 对中文/空格路径较敏感，复制到 staging 并使用简单文件名。"""
    staging_dir.mkdir(parents=True, exist_ok=True)
    for old in staging_dir.glob("*"):
        if old.is_file():
            old.unlink()
    for i, src in enumerate(images):
        dst = staging_dir / f"img_{i:04d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
    return staging_dir


def run_colmap_sparse(images_dir: Path, work_dir: Path) -> Path:
    """COLMAP SfM（CPU），返回 sparse/0 模型目录。"""
    colmap = find_colmap()
    if not colmap:
        raise RuntimeError(
            "未找到 colmap。请安装: sudo apt install -y colmap\n"
            "或: bash run_mesh_reconstruct.sh --install-deps"
        )

    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("COLMAP feature_extractor …")
    subprocess.run(
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(db_path),
            "--image_path",
            str(images_dir),
            "--ImageReader.single_camera",
            "1",
        ],
        check=True,
    )

    LOG.info("COLMAP exhaustive_matcher …")
    subprocess.run(
        [colmap, "exhaustive_matcher", "--database_path", str(db_path)],
        check=True,
    )

    LOG.info("COLMAP mapper …")
    subprocess.run(
        [
            colmap,
            "mapper",
            "--database_path",
            str(db_path),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_dir),
        ],
        check=True,
    )

    models = sorted(sparse_dir.iterdir())
    model_dirs = [p for p in models if p.is_dir() and (p / "cameras.bin").exists()]
    if not model_dirs:
        raise RuntimeError(
            "COLMAP mapper 未产生有效模型。请增加照片数量/重叠度（建议 ≥12 张、60%% 重叠）。"
        )
    model_dir = model_dirs[0]
    LOG.info("使用 sparse 模型: %s", model_dir)
    return model_dir


def export_sparse_point_cloud(model_dir: Path, ply_out: Path) -> Path:
    colmap = find_colmap()
    if not colmap:
        raise RuntimeError("未找到 colmap")
    ply_out.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("导出稀疏点云 → %s", ply_out)
    subprocess.run(
        [
            colmap,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(ply_out),
            "--output_type",
            "PLY",
        ],
        check=True,
    )
    if not ply_out.is_file() or ply_out.stat().st_size < 100:
        raise RuntimeError("稀疏点云导出失败或点数过少")
    return ply_out


def poisson_mesh_from_ply(ply_path: Path, mesh_out: Path, poisson_depth: int = 9) -> Path:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "需要 open3d: python3 -m pip install -r requirements-mesh.txt"
        ) from exc

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) < 100:
        raise RuntimeError(f"点云点数过少 ({len(pcd.points)})，无法网格化")

    LOG.info("估计法线并 Poisson 重建 (depth=%d) …", poisson_depth)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )
    if len(mesh.vertices) == 0:
        raise RuntimeError("Poisson 重建失败")

    densities = np.asarray(densities)
    if len(densities) == len(mesh.vertices):
        keep = densities > np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(~keep)

    mesh.compute_vertex_normals()
    mesh_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(mesh_out), mesh)
    LOG.info("Poisson mesh: %d 顶点, %d 面", len(mesh.vertices), len(mesh.triangles))
    return mesh_out


def load_trimesh(mesh_path: Path):
    import trimesh

    loaded = trimesh.load(str(mesh_path), process=False, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if hasattr(g, "vertices")]
        if not geoms:
            raise RuntimeError(f"Scene 中无 mesh: {mesh_path}")
        loaded = trimesh.util.concatenate(geoms)
    if not hasattr(loaded, "vertices") or len(loaded.vertices) == 0:
        raise RuntimeError(f"无法加载 mesh: {mesh_path}")
    return loaded


def postprocess_for_foundationpose(
    mesh_in: Path,
    obj_out: Path,
    target_extent_m: float,
) -> Dict[str, Any]:
    """
    居中 + 缩放到目标最大边长（米），导出 .obj 供 FoundationPose 使用。
    """
    import trimesh

    mesh = load_trimesh(mesh_in)
    centroid = mesh.centroid.copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) - centroid

    extents = mesh.extents
    max_extent = float(np.max(extents))
    if max_extent <= 1e-9:
        raise RuntimeError("mesh 尺寸异常（接近 0）")

    scale = float(target_extent_m) / max_extent
    mesh.vertices *= scale

    obj_out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(obj_out))

    meta = {
        "source_mesh": str(mesh_in.resolve()),
        "output_obj": str(obj_out.resolve()),
        "centroid_removed": centroid.tolist(),
        "scale_to_meters": scale,
        "target_extent_m": target_extent_m,
        "final_extents_m": mesh.extents.tolist(),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
    }
    LOG.info(
        "后处理完成: extent≈%.3fm, 顶点=%d → %s",
        float(np.max(mesh.extents)),
        meta["vertex_count"],
        obj_out,
    )
    return meta


def reconstruct_from_images(
    images_dir: Path,
    out_dir: Path,
    name: str,
    target_extent_m: float,
    poisson_depth: int,
    min_images: int,
) -> Dict[str, Any]:
    images = list_images(images_dir)
    if len(images) < min_images:
        raise ValueError(
            f"至少需要 {min_images} 张照片，当前 {len(images)} 张: {images_dir}"
        )

    work_dir = out_dir / name / "colmap_work"
    staging = work_dir / "images_staged"
    stage_images(images, staging)

    model_dir = run_colmap_sparse(staging, work_dir)
    sparse_ply = work_dir / "sparse_points.ply"
    export_sparse_point_cloud(model_dir, sparse_ply)

    raw_mesh = work_dir / "poisson_raw.ply"
    poisson_mesh_from_ply(sparse_ply, raw_mesh, poisson_depth=poisson_depth)

    obj_out = out_dir / name / "reconstructed.obj"
    meta = postprocess_for_foundationpose(raw_mesh, obj_out, target_extent_m)
    meta.update(
        {
            "method": "colmap_sparse_poisson",
            "name": name,
            "image_count": len(images),
            "images_dir": str(images_dir.resolve()),
            "sparse_model": str(model_dir.resolve()),
        }
    )
    return meta


def import_external_mesh(
    mesh_path: Path,
    out_dir: Path,
    name: str,
    target_extent_m: float,
) -> Dict[str, Any]:
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if mesh_path.suffix.lower() not in MESH_SUFFIXES:
        LOG.warning("非常见 mesh 后缀 %s，仍尝试加载", mesh_path.suffix)

    obj_out = out_dir / name / "reconstructed.obj"
    meta = postprocess_for_foundationpose(mesh_path, obj_out, target_extent_m)
    meta.update(
        {
            "method": "import_mesh",
            "name": name,
            "import_path": str(mesh_path.resolve()),
        }
    )
    return meta


def write_manifest(out_dir: Path, name: str, meta: Dict[str, Any]) -> Path:
    manifest_path = out_dir / name / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    LOG.info("manifest → %s", manifest_path)
    return manifest_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="方案 A：照片/Meshroom mesh → FoundationPose .obj"
    )
    parser.add_argument(
        "--images",
        type=Path,
        help="多视角照片目录（jpg/png，建议 ≥12 张）",
    )
    parser.add_argument(
        "--import-mesh",
        type=Path,
        help="已重建的 mesh（Meshroom/RealityCapture 导出 .obj/.ply）",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="物体名称，输出到 meshes/<name>/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir / "meshes",
        help="mesh 输出根目录（默认 eai/meshes）",
    )
    parser.add_argument(
        "--target-extent-m",
        type=float,
        default=0.15,
        help="后处理：mesh 最大边长（米），默认 0.15（约 15cm 物体）",
    )
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=9,
        help="Poisson 重建深度（越大越细，默认 9）",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=8,
        help="最少照片张数（默认 8）",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    if bool(args.images) == bool(args.import_mesh):
        LOG.error("请指定 --images 或 --import-mesh 之一（且只能选一个）")
        return 2

    name = args.name.strip()
    if not name or "/" in name or "\\" in name:
        LOG.error("无效的 --name: %s", name)
        return 2

    out_dir = args.out_dir.expanduser().resolve()
    try:
        if args.import_mesh:
            meta = import_external_mesh(
                args.import_mesh.expanduser().resolve(),
                out_dir,
                name,
                args.target_extent_m,
            )
        else:
            meta = reconstruct_from_images(
                args.images.expanduser().resolve(),
                out_dir,
                name,
                args.target_extent_m,
                args.poisson_depth,
                args.min_images,
            )
        write_manifest(out_dir, name, meta)
        obj_path = meta["output_obj"]
        print(json.dumps({"ok": True, "mesh": obj_path, "manifest": str(out_dir / name / "manifest.json")}))
        return 0
    except Exception as exc:
        LOG.error("%s", exc)
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
