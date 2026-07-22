#!/usr/bin/env python3
"""
CAD / Mesh 生成 Web 界面（仅标准库 + 现有 mesh 依赖）

把多视角照片或外部 mesh 转成 FoundationPose 可用的 .obj。

用法:
  bash run_cad_ui.sh
  python3 cad_generate_ui.py --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import shutil
import sys
import tempfile
import threading
import traceback
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "meshes"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MESH_SUFFIXES = {".obj", ".ply", ".glb", ".stl", ".off"}

# 全局任务状态（单机本地工具足够）
_JOB_LOCK = threading.Lock()
_JOB: Dict[str, Any] = {
    "status": "idle",  # idle | running | done | error
    "message": "等待操作…",
    "progress": "",
    "obj": None,
    "preview": None,
    "manifest": None,
}


def _safe_name(name: str) -> str:
    name = (name or "").strip().replace(" ", "_")
    for ch in '/\\:*?"<>|':
        name = name.replace(ch, "_")
    return name or f"object_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _set_job(**kwargs: Any) -> None:
    with _JOB_LOCK:
        _JOB.update(kwargs)


def _get_job() -> Dict[str, Any]:
    with _JOB_LOCK:
        return dict(_JOB)


def _render_mesh_preview(obj_path: Path, out_png: Path, max_faces: int = 8000) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import trimesh
    except Exception:
        return None

    try:
        mesh = trimesh.load(str(obj_path), process=False, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            geoms = [g for g in mesh.geometry.values() if hasattr(g, "vertices")]
            if not geoms:
                return None
            mesh = trimesh.util.concatenate(geoms)
        if len(mesh.faces) == 0:
            return None

        faces = np.asarray(mesh.faces)
        verts = np.asarray(mesh.vertices)
        if len(faces) > max_faces:
            idx = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
            faces = faces[idx]

        fig = plt.figure(figsize=(5.2, 5.2), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        tris = verts[faces]
        coll = Poly3DCollection(
            tris,
            alpha=0.88,
            facecolor="#4f7ecf",
            edgecolor="#1e3358",
            linewidths=0.12,
        )
        ax.add_collection3d(coll)
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = float(np.max(maxs - mins)) * 0.55 + 1e-6
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=22, azim=35)
        ax.set_title(obj_path.parent.name)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return out_png
    except Exception:
        return None


def list_existing_meshes(out_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not out_dir.is_dir():
        return rows
    for d in sorted(out_dir.iterdir()):
        if not d.is_dir():
            continue
        obj = d / "reconstructed.obj"
        if not obj.is_file():
            continue
        method, extent = "-", "-"
        meta_path = d / "manifest.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                method = str(meta.get("method", "-"))
                fe = meta.get("final_extents_m")
                if isinstance(fe, list) and fe:
                    extent = f"{max(float(x) for x in fe):.3f} m"
            except Exception:
                pass
        preview = d / "preview.png"
        rows.append(
            {
                "name": d.name,
                "method": method,
                "extent": extent,
                "obj": str(obj),
                "preview": str(preview) if preview.is_file() else "",
            }
        )
    return rows


def _run_photos_job(
    image_paths: List[Path],
    name: str,
    target_extent_m: float,
    poisson_depth: int,
    min_images: int,
    out_dir: Path,
) -> None:
    try:
        from photogrammetry_reconstruct import reconstruct_from_images, write_manifest

        _set_job(status="running", message="正在 COLMAP 重建…", progress="staging")
        with tempfile.TemporaryDirectory(prefix="cad_photos_") as tmp:
            staging = Path(tmp) / "images"
            staging.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(image_paths):
                dst = staging / f"img_{i:04d}{src.suffix.lower()}"
                shutil.copy2(src, dst)

            if len(image_paths) < min_images:
                raise ValueError(f"照片不足：{len(image_paths)} < {min_images}")

            _set_job(progress="colmap / poisson")
            meta = reconstruct_from_images(
                staging,
                out_dir,
                name,
                target_extent_m,
                poisson_depth,
                min_images,
            )
            write_manifest(out_dir, name, meta)

        obj_path = Path(meta["output_obj"])
        preview = _render_mesh_preview(obj_path, out_dir / name / "preview.png")
        msg = (
            f"完成：{obj_path}\n"
            f"顶点/面: {meta.get('vertex_count')} / {meta.get('face_count')}\n"
            f"最大边长: {max(meta.get('final_extents_m') or [0]):.4f} m\n"
            f"启动 FP: bash run_foundationpose.sh --host 0.0.0.0 --mesh {obj_path}"
        )
        _set_job(
            status="done",
            message=msg,
            progress="done",
            obj=str(obj_path),
            preview=str(preview) if preview else None,
            manifest=str(out_dir / name / "manifest.json"),
        )
    except Exception as exc:
        _set_job(
            status="error",
            message=f"重建失败: {exc}",
            progress="",
            obj=None,
            preview=None,
        )
        traceback.print_exc()


def _run_mesh_job(
    mesh_path: Path,
    name: str,
    target_extent_m: float,
    out_dir: Path,
) -> None:
    try:
        from photogrammetry_reconstruct import import_external_mesh, write_manifest

        _set_job(status="running", message="正在后处理 mesh…", progress="import")
        meta = import_external_mesh(mesh_path, out_dir, name, target_extent_m)
        write_manifest(out_dir, name, meta)
        obj_path = Path(meta["output_obj"])
        preview = _render_mesh_preview(obj_path, out_dir / name / "preview.png")
        msg = (
            f"完成：{obj_path}\n"
            f"顶点/面: {meta.get('vertex_count')} / {meta.get('face_count')}\n"
            f"最大边长: {max(meta.get('final_extents_m') or [0]):.4f} m\n"
            f"启动 FP: bash run_foundationpose.sh --host 0.0.0.0 --mesh {obj_path}"
        )
        _set_job(
            status="done",
            message=msg,
            progress="done",
            obj=str(obj_path),
            preview=str(preview) if preview else None,
            manifest=str(out_dir / name / "manifest.json"),
        )
    except Exception as exc:
        _set_job(
            status="error",
            message=f"导入失败: {exc}",
            progress="",
            obj=None,
            preview=None,
        )
        traceback.print_exc()


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CAD / Mesh 生成</title>
<style>
  :root {
    --bg: #0f1419;
    --panel: #1a2332;
    --line: #2c3b52;
    --text: #e8eef7;
    --muted: #8fa3bf;
    --accent: #3d7eff;
    --accent2: #2bb673;
    --danger: #e25555;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "IBM Plex Sans", "Noto Sans SC", system-ui, sans-serif;
    background:
      radial-gradient(900px 420px at 10% -10%, #1c3358 0%, transparent 55%),
      radial-gradient(700px 380px at 100% 0%, #163528 0%, transparent 50%),
      var(--bg);
    color: var(--text); min-height: 100vh;
  }
  main { max-width: 1080px; margin: 0 auto; padding: 28px 20px 48px; }
  h1 { font-size: 1.65rem; margin: 0 0 6px; letter-spacing: -0.02em; }
  .sub { color: var(--muted); margin-bottom: 22px; line-height: 1.5; }
  .grid { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 16px; }
  @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: color-mix(in srgb, var(--panel) 92%, black);
    border: 1px solid var(--line); border-radius: 14px; padding: 18px;
  }
  .tabs { display: flex; gap: 8px; margin-bottom: 14px; }
  .tab {
    background: transparent; border: 1px solid var(--line); color: var(--muted);
    padding: 8px 14px; border-radius: 999px; cursor: pointer;
  }
  .tab.active { color: var(--text); border-color: var(--accent); background: #243553; }
  label { display: block; font-size: 0.85rem; color: var(--muted); margin: 12px 0 6px; }
  input[type=text], input[type=number], input[type=file] {
    width: 100%; background: #101821; border: 1px solid var(--line); color: var(--text);
    border-radius: 10px; padding: 10px 12px;
  }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .btn {
    margin-top: 16px; width: 100%; border: 0; border-radius: 10px; padding: 12px 14px;
    background: linear-gradient(135deg, var(--accent), #5a9bff); color: white;
    font-weight: 600; cursor: pointer;
  }
  .btn:disabled { opacity: 0.55; cursor: not-allowed; }
  .btn.secondary { background: #243041; border: 1px solid var(--line); }
  pre {
    white-space: pre-wrap; background: #101821; border: 1px solid var(--line);
    border-radius: 10px; padding: 12px; min-height: 110px; color: #cfe0f7; font-size: 0.9rem;
  }
  img.preview {
    width: 100%; max-height: 360px; object-fit: contain; background: #0c1118;
    border-radius: 10px; border: 1px solid var(--line);
  }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 500; }
  a { color: #7eb0ff; }
  .hint { font-size: 0.82rem; color: var(--muted); margin-top: 8px; line-height: 1.45; }
  .status-pill {
    display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem;
    border: 1px solid var(--line); color: var(--muted);
  }
  .status-pill.running { color: #ffd27a; border-color: #80622a; }
  .status-pill.done { color: #8dffc0; border-color: #2d6b4c; }
  .status-pill.error { color: #ff9b9b; border-color: #7a2f2f; }
</style>
</head>
<body>
<main>
  <h1>CAD / Mesh 生成</h1>
  <p class="sub">多视角照片或外部 mesh → FoundationPose 可用的 <code>reconstructed.obj</code>（居中并缩放到米制）。</p>

  <div class="grid">
    <section class="card">
      <div class="tabs">
        <button class="tab active" data-tab="photos" type="button">照片重建</button>
        <button class="tab" data-tab="mesh" type="button">导入 Mesh</button>
      </div>

      <form id="form-photos">
        <label>多视角照片（建议 ≥12 张）</label>
        <input type="file" name="images" accept="image/*" multiple required/>
        <div class="row">
          <div>
            <label>物体名称</label>
            <input type="text" name="name" value="my_object" required/>
          </div>
          <div>
            <label>目标最大边长 (m)</label>
            <input type="number" name="target_extent_m" value="0.15" min="0.01" max="2" step="0.01"/>
          </div>
        </div>
        <div class="row">
          <div>
            <label>Poisson 深度</label>
            <input type="number" name="poisson_depth" value="9" min="6" max="11" step="1"/>
          </div>
          <div>
            <label>最少照片数</label>
            <input type="number" name="min_images" value="8" min="4" max="20" step="1"/>
          </div>
        </div>
        <p class="hint">环绕拍摄、约 60% 重叠；哑光非透明物体更稳。需本机已安装 COLMAP。</p>
        <button class="btn" type="submit">生成 CAD</button>
      </form>

      <form id="form-mesh" style="display:none">
        <label>已有 mesh（.obj / .ply / .glb / .stl）</label>
        <input type="file" name="mesh" accept=".obj,.ply,.glb,.stl,.off" required/>
        <div class="row">
          <div>
            <label>物体名称</label>
            <input type="text" name="name" value="imported_object" required/>
          </div>
          <div>
            <label>目标最大边长 (m)</label>
            <input type="number" name="target_extent_m" value="0.15" min="0.01" max="2" step="0.01"/>
          </div>
        </div>
        <p class="hint">适合 Meshroom / RealityCapture / 扫描仪导出后，再做尺度与居中后处理。</p>
        <button class="btn" type="submit">后处理并导出</button>
      </form>
    </section>

    <section class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
        <strong>结果</strong>
        <span id="status-pill" class="status-pill">idle</span>
      </div>
      <pre id="message">等待操作…</pre>
      <img id="preview" class="preview" alt="预览" style="display:none"/>
      <p id="download" class="hint"></p>
    </section>
  </div>

  <section class="card" style="margin-top:16px;">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <strong>已有 CAD</strong>
      <button class="btn secondary" style="width:auto;margin:0;padding:8px 12px;" type="button" id="btn-refresh">刷新</button>
    </div>
    <table>
      <thead><tr><th>名称</th><th>方法</th><th>边长</th><th>文件</th></tr></thead>
      <tbody id="mesh-table"></tbody>
    </table>
  </section>
</main>
<script>
const tabs = document.querySelectorAll('.tab');
const formPhotos = document.getElementById('form-photos');
const formMesh = document.getElementById('form-mesh');
tabs.forEach(t => t.addEventListener('click', () => {
  tabs.forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  const tab = t.dataset.tab;
  formPhotos.style.display = tab === 'photos' ? '' : 'none';
  formMesh.style.display = tab === 'mesh' ? '' : 'none';
}));

function setBusy(busy) {
  formPhotos.querySelector('button').disabled = busy;
  formMesh.querySelector('button').disabled = busy;
}

async function pollJob() {
  const r = await fetch('/api/job');
  const j = await r.json();
  const pill = document.getElementById('status-pill');
  pill.textContent = j.status + (j.progress ? ' · ' + j.progress : '');
  pill.className = 'status-pill ' + j.status;
  document.getElementById('message').textContent = j.message || '';
  const img = document.getElementById('preview');
  const dl = document.getElementById('download');
  if (j.preview) {
    img.style.display = '';
    img.src = '/file?path=' + encodeURIComponent(j.preview) + '&t=' + Date.now();
  }
  if (j.obj) {
    dl.innerHTML = '下载：<a href="/file?path=' + encodeURIComponent(j.obj) + '" download>reconstructed.obj</a>';
  }
  if (j.status === 'running') {
    setTimeout(pollJob, 1200);
  } else {
    setBusy(false);
    loadMeshes();
  }
}

async function submitForm(form, url) {
  setBusy(true);
  document.getElementById('message').textContent = '任务已提交…';
  const fd = new FormData(form);
  const r = await fetch(url, { method: 'POST', body: fd });
  const j = await r.json();
  if (!j.ok) {
    document.getElementById('message').textContent = j.error || '提交失败';
    setBusy(false);
    return;
  }
  pollJob();
}

formPhotos.addEventListener('submit', (e) => { e.preventDefault(); submitForm(formPhotos, '/api/photos'); });
formMesh.addEventListener('submit', (e) => { e.preventDefault(); submitForm(formMesh, '/api/mesh'); });

async function loadMeshes() {
  const r = await fetch('/api/meshes');
  const rows = await r.json();
  const tb = document.getElementById('mesh-table');
  tb.innerHTML = rows.map(x => `<tr>
    <td>${x.name}</td><td>${x.method}</td><td>${x.extent}</td>
    <td><a href="/file?path=${encodeURIComponent(x.obj)}" download>obj</a></td>
  </tr>`).join('') || '<tr><td colspan="4">暂无</td></tr>';
}
document.getElementById('btn-refresh').addEventListener('click', loadMeshes);
loadMeshes();
</script>
</body>
</html>
"""


class CadHandler(BaseHTTPRequestHandler):
    out_dir: Path = DEFAULT_OUT
    upload_root: Path = SCRIPT_DIR / ".cad_ui_uploads"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, raw, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/job":
            self._json(200, _get_job())
            return
        if parsed.path == "/api/meshes":
            self._json(200, list_existing_meshes(self.out_dir))
            return
        if parsed.path == "/file":
            qs = urllib.parse.parse_qs(parsed.query)
            path = Path(qs.get("path", [""])[0]).resolve()
            allowed_roots = [self.out_dir.resolve(), self.upload_root.resolve()]
            if not any(str(path).startswith(str(root)) for root in allowed_roots):
                self._json(403, {"ok": False, "error": "forbidden"})
                return
            if not path.is_file():
                self._json(404, {"ok": False, "error": "not found"})
                return
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            data = path.read_bytes()
            self._send(200, data, ctype)
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in ("/api/photos", "/api/mesh"):
            self._json(404, {"ok": False, "error": "not found"})
            return

        job = _get_job()
        if job.get("status") == "running":
            self._json(409, {"ok": False, "error": "已有任务在运行，请稍候"})
            return

        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._json(400, {"ok": False, "error": "需要 multipart/form-data"})
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": ctype,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
            keep_blank_values=True,
        )

        name = _safe_name(str(form.getvalue("name", "object")))
        try:
            target_extent_m = float(form.getvalue("target_extent_m", 0.15))
        except Exception:
            target_extent_m = 0.15

        self.upload_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch = self.upload_root / f"{stamp}_{name}"
        batch.mkdir(parents=True, exist_ok=True)

        if parsed.path == "/api/photos":
            try:
                poisson_depth = int(form.getvalue("poisson_depth", 9))
                min_images = int(form.getvalue("min_images", 8))
            except Exception:
                poisson_depth, min_images = 9, 8

            item = form["images"] if "images" in form else None
            files = item if isinstance(item, list) else ([item] if item is not None else [])
            saved: List[Path] = []
            for i, f in enumerate(files):
                if not getattr(f, "file", None) or not getattr(f, "filename", None):
                    continue
                suffix = Path(f.filename).suffix.lower()
                if suffix not in IMAGE_SUFFIXES:
                    continue
                dst = batch / f"img_{i:04d}{suffix}"
                with open(dst, "wb") as out:
                    shutil.copyfileobj(f.file, out)
                saved.append(dst)

            if not saved:
                self._json(400, {"ok": False, "error": "未收到有效图片"})
                return

            _set_job(status="running", message="任务已启动…", progress="queued", obj=None, preview=None)
            t = threading.Thread(
                target=_run_photos_job,
                args=(saved, name, target_extent_m, poisson_depth, min_images, self.out_dir),
                daemon=True,
            )
            t.start()
            self._json(200, {"ok": True})
            return

        # /api/mesh
        item = form["mesh"] if "mesh" in form else None
        if item is None or not getattr(item, "filename", None):
            self._json(400, {"ok": False, "error": "未收到 mesh 文件"})
            return
        suffix = Path(item.filename).suffix.lower()
        if suffix not in MESH_SUFFIXES:
            self._json(400, {"ok": False, "error": f"不支持后缀 {suffix}"})
            return
        dst = batch / f"import{suffix}"
        with open(dst, "wb") as out:
            shutil.copyfileobj(item.file, out)

        _set_job(status="running", message="任务已启动…", progress="queued", obj=None, preview=None)
        t = threading.Thread(
            target=_run_mesh_job,
            args=(dst, name, target_extent_m, self.out_dir),
            daemon=True,
        )
        t.start()
        self._json(200, {"ok": True})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAD / Mesh 生成界面")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    CadHandler.out_dir = out_dir
    CadHandler.upload_root = (SCRIPT_DIR / ".cad_ui_uploads").resolve()
    CadHandler.upload_root.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), CadHandler)
    print(f">>> CAD UI: http://127.0.0.1:{args.port}")
    print(f">>> 输出目录: {out_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n>>> 已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
