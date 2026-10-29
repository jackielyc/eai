"""ui.datasets.notes — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_launch_utils = module_ref('eai_app.adapters.robot.launch_utils')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def _inline_md_to_html(text: str) -> str:
    """极简行内 Markdown：链接、粗体、行内代码。"""
    import re

    s = _eai_adapters_robot_launch_utils._html_escape_text(text)
    s = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r'<a href="\2" style="color:#7cb8ff;">\1</a>',
        s,
    )
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    # 裸 URL（未在 a 标签内）
    s = re.sub(
        r'(?<!["\'>])(https?://[^\s<]+)',
        r'<a href="\1" style="color:#7cb8ff;">\1</a>',
        s,
    )
    return s


def markdown_notes_to_html(md: str) -> str:
    """把分析笔记 Markdown 转成适合深色 UI 的简易 HTML（含表格）。"""
    import re

    lines = md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # UI 中不展示 Cursor Canvas 路径块
    skip_canvas = False
    filtered: _eai_platform_bootstrap.List[str] = []
    for line in lines:
        if "配套 Canvas" in line or "canvases/" in line:
            skip_canvas = True
            continue
        if skip_canvas:
            if line.startswith("```"):
                # 跳过整段 code fence
                if filtered and filtered[-1].strip() == "```":
                    filtered.pop()
                skip_canvas = False
                continue
            if line.startswith("---") or line.startswith("#"):
                skip_canvas = False
            else:
                continue
        filtered.append(line)

    body: _eai_platform_bootstrap.List[str] = []
    i = 0
    in_code = False
    code_buf: _eai_platform_bootstrap.List[str] = []

    def flush_code() -> None:
        nonlocal code_buf
        if not code_buf:
            return
        body.append(
            "<pre style='background:#111214;border:1px solid #333;padding:10px;"
            "border-radius:4px;overflow-x:auto;color:#c8c8c8;'>"
            + _eai_adapters_robot_launch_utils._html_escape_text("\n".join(code_buf))
            + "</pre>"
        )
        code_buf = []

    while i < len(filtered):
        line = filtered[i]
        if line.strip().startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # 表格：表头 + |---| 分隔行
        sep = filtered[i + 1] if i + 1 < len(filtered) else ""
        is_table = (
            "|" in line
            and "-" in sep
            and bool(re.match(r"^[\s|:\-]+$", sep.strip()))
        )
        if is_table:
            rows: _eai_platform_bootstrap.List[_eai_platform_bootstrap.List[str]] = []
            while i < len(filtered) and "|" in filtered[i]:
                raw = filtered[i].strip()
                if re.match(r"^[\s|:\-]+$", raw) and "-" in raw:
                    i += 1
                    continue
                cells = [c.strip() for c in raw.strip("|").split("|")]
                rows.append(cells)
                i += 1
            if rows:
                body.append(
                    "<table style='border-collapse:collapse;width:100%;"
                    "margin:8px 0 14px 0;font-size:12px;'>"
                )
                for ri, row in enumerate(rows):
                    body.append("<tr>")
                    tag = "th" if ri == 0 else "td"
                    bg = (
                        "#25262b"
                        if ri == 0
                        else ("#1e1f23" if ri % 2 == 0 else "#1a1b1e")
                    )
                    for cell in row:
                        body.append(
                            f"<{tag} style='border:1px solid #3a3b40;padding:6px 8px;"
                            f"background:{bg};text-align:left;vertical-align:top;'>"
                            f"{_inline_md_to_html(cell)}</{tag}>"
                        )
                    body.append("</tr>")
                body.append("</table>")
            continue

        if line.startswith("# "):
            body.append(
                f"<h1 style='color:#f0f0f0;font-size:18px;margin:12px 0 8px 0;'>"
                f"{_inline_md_to_html(line[2:].strip())}</h1>"
            )
        elif line.startswith("## "):
            body.append(
                f"<h2 style='color:#e8e8e8;font-size:15px;margin:16px 0 8px 0;"
                f"border-bottom:1px solid #333;padding-bottom:4px;'>"
                f"{_inline_md_to_html(line[3:].strip())}</h2>"
            )
        elif line.startswith("### "):
            body.append(
                f"<h3 style='color:#ddd;font-size:13px;margin:12px 0 6px 0;'>"
                f"{_inline_md_to_html(line[4:].strip())}</h3>"
            )
        elif line.strip() == "---":
            body.append("<hr style='border:none;border-top:1px solid #333;margin:14px 0;'/>")
        elif line.startswith("> "):
            body.append(
                "<div style='color:#aaa;border-left:3px solid #4a6a8a;padding:4px 10px;"
                f"margin:6px 0;'>{_inline_md_to_html(line[2:].strip())}</div>"
            )
        elif re.match(r"^[-*] ", line.strip()):
            body.append(
                f"<div style='margin:2px 0 2px 12px;'>• {_inline_md_to_html(line.strip()[2:])}</div>"
            )
        elif not line.strip():
            body.append("<div style='height:6px;'></div>")
        else:
            body.append(
                f"<p style='margin:4px 0;line-height:1.55;'>{_inline_md_to_html(line)}</p>"
            )
        i += 1

    if in_code:
        flush_code()

    return (
        "<html><body style='background:#1a1b1e;color:#ececec;"
        "font-family:Sans-Serif;font-size:13px;padding:8px 12px;'>"
        + "\n".join(body)
        + "</body></html>"
    )


_DATASET_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})


_DATASET_VIDEO_EXTS = frozenset({".mp4", ".avi", ".mkv", ".webm", ".mov"})


_DATASET_RRD_EXTS = frozenset({".rrd"})


_DATASET_JSON_EXTS = frozenset({".json"})


def _dataset_format_bytes(n: int) -> str:
    size = float(max(0, int(n)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{int(n)} B"


def resolve_rrd_path(path: str) -> str:
    """解析 RRD 路径，兼容容器内 /root/dataset 与 /home/psibot/dataset。"""
    if not path:
        return path
    expanded = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(path.strip()))
    if _eai_platform_bootstrap.os.path.isfile(expanded):
        return expanded
    candidates = [expanded]
    if expanded.startswith("/home/psibot/dataset"):
        candidates.append(expanded.replace("/home/psibot/dataset", "/root/dataset", 1))
    elif expanded.startswith("/root/dataset"):
        candidates.append(expanded.replace("/root/dataset", "/home/psibot/dataset", 1))
    for candidate in candidates:
        if _eai_platform_bootstrap.os.path.isfile(candidate):
            return candidate
    return expanded


_RERUN_ARGV_CACHE: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[str]] = None


_RERUN_ARGV_RESOLVED = False


def resolve_rerun_argv() -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[str]]:
    """解析用于打开 .rrd 的 Rerun 命令（可用环境变量 RERUN_BIN 覆盖）。"""
    global _RERUN_ARGV_CACHE, _RERUN_ARGV_RESOLVED
    if _RERUN_ARGV_RESOLVED and _eai_platform_bootstrap.os.environ.get("RERUN_BIN", "").strip() == "":
        return list(_RERUN_ARGV_CACHE) if _RERUN_ARGV_CACHE else None

    result: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[str]] = None
    env_bin = _eai_platform_bootstrap.os.environ.get("RERUN_BIN", "").strip()
    if env_bin:
        parts = _eai_platform_bootstrap.shlex.split(env_bin)
        if parts and (_eai_platform_bootstrap.os.path.isfile(parts[0]) or _eai_platform_bootstrap.shutil.which(parts[0])):
            result = parts
    if result is None:
        which = _eai_platform_bootstrap.shutil.which("rerun")
        if which:
            result = [which]
    if result is None:
        # rerun-sdk 常以 python -m rerun 提供入口
        for py in (_eai_platform_bootstrap.sys.executable, _eai_platform_bootstrap.shutil.which("python3"), _eai_platform_bootstrap.shutil.which("python")):
            if not py:
                continue
            try:
                probe = _eai_platform_bootstrap.subprocess.run(
                    [py, "-c", "import rerun"],
                    capture_output=True,
                    timeout=8,
                )
            except (OSError, _eai_platform_bootstrap.subprocess.TimeoutExpired):
                continue
            if probe.returncode == 0:
                result = [py, "-m", "rerun"]
                break

    if not env_bin:
        _RERUN_ARGV_CACHE = list(result) if result else None
        _RERUN_ARGV_RESOLVED = True
    return list(result) if result else None


def resolve_modern_chromium_browser() -> _eai_platform_bootstrap.Optional[str]:
    """Chrome 91+ / Edge 等现代 Chromium（Qt5 WebEngine=Chrome87，无法跑 Rerun WASM）。"""
    env = _eai_platform_bootstrap.os.environ.get("RERUN_BROWSER", "").strip()
    if env:
        which = _eai_platform_bootstrap.shutil.which(env) or (env if _eai_platform_bootstrap.os.path.isfile(env) else "")
        if which:
            return which
    for name in (
        "microsoft-edge",
        "microsoft-edge-stable",
        "google-chrome-stable",
        "google-chrome",
        "chromium-browser",
        "chromium",
    ):
        which = _eai_platform_bootstrap.shutil.which(name)
        if which:
            return which
    return None


def open_url_in_chromium_app(url: str) -> _eai_platform_bootstrap.Tuple[bool, str]:
    """用系统 Chromium --app 打开 URL。返回 (ok, detail)。"""
    browser = resolve_modern_chromium_browser()
    if not browser:
        return False, "未找到 Edge/Chrome/Chromium（可用 RERUN_BROWSER 指定）"
    ok = _eai_platform_bootstrap.QProcess.startDetached(browser, [f"--app={url}", "--new-window"])
    if not ok:
        quoted = " ".join(_eai_platform_bootstrap.shlex.quote(x) for x in [browser, f"--app={url}", "--new-window"])
        ok = _eai_platform_bootstrap.QProcess.startDetached("bash", ["-lc", quoted])
    if ok:
        return True, browser
    return False, f"启动失败: {browser}"


def _pick_free_tcp_port() -> int:
    with _eai_platform_bootstrap.socket.socket(_eai_platform_bootstrap.socket.AF_INET, _eai_platform_bootstrap.socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

