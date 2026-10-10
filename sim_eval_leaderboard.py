"""Load / live-refresh industry sim-eval leaderboards for the EAI 仿真评测 tab."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

_JSON_LOCK = threading.Lock()

from sim_backends_registry import (
    ALL_BACKENDS,
    embodiedbench_url,
    get_backend,
    list_backend_keys,
    normalize_backend_key,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PATH = os.path.join(_HERE, "sim_eval_leaderboards.json")

BACKEND_KEYS = tuple(list_backend_keys())

_UA = "EAI-sim-eval-leaderboard/1.0 (+https://github.com/local/eai)"
_HTTP_TIMEOUT = 25

_ROBODOJO_CSV = (
    "https://galaxygeneralrobotics.github.io/astra-policy/"
    "joint-data/robodojo-scores.csv"
)
_MOLMO_INDEX = "https://molmospaces.allen.ai/benchmark/data/index.json"
_MOLMO_DATA = "https://molmospaces.allen.ai/benchmark/data"
_OPENPI_LIBERO_README = (
    "https://raw.githubusercontent.com/Physical-Intelligence/openpi/"
    "main/examples/libero/README.md"
)
_VLA_HARNESS_LB = "https://allenai.github.io/vla-evaluation-harness/leaderboard/"
_GENESIS_URL = "https://genesis-embodied-ai.github.io/"


@lru_cache(maxsize=4)
def load_leaderboards(path: str = "") -> Dict[str, Any]:
    """Load JSON board data. Cached; call clear_leaderboard_cache() to reload."""
    use = (path or "").strip() or _DEFAULT_PATH
    with open(use, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("leaderboard root must be an object")
    return data


def clear_leaderboard_cache() -> None:
    load_leaderboards.cache_clear()


def default_json_path() -> str:
    return _DEFAULT_PATH


def board_for_backend(
    backend: str, *, path: str = ""
) -> Optional[Dict[str, Any]]:
    """Return one backend board dict, or None if missing."""
    key = _normalize_backend(backend)
    data = load_leaderboards(path)
    boards = data.get("backends") or {}
    board = boards.get(key)
    if not isinstance(board, dict):
        # Synthesize a stub from registry so GUI never shows empty for known keys.
        spec = get_backend(key)
        if spec is None:
            return None
        return {
            "title": f"{spec.label} · 业内评测榜单",
            "metric": spec.embodied_metric or "Success Rate",
            "protocol": "本地尚无快照；点「刷新」从公开源抓取。",
            "source_url": spec.source_url
            or (embodiedbench_url(spec.embodied_slug) if spec.embodied_slug else ""),
            "source_label": spec.source_label or "官方 / EmbodiedBench",
            "entries": [],
            "_backend": key,
            "_updated": data.get("updated") or "",
            "_global_note": data.get("note") or "",
        }
    out = dict(board)
    out["_backend"] = key
    out["_updated"] = data.get("updated") or ""
    out["_global_note"] = data.get("note") or ""
    return out


def entries_for_backend(backend: str, *, path: str = "") -> List[Dict[str, Any]]:
    board = board_for_backend(backend, path=path)
    if not board:
        return []
    rows = board.get("entries") or []
    return [r for r in rows if isinstance(r, dict)]


def _normalize_backend(backend: str) -> str:
    return normalize_backend_key(backend)


def _http_get(url: str, *, timeout: float = _HTTP_TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_get_text(url: str, *, timeout: float = _HTTP_TIMEOUT) -> str:
    data = _http_get(url, timeout=timeout)
    if data.lstrip()[:15].lower().startswith(b"<!doctype") or data.lstrip()[
        :6
    ].lower().startswith(b"<html"):
        raise RuntimeError(f"expected data, got HTML: {url}")
    return data.decode("utf-8", "replace")


def _now_stamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _decode_rsc_string(raw: str) -> str:
    """Decode a Next.js RSC string payload to UTF-8 text."""
    try:
        return json.loads(f'"{raw}"')
    except Exception:
        try:
            return (
                raw.encode("utf-8")
                .decode("unicode_escape")
                .encode("latin-1")
                .decode("utf-8")
            )
        except Exception:
            return raw


def _extract_json_array_after(text: str, marker: str) -> List[Any]:
    j = text.find(marker)
    if j < 0:
        raise RuntimeError(f"marker not found: {marker}")
    start = text.find("[", j)
    if start < 0:
        raise RuntimeError(f"array start not found after {marker}")
    depth = 0
    end = None
    for idx, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        raise RuntimeError(f"array end not found after {marker}")
    return json.loads(text[start:end])


def fetch_embodiedbench_results(
    url: str,
    *,
    setting: str = "Average",
    metric: str = "Success Rate",
    limit: int = 12,
) -> List[Dict[str, Any]]:
    """Parse EmbodiedBench page RSC payload into ranked entries."""
    raw = _http_get(url).decode("utf-8", "replace")
    payloads: List[str] = []
    for m in re.finditer(
        r'self\.__next_f\.push\(\[.*?,"((?:\\.|[^"\\])*)"\]\)', raw, re.S
    ):
        payloads.append(_decode_rsc_string(m.group(1)))
    joined = "".join(payloads) if payloads else raw
    arr = _extract_json_array_after(joined, '"results":[')
    rows = [
        x
        for x in arr
        if isinstance(x, dict)
        and x.get("metric") == metric
        and (not setting or x.get("setting") == setting)
    ]
    if not rows and setting:
        rows = [x for x in arr if isinstance(x, dict) and x.get("metric") == metric]
    if not rows:
        # last resort: any scored rows
        rows = [
            x
            for x in arr
            if isinstance(x, dict) and x.get("score") is not None and x.get("model")
        ]
    rows = sorted(
        rows, key=lambda x: (-float(x.get("score") or 0), int(x.get("rank") or 999))
    )
    seen = set()
    out: List[Dict[str, Any]] = []
    for x in rows:
        model = str(x.get("model") or "").strip()
        if not model or model in seen:
            continue
        seen.add(model)
        score = x.get("scoreDisplay")
        if score is None or score == "":
            sc = x.get("score")
            if sc is None:
                score = ""
            else:
                # CALVIN-style averages are small floats; SR is percent.
                try:
                    fsc = float(sc)
                    score = f"{fsc}%" if fsc > 10 or metric == "Success Rate" else f"{fsc}"
                except (TypeError, ValueError):
                    score = str(sc)
        detail_bits = []
        if x.get("setting"):
            detail_bits.append(str(x["setting"]))
        if x.get("organization") and str(x["organization"]) not in (
            "",
            "Unspecified",
            "—",
        ):
            detail_bits.append(str(x["organization"]))
        out.append(
            {
                "rank": int(x.get("rank") or len(out) + 1),
                "method": model,
                "score": str(score),
                "detail": " · ".join(detail_bits) or metric,
                "org": str(x.get("organization") or "")
                if str(x.get("organization") or "") not in ("Unspecified", "—")
                else "",
                "date": str(x.get("year") or ""),
                "note": "EmbodiedBench 实时抓取",
            }
        )
        if len(out) >= limit:
            break
    if not out:
        raise RuntimeError(f"no ranked rows from {url}")
    for i, row in enumerate(out, 1):
        row["rank"] = i
    return out


def fetch_embodied_for_backend(
    backend: str, *, limit: int = 12
) -> Dict[str, Any]:
    """Generic EmbodiedBench fetcher driven by sim_backends_registry."""
    spec = get_backend(backend)
    if spec is None or not spec.embodied_slug:
        raise KeyError(f"no embodied_slug for backend: {backend}")
    url = embodiedbench_url(spec.embodied_slug)
    metric = spec.embodied_metric or "Success Rate"
    setting = spec.embodied_setting or "Average"
    try:
        entries = fetch_embodiedbench_results(
            url, setting=setting, metric=metric, limit=limit
        )
    except Exception:
        # Retry without setting / with Success Rate
        entries = fetch_embodiedbench_results(
            url, setting="", metric=metric, limit=limit
        )
    return {
        "title": f"{spec.label} · 实时 Leaderboard",
        "metric": metric,
        "protocol": f"EmbodiedBench 实时抓取 · setting={setting or 'any'}",
        "source_url": spec.source_url or url,
        "source_label": spec.source_label or "EmbodiedBench",
        "entries": entries,
        "_live": True,
        "_fetched_at": _now_stamp(),
    }


def fetch_robotwin_live(*, limit: int = 12) -> Dict[str, Any]:
    board = fetch_embodied_for_backend("robotwin", limit=limit)
    board["title"] = "RoboTwin 2.0 · 实时 Leaderboard"
    board["metric"] = "Average SR% (clean2clean / clean2random)"
    board["protocol"] = "EmbodiedBench 实时抓取 · 默认 Average=(c2c+c2r)/2"
    board["source_url"] = "https://robotwin-platform.github.io/leaderboard"
    board["source_label"] = "RoboTwin / EmbodiedBench"
    return board


def fetch_libero_live(*, limit: int = 12) -> Dict[str, Any]:
    board = fetch_embodied_for_backend("libero", limit=limit)
    entries = list(board.get("entries") or [])
    try:
        md = _http_get_text(_OPENPI_LIBERO_README)
        m = re.search(
            r"\|\s*π0\.5[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)",
            md,
        )
        if m:
            spat, obj, goal, long_, avg = m.groups()
            method = "π₀.₅ @ 30k (openpi 官方)"
            if not any(method.split()[0] in str(e.get("method")) for e in entries):
                entries.append(
                    {
                        "rank": len(entries) + 1,
                        "method": method,
                        "score": f"{spat}/{obj}/{goal}/{long_} · {avg}",
                        "detail": "Spatial/Object/Goal/Long · Avg",
                        "org": "Physical Intelligence",
                        "date": "openpi README",
                        "note": "官方公开 ckpt 表",
                    }
                )
    except Exception:
        pass
    for i, row in enumerate(entries[:limit], 1):
        row["rank"] = i
    board["entries"] = entries[:limit]
    board["title"] = "LIBERO · 实时仿真榜单"
    board["metric"] = "Average Success Rate %"
    board["protocol"] = "EmbodiedBench Average；并附 openpi README 官方 π0.5 行"
    return board


def fetch_mujoco_live(*, limit: int = 12) -> Dict[str, Any]:
    board = fetch_embodied_for_backend("metaworld", limit=limit)
    board["title"] = "MuJoCo · Meta-World 实时榜（参考）"
    board["protocol"] = (
        "本页是 MJCF viewer；实时抓取 EmbodiedBench Meta-World 作为 MuJoCo 系参考榜"
    )
    return board


def fetch_isaac_live(*, limit: int = 12) -> Dict[str, Any]:
    text = _http_get_text(_ROBODOJO_CSV)
    rows = list(csv.DictReader(io.StringIO(text)))
    agg: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"sr": [], "score": [], "origin": set()}
    )
    for r in rows:
        model = (r.get("model") or "").strip()
        if not model:
            continue
        try:
            sr = float(r["success_rate_percent"])
            sc = float(r["score_100"])
        except (KeyError, TypeError, ValueError):
            continue
        agg[model]["sr"].append(sr)
        agg[model]["score"].append(sc)
        agg[model]["origin"].add((r.get("origin") or "").strip() or "unknown")

    ranked: List[Tuple[float, float, str, str]] = []
    for model, v in agg.items():
        if not v["sr"]:
            continue
        mean_sc = sum(v["score"]) / len(v["score"])
        mean_sr = sum(v["sr"]) / len(v["sr"])
        origin = ",".join(sorted(v["origin"]))
        ranked.append((mean_sc, mean_sr, model, origin))
    ranked.sort(key=lambda t: (-t[0], -t[1], t[2]))

    entries: List[Dict[str, Any]] = []
    for i, (sc, sr, model, origin) in enumerate(ranked[:limit], 1):
        note = "官方条目" if origin == "official" else f"来源 {origin}"
        entries.append(
            {
                "rank": i,
                "method": model,
                "score": f"{sc:.2f} / {sr:.2f}%",
                "detail": "Average Score / SR%（按任务微平均）",
                "org": "",
                "date": _now_stamp().split(" ")[0],
                "note": note,
            }
        )
    if not entries:
        raise RuntimeError("RoboDojo CSV empty")
    return {
        "title": "Isaac / RoboDojo · 实时仿真榜单",
        "metric": "Average Score / SR%",
        "protocol": "Galaxy Astra 公开 robodojo-scores.csv 实时聚合（含 official / local）",
        "source_url": (
            "https://galaxygeneralrobotics.github.io/astra-policy/"
            "index.html?view=gripper-manipulation"
        ),
        "source_label": "Astra×RoboDojo CSV",
        "entries": entries,
        "_live": True,
        "_fetched_at": _now_stamp(),
    }


def _molmo_overall_oracle(csv_text: str) -> Optional[float]:
    lines = [
        ln
        for ln in csv_text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not lines:
        return None
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    for row in reader:
        cat = (row.get("category") or "").strip().upper()
        if cat == "OVERALL":
            try:
                return float(row.get("oracle_rate_pct") or "")
            except ValueError:
                return None
    return None


def fetch_molmospaces_live(*, limit: int = 12) -> Dict[str, Any]:
    """Aggregate MS Combined (pick/pnp/open/close) oracle rates from public CSVs."""
    index = json.loads(_http_get_text(_MOLMO_INDEX))
    tasks = index.get("tasks") or []
    ms_ids = ["ms_pick", "ms_pick_place", "ms_open", "ms_close"]
    task_map = {t.get("id"): t for t in tasks if isinstance(t, dict)}

    rates: Dict[str, List[float]] = defaultdict(list)
    meta_by_file: Dict[str, Dict[str, str]] = {}

    for tid in ms_ids:
        task = task_map.get(tid) or {}
        policies = list(task.get("policies") or [])
        try:
            meta_txt = _http_get_text(f"{_MOLMO_DATA}/{tid}/metadata.csv")
            for row in csv.DictReader(io.StringIO(meta_txt)):
                fid = (row.get("file") or "").strip()
                if fid:
                    meta_by_file[fid] = row
                    if fid not in policies:
                        policies.append(fid)
        except Exception:
            pass

        def _one(fid: str) -> Optional[Tuple[str, float]]:
            try:
                csv_txt = _http_get_text(f"{_MOLMO_DATA}/{tid}/{fid}.csv")
            except Exception:
                return None
            rate = _molmo_overall_oracle(csv_txt)
            if rate is None:
                return None
            return fid, rate

        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(_one, fid) for fid in policies]
            for fut in as_completed(futs):
                item = fut.result()
                if item is None:
                    continue
                fid, rate = item
                rates[fid].append(rate)

    ranked: List[Tuple[float, int, str]] = []
    for fid, vals in rates.items():
        if len(vals) < 3:
            continue
        ranked.append((sum(vals) / len(vals), len(vals), fid))
    ranked.sort(key=lambda t: (-t[0], -t[1], t[2]))

    entries: List[Dict[str, Any]] = []
    for i, (avg, n_tasks, fid) in enumerate(ranked[:limit], 1):
        meta = meta_by_file.get(fid) or {}
        method = _strip_html(meta.get("Policy") or fid)
        org = (meta.get("Affiliation") or "").strip()
        released = (meta.get("Date Released") or "").strip()
        entries.append(
            {
                "rank": i,
                "method": method,
                "score": f"{avg:.1f}%",
                "detail": f"MS Combined oracle · {n_tasks}/4 tasks",
                "org": org,
                "date": released or _now_stamp().split(" ")[0],
                "note": "MolmoSpaces CSV 实时聚合（不含 MolmoSpaces 训练数据筛选）",
            }
        )
    if not entries:
        raise RuntimeError("MolmoSpaces CSV aggregation empty")
    return {
        "title": "MolmoSpaces · 实时 Leaderboard",
        "metric": "Oracle Success Rate % (MS Combined)",
        "protocol": "公开 /benchmark/data CSV 聚合 MS-Pick / PnP / Open / Close OVERALL",
        "source_url": "https://molmospaces.allen.ai/leaderboard",
        "source_label": "molmospaces.allen.ai",
        "entries": entries,
        "_live": True,
        "_fetched_at": _now_stamp(),
    }


def fetch_arena_live(*, limit: int = 8) -> Dict[str, Any]:
    """Arena has no single public board; refresh linked community boards as pointers."""
    pointers: List[Dict[str, Any]] = []
    try:
        lib = fetch_libero_live(limit=3)
        for e in lib.get("entries") or []:
            row = dict(e)
            row["detail"] = "LIBERO via Arena 社区适配 · " + str(row.get("detail") or "")
            row["note"] = "Arena 无总榜；显示关联 LIBERO 实时结果"
            pointers.append(row)
    except Exception as exc:
        pointers.append(
            {
                "rank": 1,
                "method": "(LIBERO 实时抓取失败)",
                "score": "—",
                "detail": str(exc)[:120],
                "org": "",
                "date": _now_stamp().split(" ")[0],
                "note": "",
            }
        )
    try:
        rt = fetch_robotwin_live(limit=3)
        for e in rt.get("entries") or []:
            row = dict(e)
            row["detail"] = "RoboTwin via Arena · " + str(row.get("detail") or "")
            row["note"] = "Arena 无总榜；显示关联 RoboTwin 实时结果"
            pointers.append(row)
    except Exception:
        pass
    for i, row in enumerate(pointers[:limit], 1):
        row["rank"] = i
    return {
        "title": "IsaacLab-Arena · 关联实时榜（参考）",
        "metric": "Success Rate % (linked benchmarks)",
        "protocol": "Arena 本身无冻结总榜；刷新时拉取可挂载的 LIBERO / RoboTwin 实时结果",
        "source_url": "https://isaac-sim.github.io/IsaacLab-Arena/main/index.html",
        "source_label": "Isaac Lab-Arena docs",
        "entries": pointers[:limit],
        "_live": True,
        "_fetched_at": _now_stamp(),
    }


def fetch_vla_harness_live(*, limit: int = 12) -> Dict[str, Any]:
    """Pointer board for AllenAI VLA Evaluation Harness unified leaderboard."""
    # Page is a SPA; expose curated cross-bench highlights + link.
    highlights = [
        ("LIBERO", "libero"),
        ("SimplerEnv", "simpler"),
        ("CALVIN", "calvin"),
        ("RoboCasa", "robocasa"),
        ("RoboTwin", "robotwin"),
        ("ManiSkill2", "maniskill2"),
    ]
    entries: List[Dict[str, Any]] = []
    for label, key in highlights:
        try:
            sub = fetch_embodied_for_backend(key, limit=1)
            top = (sub.get("entries") or [{}])[0]
            entries.append(
                {
                    "rank": len(entries) + 1,
                    "method": f"{label} · {top.get('method') or '—'}",
                    "score": str(top.get("score") or "—"),
                    "detail": f"各榜 #1 · 见 AllenAI VLA Harness 总榜",
                    "org": str(top.get("org") or ""),
                    "date": str(top.get("date") or ""),
                    "note": "跨基准入口；完整表打开来源",
                }
            )
        except Exception as exc:
            entries.append(
                {
                    "rank": len(entries) + 1,
                    "method": f"{label} · (抓取失败)",
                    "score": "—",
                    "detail": str(exc)[:100],
                    "org": "",
                    "date": _now_stamp().split(" ")[0],
                    "note": "",
                }
            )
        if len(entries) >= limit:
            break
    return {
        "title": "VLA Evaluation Harness · 跨基准总榜入口",
        "metric": "Per-benchmark #1 (aggregated)",
        "protocol": (
            "AllenAI vla-evaluation-harness：Docker 化统一评测；"
            "此处展示各主榜当前 #1 作为入口，完整矩阵见官方 Leaderboard"
        ),
        "source_url": _VLA_HARNESS_LB,
        "source_label": "AllenAI VLA Leaderboard",
        "entries": entries,
        "_live": True,
        "_fetched_at": _now_stamp(),
    }


def fetch_genesis_live(*, limit: int = 8) -> Dict[str, Any]:
    """Genesis is a sim engine; surface related ManiSkill / Meta-World pointers."""
    pointers: List[Dict[str, Any]] = []
    for key, tag in (("maniskill3", "ManiSkill3"), ("metaworld", "Meta-World")):
        try:
            sub = fetch_embodied_for_backend(key, limit=2)
            for e in sub.get("entries") or []:
                row = dict(e)
                row["detail"] = f"{tag}（Genesis 可复现同类任务） · " + str(
                    row.get("detail") or ""
                )
                row["note"] = "Genesis 无单一冻结总榜；显示关联基准"
                pointers.append(row)
        except Exception:
            pass
    if not pointers:
        pointers = [
            {
                "rank": 1,
                "method": "Genesis (engine)",
                "score": "—",
                "detail": "多物理求解器 · 见官方文档 / 示例任务",
                "org": "Genesis-Embodied-AI",
                "date": "",
                "note": "非策略榜",
            }
        ]
    for i, row in enumerate(pointers[:limit], 1):
        row["rank"] = i
    return {
        "title": "Genesis · 仿真后端（关联榜）",
        "metric": "N/A (engine) · linked SR%",
        "protocol": "Genesis 是仿真引擎；刷新时附带 ManiSkill3 / Meta-World 参考结果",
        "source_url": _GENESIS_URL,
        "source_label": "Genesis docs",
        "entries": pointers[:limit],
        "_live": True,
        "_fetched_at": _now_stamp(),
    }


def _make_embodied_fetcher(key: str) -> Callable[..., Dict[str, Any]]:
    def _fn(*, limit: int = 12) -> Dict[str, Any]:
        return fetch_embodied_for_backend(key, limit=limit)

    _fn.__name__ = f"fetch_{key}_live"
    return _fn


_FETCHERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "isaac": fetch_isaac_live,
    "mujoco": fetch_mujoco_live,
    "molmospaces": fetch_molmospaces_live,
    "arena": fetch_arena_live,
    "libero": fetch_libero_live,
    "robotwin": fetch_robotwin_live,
    "vla_harness": fetch_vla_harness_live,
    "genesis": fetch_genesis_live,
}

# Auto-register EmbodiedBench-backed extended backends.
for _spec in ALL_BACKENDS:
    if _spec.key in _FETCHERS:
        continue
    if _spec.embodied_slug:
        _FETCHERS[_spec.key] = _make_embodied_fetcher(_spec.key)


def refresh_backend(
    backend: str,
    *,
    path: str = "",
    persist: bool = True,
    limit: int = 12,
) -> Dict[str, Any]:
    """Fetch live board for one backend, optionally merge into JSON on disk."""
    key = _normalize_backend(backend)
    fetcher = _FETCHERS.get(key)
    if fetcher is None:
        # Fall back to embodied if registry has slug
        if get_backend(key) and get_backend(key).embodied_slug:
            fetcher = _make_embodied_fetcher(key)
        else:
            raise KeyError(f"unsupported backend: {key}")
    board = fetcher(limit=limit)
    board["_backend"] = key
    if persist:
        _persist_board(key, board, path=path)
        clear_leaderboard_cache()
        data = load_leaderboards(path)
        board["_updated"] = data.get("updated") or board.get("_fetched_at") or ""
        board["_global_note"] = data.get("note") or ""
    else:
        board["_updated"] = board.get("_fetched_at") or _now_stamp()
        board["_global_note"] = "实时抓取（未写入本地 JSON）"
    return board


def refresh_all(
    *, path: str = "", persist: bool = True, limit: int = 12
) -> Dict[str, Any]:
    """Refresh every backend. Returns {backend: board_or_error}."""
    out: Dict[str, Any] = {}
    for key in BACKEND_KEYS:
        try:
            out[key] = refresh_backend(key, path=path, persist=persist, limit=limit)
        except Exception as exc:
            out[key] = {"_backend": key, "_error": str(exc)}
    return out


def _persist_board(backend: str, board: Dict[str, Any], *, path: str = "") -> None:
    use = (path or "").strip() or _DEFAULT_PATH
    with _JSON_LOCK:
        try:
            with open(use, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"note": "", "backends": {}}
        if not isinstance(data, dict):
            data = {"backends": {}}
        backends = data.setdefault("backends", {})
        if not isinstance(backends, dict):
            backends = {}
            data["backends"] = backends
        stored = {
            "title": board.get("title"),
            "metric": board.get("metric"),
            "protocol": board.get("protocol"),
            "source_url": board.get("source_url"),
            "source_label": board.get("source_label"),
            "entries": board.get("entries") or [],
            "last_live_fetch": board.get("_fetched_at") or _now_stamp(),
        }
        backends[backend] = stored
        data["updated"] = board.get("_fetched_at") or _now_stamp()
        note = data.get("note") or ""
        if "实时" not in note:
            data["note"] = (
                (note + " ").strip()
                + " 可用「刷新」从公开源实时更新并写回本文件。"
            ).strip()
        tmp = use + f".tmp.{os.getpid()}.{threading.get_ident()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, use)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Refresh sim-eval leaderboards")
    p.add_argument("--backend", default="", help="one key, or empty = all")
    p.add_argument("--no-persist", action="store_true")
    p.add_argument("--limit", type=int, default=12)
    args = p.parse_args()
    if args.backend:
        b = refresh_backend(
            args.backend, persist=not args.no_persist, limit=args.limit
        )
        print(json.dumps({"ok": True, "backend": args.backend, "n": len(b.get("entries") or [])}, ensure_ascii=False))
    else:
        r = refresh_all(persist=not args.no_persist, limit=args.limit)
        summary = {
            k: ("err: " + v["_error"] if isinstance(v, dict) and "_error" in v else f"n={len((v or {}).get('entries') or [])}")
            for k, v in r.items()
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
