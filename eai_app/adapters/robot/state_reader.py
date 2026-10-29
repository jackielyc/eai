from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
import ctypes
import html
import json
import math
import os
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from eai_app.perception.geometry import valid_depth_mask, canonical_color_topic
from eai_app.infrastructure.module_ref import module_ref

def read_q16(max_age_s: float = 2.0) -> Tuple[Optional[List[float]], float]:
    path = os.environ.get(
        "EAI_HOST_FK_STATE",
        os.path.join(tempfile.gettempdir(), "eai_host_fk_state.json"),
    )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        q = data.get("q16")
        stamp = float(data.get("t") or 0.0)
        age = time.time() - stamp
        if isinstance(q, list) and len(q) >= 16 and age <= max_age_s:
            return [float(x) for x in q[:16]], age
        if isinstance(q, list) and len(q) >= 16:
            return [float(x) for x in q[:16]], age
    except Exception:
        return None, -1.0
    return None, -1.0

