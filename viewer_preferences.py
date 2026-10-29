"""Atomic UI preferences shared by the lazy robot page and main window."""
import json
import os
from pathlib import Path
import tempfile


DEFAULT_PATH = Path(__file__).with_name('viewer_ui_state.json')


def load_state(path=DEFAULT_PATH):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_state(data, path=DEFAULT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return str(path)


def save_robot_preferences(selections, path=DEFAULT_PATH):
    state = load_state(path)
    state['robot_view'] = {'version': 1, 'selections': dict(selections)}
    return write_state(state, path)


def save_main_preferences(data, path=DEFAULT_PATH):
    state = dict(data)
    robot = load_state(path).get('robot_view')
    if isinstance(robot, dict):
        state['robot_view'] = robot
    return write_state(state, path)
