#!/usr/bin/env bash
# UI/ROS and Pinocchio/MuJoCo intentionally use their existing separate runtimes.
set -eo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
case "${1:-all}" in
    all|ui|planning) suite="${1:-all}" ;;
    *) echo 'usage: scripts/run_offline_tests.sh [all|ui|planning]' >&2; exit 2 ;;
esac
if [[ "$suite" != planning ]]; then
    if [[ -f /opt/ros/humble/setup.bash ]]; then source /opt/ros/humble/setup.bash; fi
    export QT_QPA_PLATFORM=offscreen
    "${EAI_UI_PYTHON:-/usr/bin/python3}" - <<'PY'
from pathlib import Path
import unittest
planner = {'test_host_grasp_plan', 'test_bounded_arm_ik', 'test_grasp_scene', 'test_phone_assembly'}
modules = ['tests.' + p.stem for p in sorted(Path('tests').glob('test_*.py')) if p.stem not in planner]
suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
fi
if [[ "$suite" != ui ]]; then
    env -u PYTHONPATH -u PYTHONHOME "${EAI_PLANNER_PYTHON:-$root/.cache/grasp_ik_env/bin/python}" -m unittest -v \
        tests.test_host_grasp_plan tests.test_bounded_arm_ik tests.test_grasp_scene \
        tests.test_phone_assembly
fi
