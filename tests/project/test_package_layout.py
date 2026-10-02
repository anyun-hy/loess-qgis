"""Keep the worker's shared imports independent of the QGIS host and models."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def _isolated_imports(tmp_path, code):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_shared_imports_do_not_load_the_gui_or_models(tmp_path):
    _isolated_imports(tmp_path, """
import importlib
import sys

for module in (
    'labeling_tool.shared.contracts.run_spec',
    'labeling_tool.shared.contracts.monitor_contract',
    'labeling_tool.shared.state.postgres_state',
    'labeling_tool.shared.state.run_state_db',
    'labeling_tool.shared.planning.ownership_neighbors',
    'labeling_tool.shared.planning.work_package_planner',
):
    importlib.import_module(module)

for prefix in ('qgis', 'PyQt6', 'torch', 'labeling_tool.runs', 'labeling_tool.main'):
    loaded = [name for name in sys.modules if name == prefix or name.startswith(prefix + '.')]
    assert not loaded, loaded
""")


def test_runner_facade_does_not_load_qt_until_requested(tmp_path):
    _isolated_imports(tmp_path, """
import sys
import labeling_tool.runs

assert 'V5AsyncInferenceRunner' in labeling_tool.runs.__all__
assert 'labeling_tool.runs.v5_async_runner' not in sys.modules
assert not any(name == 'qgis' or name.startswith('qgis.') for name in sys.modules)
""")
