# ruff: noqa: E402
"""Native QGIS reviewed/reopened class commits on temporary GeoPackages."""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication
except ModuleNotFoundError:
    raise SystemExit(77)

from refinement_fixtures import layer

from labeling_tool.refinement.class_review import commit_class_review
from labeling_tool.refinement.edit_tracking import EditTracker


def confirm_reopen(_app, root):
    source = layer(root, "class-review", positions=(0, 10))
    tracker = EditTracker()
    timestamps = iter(("confirmed-first", "confirmed-second"))

    commit_class_review(
        source,
        reviewed=True,
        now=lambda: next(timestamps),
        tracker=tracker,
        keep_editing=False,
    )
    assert not source.isEditable() and not tracker.suppressed
    saved = list(source.getFeatures())
    assert [feature["reviewed"] for feature in saved] == [1, 1]
    assert [feature["updated_at"] for feature in saved] == [
        "confirmed-first",
        "confirmed-second",
    ]

    commit_class_review(
        source,
        reviewed=False,
        now=lambda: "reopened",
        tracker=tracker,
        keep_editing=False,
    )
    assert not source.isEditable() and not tracker.suppressed
    assert all(feature["reviewed"] == 0 for feature in source.getFeatures())
    assert all(feature["updated_at"] == "reopened" for feature in source.getFeatures())


app = QgsApplication([], False)
app.initQgis()
try:
    with tempfile.TemporaryDirectory(prefix="loess-class-review-") as directory:
        scenario = sys.argv[2]
        globals()[scenario](app, Path(directory))
        print(scenario + ": passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    app.exitQgis()
