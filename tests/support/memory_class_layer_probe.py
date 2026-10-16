# ruff: noqa: E402
"""Exercise native layer construction without production data or settings."""

import sys
import tempfile
import time
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(sys.argv[1]).resolve()
SCENARIO = sys.argv[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/support")]
try:
    from qgis.core import QgsApplication, QgsTask, QgsVectorDataProvider, QgsVectorLayer
    from qgis.PyQt import sip
except ModuleNotFoundError:
    raise SystemExit(77)

from refinement_fixtures import layer

from labeling_tool.refinement import class_workspace


def check_layer(source):
    result, count = class_workspace._memory_class_layer(source, 12, "fusion:probe")
    assert count == result.featureCount() == source.featureCount()
    assert not result.isEditable()
    assert {field.name() for field in result.fields()} == {
        *[field.name() for field in source.fields()],
        *[name for name, _ in class_workspace.WORK_FIELDS],
    }
    expected = {
        feature["object_id"]: bytes(feature.geometry().asWkb())
        for feature in source.getFeatures()
    }
    for feature in result.getFeatures():
        assert bytes(feature.geometry().asWkb()) == expected.pop(feature["object_id"])
        assert feature["class_code"] == 12
        assert feature["baseline_stream_id"] == "fusion:probe"
        assert feature["geometry_source"] == "fusion"
        assert feature["geometry_revision"] == 0
        assert feature["reviewed"] == 0
        assert feature["updated_at"]
    assert not expected


def invalid_provider(root):
    source = layer(root, "source", positions=(0, 10))
    original = QgsVectorDataProvider.addAttributes

    def add_then_invalidate(provider, fields):
        result = original(provider, fields)
        # Reproduce the observed binding state, without deleting the C++
        # provider or its owning layer. Holding this wrapper must be unsafe.
        sip.setdeleted(provider)
        return result

    with patch.object(QgsVectorDataProvider, "addAttributes", add_then_invalidate):
        check_layer(source)


def background_batches(app, root):
    small = layer(root, "small", positions=(0, 10))
    large = layer(root, "large", positions=range(4101))
    uris = (small.source(), large.source())

    class Build(QgsTask):
        def __init__(self):
            super().__init__("class-layer regression", QgsTask.Flag.CanCancel)
            self.done = False
            self.error = ""

        def run(self):
            try:
                source = QgsVectorLayer(uris[0], "small", "ogr")
                for _ in range(120):
                    if self.isCanceled():
                        return False
                    check_layer(source)
                    time.sleep(0.001)
                # Exercise both the 4096-feature batch and trailing features.
                check_layer(QgsVectorLayer(uris[1], "large", "ogr"))
                empty, count = class_workspace._memory_class_layer(
                    source, 21, "fusion:probe"
                )
                assert count == empty.featureCount() == 0
                self.done = True
                return True
            except Exception:
                self.error = traceback.format_exc()
                return False

    task = Build()
    QgsApplication.taskManager().addTask(task)
    deadline = time.monotonic() + 25
    while not task.done and not task.error and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.001)
    if not task.done:
        task.cancel()
    assert task.done, task.error or "class-layer task timed out"


with (
    tempfile.TemporaryDirectory(prefix="class-memory-profile-") as profile,
    tempfile.TemporaryDirectory(prefix="class-memory-data-") as directory,
):
    app = QgsApplication([], False, profile)
    app.initQgis()
    try:
        if SCENARIO == "invalid_provider":
            invalid_provider(Path(directory))
        else:
            background_batches(app, Path(directory))
    finally:
        # Wait for this process's tasks before releasing its temporary files.
        app.exitQgis()
print(SCENARIO + ": passed", flush=True)
