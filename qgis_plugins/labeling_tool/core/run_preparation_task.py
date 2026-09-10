"""Freeze detached input sources and perform spatial preflight off the GUI."""

from pathlib import Path

from qgis.core import QgsTask, QgsVectorLayer, QgsVectorFileWriter

from . import accepted_integrity, difference_filter, tile_manager
from .class_workspace import _sha256_file
from .layer_names import LAYER_NAMES


def write_source_snapshot(source, wkb_type, path, layer_name, transform_context, is_canceled):
    """Stream a GUI-captured QgsVectorLayerFeatureSource, never a live layer."""
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = str(layer_name)
    writer = QgsVectorFileWriter.create(
        str(path), source.fields(), wkb_type, source.crs(),
        transform_context, options,
    )
    try:
        if writer.hasError() != QgsVectorFileWriter.WriterError.NoError:
            raise RuntimeError(writer.errorMessage())
        for feature in source.getFeatures():
            if is_canceled():
                raise RuntimeError("输入快照已取消")
            if not writer.addFeature(feature):
                raise RuntimeError(writer.lastError())
        if not writer.flushBuffer():
            raise RuntimeError(writer.lastError())
    finally:
        del writer


class RunPreparationTask(QgsTask):
    def __init__(self, request, *, range_source, accepted_source, raster_crs, transform_context,
                 range_wkb_type=None, accepted_wkb_type=None):
        super().__init__("后台冻结并审计 Run 输入", QgsTask.Flag.CanCancel)
        self.request = dict(request)
        self.range_source = range_source
        self.accepted_source = accepted_source
        self.range_wkb_type = range_wkb_type
        self.accepted_wkb_type = accepted_wkb_type
        self.raster_crs = raster_crs
        self.transform_context = transform_context
        self.result_data = None
        self.error_message = ""

    def run(self):
        try:
            request = self.request
            root = Path(request["run_dir"])
            selected = [dict(tile) for tile in request["active_tiles"]]
            selection = dict(request["range_selection"])
            range_path = ""
            if self.range_source is not None:
                path = root / "range_snapshot.gpkg"
                write_source_snapshot(self.range_source, self.range_wkb_type, path, "range_mask",
                                      self.transform_context, self.isCanceled)
                layer = QgsVectorLayer(f"{path}|layername=range_mask", "frozen_range", "ogr")
                geometries = tile_manager.snapshot_vector_geometries(
                    layer, self.raster_crs, transform_context=self.transform_context,
                    is_canceled=self.isCanceled,
                )
                selected = tile_manager.select_tiles_intersecting_geometries(
                    [dict(tile) for tile in request["grid_tiles"]], geometries,
                    is_canceled=self.isCanceled,
                    progress=lambda current, total: self.setProgress(40 * current / max(1, total)),
                )
                if self.isCanceled():
                    return False
                if not selected:
                    raise ValueError("冻结的范围矢量没有选中任何完整 Tile")
                range_path = str(path)
                selection.update(
                    vector_source=range_path, vector_path=range_path,
                    vector_sha256=_sha256_file(path, self.isCanceled),
                    vector_crs=layer.crs().authid(), clip_outputs=True,
                    selected_tile_count=len(selected),
                    excluded_tile_count=len(request["grid_tiles"]) - len(selected),
                )
                del layer
            self.setProgress(40)
            validation = dict(request["accepted_validation"])
            accepted_path = ""
            skipped = []
            if self.accepted_source is not None:
                if request["skip_accepted"]:
                    path = root / "accepted_snapshot.gpkg"
                    write_source_snapshot(self.accepted_source, self.accepted_wkb_type,
                                          path, LAYER_NAMES.ACCEPTED,
                                          self.transform_context, self.isCanceled)
                    source_path = f"{path}|layername={LAYER_NAMES.ACCEPTED}"
                else:
                    # Auditing a non-skipped target must not copy a potentially
                    # huge file or charge the Run for an unused snapshot.
                    source_path = request["accepted_source_path"]
                layer = QgsVectorLayer(source_path, "accepted_audit", "ogr")
                validation = accepted_integrity.audit_accepted_layer(
                    layer, overlap_tolerance=validation["overlap_tolerance"],
                    expected_crs=self.raster_crs, is_canceled=self.isCanceled,
                )
                validation["source"] = "run_snapshot" if request["skip_accepted"] else "existing_target"
                if request["skip_accepted"]:
                    accepted_path = str(path)
                    for index, tile in enumerate(selected):
                        if self.isCanceled():
                            return False
                        if difference_filter.tile_is_fully_accepted(tile["bounds"], layer, self.raster_crs):
                            skipped.append({**tile, "skip_reason": "fully_accepted"})
                        if index % 256 == 0:
                            self.setProgress(60 + 40 * index / max(1, len(selected)))
                del layer
            if self.isCanceled():
                return False
            self.result_data = dict(
                active_tiles=selected, skipped_tiles=skipped,
                range_selection=selection, range_snapshot=range_path,
                accepted_snapshot=accepted_path, accepted_validation=validation,
                inputs_prepared=True,
            )
            self.setProgress(100)
            return True
        except Exception as error:
            self.error_message = f"{type(error).__name__}: {error}"
            return False
