import os

from qgis.core import (
    Qgis,
    QgsProject, QgsRasterLayer, QgsVectorLayer, QgsLayerTreeGroup,
    QgsVectorFileWriter, QgsField, QgsFeature,
    QgsSimpleFillSymbolLayer, QgsSymbol, QgsFillSymbol,
    QgsSingleBandPseudoColorRenderer, QgsColorRampShader, QgsRasterShader,
    QgsContrastEnhancement,
)
from qgis.PyQt.QtCore import QVariant, QObject
from qgis.PyQt.QtGui import QColor

from .style_manager import StyleManager
from .layer_names import LAYER_NAMES
from .accepted_writer import ACCEPTED_FIELDS_QGS as ACCEPTED_FIELDS
from .qgis_writer import write_vector_layer


ANNOTATION_LAYER_PREFIXES = (
    "semantic_mask_preview",
    "confidence_mosaic",
    "semantic_polygons",
    "sam_refined_polygons",
    "accepted_labels",
)
MANAGED_PROPERTY = "labeling_tool/managed"

class LayerManager(QObject):
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.project = QgsProject.instance()

    def _resolve_crs(self, layer=None):
        if layer and layer.crs().isValid():
            return layer.crs().authid()
        if self.project.crs().isValid():
            return self.project.crs().authid()
        return "EPSG:4490"

    # ---- Raster layers ----

    def _run_group(self, run_id, section):
        root = self.project.layerTreeRoot()
        run_name = f"{run_id} 标注结果"
        group = root.findGroup(run_name)
        if group is None:
            group = root.addGroup(run_name)
        subgroup = group.findGroup(section)
        if subgroup is None:
            subgroup = group.addGroup(section)
        return subgroup

    def _add_managed_layer(self, layer, run_id, section, stream_id=""):
        layer.setCustomProperty(MANAGED_PROPERTY, True)
        layer.setCustomProperty("labeling_tool/run_id", run_id)
        layer.setCustomProperty("labeling_tool/stream_id", stream_id)
        self.project.addMapLayer(layer, False)
        self._run_group(run_id, section).addLayer(layer)
        return layer

    def load_result_stream(self, run_id, stream):
        """Load mask, confidence and review polygons for one result stream."""
        kind = stream.get("kind")
        if kind not in ("model", "fusion"):
            raise ValueError(f"Unsupported result stream kind: {kind}")
        identifier = stream.get("model_id") if kind == "model" else stream.get("fusion_profile_id")
        category = "Model" if kind == "model" else "Fusion"
        section = "Models" if kind == "model" else "Fusion"
        prefix = f"{run_id} | {category} | {identifier}"
        paths = stream.get("paths") or {}
        stream_id = stream.get("stream_id", "")
        loaded = {}

        mask = QgsRasterLayer(paths.get("mask_mosaic", ""), f"{prefix} | Mask")
        if not mask.isValid():
            raise RuntimeError(f"Failed to load mask for {stream_id}: {paths.get('mask_mosaic')}")
        StyleManager.apply_semantic_raster_style(mask)
        self._add_managed_layer(mask, run_id, section, stream_id)
        mask.triggerRepaint()
        loaded["mask"] = mask.id()

        confidence = QgsRasterLayer(paths.get("confidence_mosaic", ""), f"{prefix} | Confidence")
        if not confidence.isValid():
            raise RuntimeError(f"Failed to load confidence for {stream_id}: {paths.get('confidence_mosaic')}")
        StyleManager.apply_confidence_style(confidence)
        self._add_managed_layer(confidence, run_id, section, stream_id)
        confidence.triggerRepaint()
        loaded["confidence"] = confidence.id()

        polygon_path = stream.get("review_polygons") or paths.get("semantic_polygons", "")
        layer_name = stream.get("review_layer_name") or LAYER_NAMES.SEMANTIC
        polygons = QgsVectorLayer(
            f"{polygon_path}|layername={layer_name}", f"{prefix} | Polygons", "ogr"
        )
        if not polygons.isValid():
            raise RuntimeError(f"Failed to load polygons for {stream_id}: {polygon_path}")
        StyleManager.apply_categorized_style(polygons)
        self._add_managed_layer(polygons, run_id, section, stream_id)
        polygons.triggerRepaint()
        loaded["polygons"] = polygons.id()

        raw_path = paths.get("semantic_polygons_raw", "")
        if raw_path:
            raw_polygons = QgsVectorLayer(
                f"{raw_path}|layername={LAYER_NAMES.SEMANTIC_RAW}",
                f"{prefix} | Polygons Raw",
                "ogr",
            )
            if not raw_polygons.isValid():
                raise RuntimeError(
                    f"Failed to load raw polygons for {stream_id}: {raw_path}"
                )
            StyleManager.apply_categorized_style(raw_polygons)
            raw_polygons.setReadOnly(True)
            raw_polygons.setCustomProperty("labeling_tool/result_role", "polygons_raw")
            self._add_managed_layer(raw_polygons, run_id, section, stream_id)
            raw_node = self.project.layerTreeRoot().findLayer(raw_polygons.id())
            if raw_node is not None:
                raw_node.setItemVisibilityChecked(False)
            raw_polygons.triggerRepaint()
            loaded["polygons_raw"] = raw_polygons.id()
        return loaded

    def load_run_results(self, result):
        loaded = {}
        run_id = str(result.get("run_id") or "")
        for stream in result.get("ready_streams") or []:
            loaded[stream["stream_id"]] = self.load_result_stream(run_id, stream)
        return loaded

    def load_class_layer(self, run_id, class_code, source_label, gpkg_path, layer_name):
        display_name = f"{run_id} | Class | {int(class_code)} | {source_label}"
        layer = QgsVectorLayer(f"{gpkg_path}|layername={layer_name}", display_name, "ogr")
        if not layer.isValid():
            raise RuntimeError(f"Failed to load class layer: {gpkg_path}")
        StyleManager.apply_categorized_style(layer)
        layer.setCustomProperty("labeling_tool/class_code", int(class_code))
        layer.setCustomProperty("labeling_tool/source_label", source_label)
        self._add_managed_layer(layer, run_id, "Classes", source_label)
        layer.triggerRepaint()
        return layer.id()

    def load_workspace_class(self, run_id, record):
        class_code = int(record["class_code"])
        for layer in self.project.mapLayers().values():
            if (
                bool(layer.customProperty(MANAGED_PROPERTY, False))
                and str(layer.customProperty("labeling_tool/run_id", "")) == str(run_id)
                and int(layer.customProperty("labeling_tool/class_code", -1)) == class_code
                and str(layer.customProperty("labeling_tool/workspace_path", ""))
                == str(record["path"])
            ):
                return layer.id()
        display_name = (
            f"{run_id} | Class | {class_code} "
            f"{record.get('class_name', '')} | Working"
        )
        layer = QgsVectorLayer(
            f"{record['path']}|layername={record['layer_name']}",
            display_name,
            "ogr",
        )
        if not layer.isValid():
            raise RuntimeError(f"Failed to load class workspace layer: {record['path']}")
        StyleManager.apply_categorized_style(layer)
        layer.setCustomProperty("labeling_tool/class_code", class_code)
        layer.setCustomProperty("labeling_tool/workspace_path", str(record["path"]))
        self._add_managed_layer(layer, run_id, "Classes", f"class:{class_code}")
        layer.triggerRepaint()
        return layer.id()

    def load_workspace_classes(self, run_id, workspace):
        return {
            int(code): self.load_workspace_class(run_id, record)
            for code, record in (workspace.get("classes") or {}).items()
        }

    def load_final_composite(self, run_id, gpkg_path):
        layer = QgsVectorLayer(
            f"{gpkg_path}|layername={LAYER_NAMES.FINAL_COMPOSITE}",
            f"{run_id} | Final | Composite", "ogr",
        )
        if not layer.isValid():
            raise RuntimeError(f"Failed to load final_composite: {gpkg_path}")
        StyleManager.apply_categorized_style(layer)
        self._add_managed_layer(layer, run_id, "Final", "final")
        layer.triggerRepaint()
        return layer.id()

    def load_topology_issues(self, run_id, gpkg_path):
        layer = QgsVectorLayer(
            f"{gpkg_path}|layername={LAYER_NAMES.TOPOLOGY_ISSUES}",
            f"{run_id} | Final | Topology Issues", "ogr",
        )
        if not layer.isValid():
            raise RuntimeError(f"Failed to load topology_issues: {gpkg_path}")
        StyleManager.apply_outline_style(layer, color="#d7191c", width=0.8)
        self._add_managed_layer(layer, run_id, "Final", "topology")
        layer.triggerRepaint()
        return layer.id()

    def load_semantic_mask_preview(self, raster_path):
        layer = QgsRasterLayer(raster_path, "semantic_mask_preview")
        if not layer.isValid():
            raise RuntimeError(f"Failed to load semantic mask raster: {raster_path}")
        self.project.addMapLayer(layer)

        StyleManager.apply_semantic_raster_style(layer)
        layer.triggerRepaint()
        return layer.id()

    def load_confidence_mosaic(self, raster_path):
        """Load confidence raster with RdYlGn pseudocolor ramp."""
        layer = QgsRasterLayer(raster_path, "confidence_mosaic")
        if not layer.isValid():
            raise RuntimeError(f"Failed to load confidence raster: {raster_path}")
        self.project.addMapLayer(layer)

        shader = QgsColorRampShader()
        shader.setColorRampType(Qgis.ShaderInterpolationMethod.Linear)
        items = [
            QgsColorRampShader.ColorRampItem(0.0, QColor("#FF0000"), "0.00"),
            QgsColorRampShader.ColorRampItem(0.25, QColor("#FF6666"), "0.25"),
            QgsColorRampShader.ColorRampItem(0.5, QColor("#FFFF00"), "0.50"),
            QgsColorRampShader.ColorRampItem(0.75, QColor("#66FF66"), "0.75"),
            QgsColorRampShader.ColorRampItem(1.0, QColor("#00FF00"), "1.00"),
        ]
        shader.setColorRampItemList(items)

        raster_shader = QgsRasterShader()
        raster_shader.setRasterShaderFunction(shader)

        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, raster_shader)
        renderer.setClassificationMin(0.0)
        renderer.setClassificationMax(1.0)

        layer.setRenderer(renderer)
        layer.triggerRepaint()
        return layer.id()

    # ---- Vector layers ----

    def load_semantic_polygons(self, gpkg_path):
        """Load semantic_polygons layer from GPKG with 14-class style."""
        uri = f"{gpkg_path}|layername={LAYER_NAMES.SEMANTIC}"
        layer = QgsVectorLayer(uri, LAYER_NAMES.SEMANTIC, "ogr")
        if not layer.isValid():
            raise RuntimeError(f"Failed to load semantic_polygons from {gpkg_path}")
        self.project.addMapLayer(layer)

        StyleManager.apply_categorized_style(layer)
        layer.triggerRepaint()
        return layer.id()

    def load_sam_refined_polygons(self, gpkg_path):
        """Load SAM3 refined polygons with the shared 14-class style."""
        uri = f"{gpkg_path}|layername={LAYER_NAMES.SAM_REFINED}"
        layer = QgsVectorLayer(uri, LAYER_NAMES.SAM_REFINED, "ogr")
        if not layer.isValid():
            raise RuntimeError(f"Failed to load sam_refined_polygons from {gpkg_path}")
        self.project.addMapLayer(layer)

        StyleManager.apply_categorized_style(layer)
        layer.triggerRepaint()
        return layer.id()

    def get_or_create_accepted_labels(self, gpkg_path):
        """Load or create the accepted_labels layer (editable final labels)."""
        # Try loading existing GPKG layer
        uri = f"{gpkg_path}|layername={LAYER_NAMES.ACCEPTED}"
        layer = QgsVectorLayer(uri, LAYER_NAMES.ACCEPTED, "ogr")
        if layer.isValid():
            self.project.addMapLayer(layer)
            StyleManager.apply_accepted_style(layer)
            layer.triggerRepaint()
            return layer.id()

        # Create in-memory layer
        project_crs = self._resolve_crs()
        mem_layer = QgsVectorLayer(
            f"Polygon?crs={project_crs}", LAYER_NAMES.ACCEPTED, "memory"
        )
        mem_layer.dataProvider().addAttributes(ACCEPTED_FIELDS)
        mem_layer.updateFields()

        self.project.addMapLayer(mem_layer)
        mem_layer_id = mem_layer.id()
        StyleManager.apply_accepted_style(layer=mem_layer)

        # Save to GPKG
        save_options = QgsVectorFileWriter.SaveVectorOptions()
        save_options.driverName = "GPKG"
        save_options.layerName = LAYER_NAMES.ACCEPTED
        save_options.actionOnExistingFile = (
            QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
        )
        if not os.path.exists(gpkg_path):
            save_options.actionOnExistingFile = (
                QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
            )

        error, err_msg = write_vector_layer(mem_layer, gpkg_path, save_options)
        if error != QgsVectorFileWriter.WriterError.NoError:
            raise RuntimeError(f"Failed to save accepted_labels to GPKG: {err_msg}")

        # Remove memory layer, load GPKG layer instead
        self.project.removeMapLayer(mem_layer_id)
        gpkg_layer = QgsVectorLayer(uri, LAYER_NAMES.ACCEPTED, "ogr")
        if gpkg_layer.isValid():
            self.project.addMapLayer(gpkg_layer)
            StyleManager.apply_accepted_style(gpkg_layer)
            gpkg_layer.triggerRepaint()
            return gpkg_layer.id()
        return mem_layer_id  # fallback if GPKG load fails

    # ---- Bulk operations ----

    def remove_annotation_layers(self):
        """Remove all annotation layers managed by this tool."""
        to_remove = []
        for layer in self.project.mapLayers().values():
            if bool(layer.customProperty(MANAGED_PROPERTY, False)):
                to_remove.append(layer.id())
                continue
            for prefix in ANNOTATION_LAYER_PREFIXES:
                if layer.name().startswith(prefix):
                    to_remove.append(layer.id())
                    break
        for lid in to_remove:
            self.project.removeMapLayer(lid)

    def set_layer_visibility(self, layer_id, visible):
        """Show or hide a layer by ID."""
        layer = self.project.mapLayer(layer_id)
        if layer:
            tree_layer = self.project.layerTreeRoot().findLayer(layer_id)
            if tree_layer:
                tree_layer.setItemVisibilityChecked(visible)

    def group_layers(self, group_name="标注结果"):
        """Move all annotation layers into a named group."""
        root = self.project.layerTreeRoot()
        candidates = []
        for layer in self.project.mapLayers().values():
            for prefix in ANNOTATION_LAYER_PREFIXES:
                if layer.name().startswith(prefix):
                    tree_layer = root.findLayer(layer.id())
                    if tree_layer is not None:
                        candidates.append(tree_layer)
                    break

        group = root.findGroup(group_name)
        if not candidates:
            if group is not None and not group.children():
                parent = group.parent()
                if parent is not None:
                    parent.removeChildNode(group)
            return None

        if group is None:
            group = root.addGroup(group_name)
        for tree_layer in candidates:
            if tree_layer.parent() != group:
                clone = tree_layer.clone()
                group.addChildNode(clone)
                tree_layer.parent().removeChildNode(tree_layer)
        return group
