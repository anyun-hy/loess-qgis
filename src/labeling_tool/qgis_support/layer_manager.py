from pathlib import Path

from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QObject

from labeling_tool.qgis_support.layer_names import LAYER_NAMES
from labeling_tool.qgis_support.style_manager import StyleManager

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

    # ---- Raster layers ----

    @staticmethod
    def _directory(path):
        return str(Path(str(path).split("|", 1)[0]).expanduser().resolve()) if path else ""

    def _run_group(self, run_id, section, run_dir=""):
        root = self.project.layerTreeRoot()
        run_dir = self._directory(run_dir)
        group = next((node for node in root.findGroups()
                      if str(node.customProperty("labeling_tool/run_id", "")) == str(run_id)
                      and str(node.customProperty("labeling_tool/run_dir", "")) == run_dir), None)
        if group is None:
            legacy = root.findGroup(f"{run_id} 标注结果")
            if legacy is not None and not legacy.customProperty("labeling_tool/run_dir", ""):
                managed = [node.layer() for node in legacy.findLayers()
                           if node.layer() is not None and node.layer().customProperty(MANAGED_PROPERTY, False)]
                if managed and all(self._belongs_to_run(layer, run_id, run_dir) for layer in managed):
                    group = legacy
                    group.setCustomProperty("labeling_tool/run_id", str(run_id))
                    group.setCustomProperty("labeling_tool/run_dir", run_dir)
        if group is None:
            name = f"{run_id} 标注结果"
            if root.findGroup(name) is not None:
                name += f" ({run_dir})"
            group = root.addGroup(name)
            group.setCustomProperty("labeling_tool/run_id", str(run_id))
            group.setCustomProperty("labeling_tool/run_dir", run_dir)
        subgroup = group.findGroup(section)
        if subgroup is None:
            subgroup = group.addGroup(section)
        return subgroup

    def _add_managed_layer(self, layer, run_id, section, stream_id="", *, run_dir=""):
        layer.setCustomProperty(MANAGED_PROPERTY, True)
        layer.setCustomProperty("labeling_tool/run_id", run_id)
        layer.setCustomProperty("labeling_tool/run_dir", self._directory(run_dir))
        layer.setCustomProperty("labeling_tool/section", section)
        layer.setCustomProperty("labeling_tool/stream_id", stream_id)
        self.project.addMapLayer(layer, False)
        self._run_group(run_id, section, run_dir).addLayer(layer)
        return layer

    def _belongs_to_run(self, layer, run_id, run_dir):
        if not bool(layer.customProperty(MANAGED_PROPERTY, False)):
            return False
        if str(layer.customProperty("labeling_tool/run_id", "")) != str(run_id):
            return False
        recorded = str(layer.customProperty("labeling_tool/run_dir", ""))
        if recorded:
            return self._directory(recorded) == self._directory(run_dir)
        # Existing projects may predate the directory property. Use the data
        # source, never run_id alone, to recognize these managed layers.
        return bool(run_dir and Path(self._directory(layer.source())).is_relative_to(Path(run_dir).resolve()))

    def load_result_stream(self, run_id, stream, *, run_dir=""):
        """Validate a complete stream before adding layers; reuse exact sources."""
        kind = stream.get("kind")
        if kind not in ("model", "fusion"):
            raise ValueError(f"Unsupported result stream kind: {kind}")
        paths = stream.get("paths") or {}
        if not run_dir:
            run_dir = str(Path(paths.get("semantic_polygons", "")).resolve().parents[2])
        identifier = stream.get("model_id") if kind == "model" else stream.get("fusion_profile_id")
        category = "Model" if kind == "model" else "Fusion"
        section = "Models" if kind == "model" else "Fusion"
        prefix = f"{run_id} | {category} | {identifier}"
        stream_id = stream.get("stream_id", "")
        polygon_path = stream.get("review_polygons") or paths.get("semantic_polygons", "")
        layer_name = stream.get("review_layer_name") or LAYER_NAMES.SEMANTIC
        entries = [
            ("mask", paths.get("mask_mosaic", ""), "Mask", None),
            ("confidence", paths.get("confidence_mosaic", ""), "Confidence", None),
            ("polygons", f"{polygon_path}|layername={layer_name}", "Polygons", "ogr"),
        ]
        if paths.get("semantic_polygons_raw"):
            entries.append(("polygons_raw", f"{paths['semantic_polygons_raw']}|layername={LAYER_NAMES.SEMANTIC_RAW}", "Polygons Raw", "ogr"))
        loaded, pending = {}, []
        for role, source, title, provider in entries:
            existing = next((layer for layer in self.project.mapLayers().values()
                             if self._belongs_to_run(layer, run_id, run_dir)
                             and str(layer.customProperty("labeling_tool/stream_id", "")) == stream_id
                             and layer.source() == source), None)
            if existing is not None:
                existing.setCustomProperty("labeling_tool/run_dir", self._directory(run_dir))
                existing.setCustomProperty("labeling_tool/section", section)
                existing.setCustomProperty("labeling_tool/result_role", role)
                target = self._run_group(run_id, section, run_dir)
                node = self.project.layerTreeRoot().findLayer(existing.id())
                if node is not None and node.parent() is not target:
                    target.addChildNode(node.clone())
                    node.parent().removeChildNode(node)
                loaded[role] = existing.id()
                continue
            layer = (QgsVectorLayer(source, f"{prefix} | {title}", provider) if provider
                     else QgsRasterLayer(source, f"{prefix} | {title}"))
            if not layer.isValid():
                raise RuntimeError(f"无法加载 {stream_id} / {role}: {source}")
            if role == "mask":
                StyleManager.apply_semantic_raster_style(layer)
            elif role == "confidence":
                StyleManager.apply_confidence_style(layer)
            else:
                StyleManager.apply_categorized_style(layer)
                layer.setReadOnly(True)
            layer.setCustomProperty("labeling_tool/result_role", role)
            pending.append((role, layer))
        for role, layer in pending:
            self._add_managed_layer(layer, run_id, section, stream_id, run_dir=run_dir)
            if role == "polygons_raw":
                self.set_layer_visibility(layer.id(), False)
            loaded[role] = layer.id()
        return loaded

    def load_run_results(self, result):
        before = set(self.project.mapLayers())
        try:
            return {
                stream["stream_id"]: self.load_result_stream(
                    str(result.get("run_id") or ""), stream, run_dir=result.get("run_dir", "")
                )
                for stream in result.get("ready_streams") or []
                if stream.get("kind") in {"model", "fusion"}
            }
        except Exception:
            self.project.removeMapLayers(list(set(self.project.mapLayers()) - before))
            raise

    def apply_load_scope(self, result, scope, *, loaded=None):
        if scope not in {"classes", "all"}:
            raise ValueError(f"未知加载范围: {scope}")
        previous = [layer for layer in self.project.mapLayers().values()
                    if self._belongs_to_run(layer, result["run_id"], result["run_dir"])
                    and str(layer.customProperty("labeling_tool/stream_id", "")).startswith(("model:", "fusion:"))]
        if any(isinstance(layer, QgsVectorLayer) and layer.isEditable() and layer.isModified() for layer in previous):
            raise RuntimeError("结果图层存在未保存编辑，请先保存或回滚后切换加载范围")
        if loaded is None:
            loaded = self.load_run_results(result) if scope == "all" else {}
        retained = {layer_id for layers in loaded.values() for layer_id in layers.values()}
        self.project.removeMapLayers([layer.id() for layer in previous if layer.id() not in retained])
        for group in reversed(self.project.layerTreeRoot().findGroups()):
            if group.name() in {"Models", "Fusion"} and not group.children():
                parent = group.parent()
                if parent and str(parent.customProperty("labeling_tool/run_dir", "")) == self._directory(result["run_dir"]):
                    parent.removeChildNode(group)
        return loaded

    def load_workspace_class(self, run_id, record, *, visible=False):
        class_code = int(record["class_code"])
        for layer in self.project.mapLayers().values():
            if (
                bool(layer.customProperty(MANAGED_PROPERTY, False))
                and str(layer.customProperty("labeling_tool/run_id", "")) == str(run_id)
                and int(layer.customProperty("labeling_tool/class_code", -1)) == class_code
                and str(layer.customProperty("labeling_tool/workspace_path", ""))
                == str(record["path"])
            ):
                tree_layer = self.project.layerTreeRoot().findLayer(layer.id())
                if tree_layer is not None:
                    tree_layer.setItemVisibilityChecked(bool(visible))
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
        self._add_managed_layer(layer, run_id, "Classes", f"class:{class_code}",
                                run_dir=str(Path(record["path"]).resolve().parents[1]))
        tree_layer = self.project.layerTreeRoot().findLayer(layer.id())
        if tree_layer is not None:
            tree_layer.setItemVisibilityChecked(bool(visible))
        return layer.id()

    def load_workspace_classes(self, run_id, workspace):
        return {
            int(code): self.load_workspace_class(run_id, record, visible=False)
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
        self._add_managed_layer(layer, run_id, "Final", "final", run_dir=str(Path(gpkg_path).resolve().parents[1]))
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
        self._add_managed_layer(layer, run_id, "Final", "topology", run_dir=str(Path(gpkg_path).resolve().parents[1]))
        layer.triggerRepaint()
        return layer.id()

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
