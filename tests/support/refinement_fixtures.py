"""Native temporary GeoPackage fixtures shared by refinement behavior probes."""

from qgis.core import (
    QgsFeature,
    QgsGeometry,
    QgsRectangle,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

from labeling_tool.qgis_support.qgis_writer import write_vector_layer
from labeling_tool.refinement.final_assembler import FINAL_FIELDS
from labeling_tool.shared.contracts.run_spec import CLASS_NAMES

SPEC = {"run_id": "manual-probe"}


def rectangle(x, width=4):
    geometry = QgsGeometry.fromRect(QgsRectangle(x, 0, x + width, 4))
    geometry.convertToMultiType()
    return geometry


def layer(root, name, code=12, positions=(), crs="EPSG:3857"):
    memory = QgsVectorLayer(f"MultiPolygon?crs={crs}", name, "memory")
    assert memory.isValid()
    memory.dataProvider().addAttributes(FINAL_FIELDS)
    memory.updateFields()
    features = []
    for index, position in enumerate(positions):
        feature = QgsFeature(memory.fields())
        feature.setGeometry(rectangle(position))
        for key, value in dict(
            run_id=SPEC["run_id"],
            object_id=f"{name}-{index}",
            part_id="007",
            class_code=code,
            class_name=CLASS_NAMES[code],
            geometry_source="fusion",
            geometry_revision=3,
            baseline_stream_id="fusion:probe",
            confidence_mean=0.8,
            confidence_std=0.2,
            reviewed=1,
        ).items():
            feature.setAttribute(key, value)
        features.append(feature)
    assert memory.dataProvider().addFeatures(features)[0]
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName, options.layerName = "GPKG", "features"
    path = root / f"{name}.gpkg"
    assert write_vector_layer(memory, path, options)[0] == 0
    result = QgsVectorLayer(f"{path}|layername=features", name, "ogr")
    assert result.isValid()
    return result


def records(value):
    return {feature["object_id"]: feature for feature in value.getFeatures()}


def signature(value):
    return {
        identity: (bytes(feature.geometry().asWkb()), tuple(feature.attributes()))
        for identity, feature in records(value).items()
    }
