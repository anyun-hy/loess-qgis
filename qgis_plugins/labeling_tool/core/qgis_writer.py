"""Normalize QgsVectorFileWriter return values in QGIS 4.2."""

from qgis.core import QgsProject, QgsVectorFileWriter


def write_vector_layer(layer, path, options, *, transform_context=None):
    result = QgsVectorFileWriter.writeAsVectorFormatV3(
        layer, str(path),
        transform_context if transform_context is not None else QgsProject.instance().transformContext(),
        options,
    )
    if isinstance(result, (tuple, list)):
        error = result[0]
        message = str(result[1]) if len(result) > 1 else ""
    else:
        error = result
        message = ""
    return error, message
