"""Normalize QgsVectorFileWriter return values across supported QGIS 3.x."""

from qgis.core import QgsProject, QgsVectorFileWriter


def write_vector_layer(layer, path, options):
    result = QgsVectorFileWriter.writeAsVectorFormatV3(
        layer, str(path), QgsProject.instance().transformContext(), options
    )
    if isinstance(result, (tuple, list)):
        error = result[0]
        message = str(result[1]) if len(result) > 1 else ""
    else:
        error = result
        message = ""
    return error, message
