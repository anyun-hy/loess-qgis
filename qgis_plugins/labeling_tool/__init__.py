def classFactory(iface):
    from .plugin import LabelingTool
    return LabelingTool(iface)