def classFactory(iface):
    from labeling_tool.plugin import LabelingTool
    return LabelingTool(iface)