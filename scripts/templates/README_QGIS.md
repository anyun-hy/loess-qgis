# QGIS 项目说明

项目初始化与 QGIS 插件安装相互独立。本目录不自动创建空 QGIS 工程，
也不固定用户的工程文件位置。

在 QGIS 中安装插件后，选择本项目的：

- `inference_scripts/` 作为推理脚本目录；
- `output/` 作为输出工作区；
- `output/accepted_labels.gpkg` 作为长期确认库路径。

这些路径仍可在插件界面中修改。`accepted_labels.gpkg` 会在具备实际
raster CRS 的正式写入阶段创建，初始化时不会生成。
