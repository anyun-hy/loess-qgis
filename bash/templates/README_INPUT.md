# QGIS 输入数据

- `rasters/`：推荐存放用于推理的原始栅格，但插件仍允许选择其他位置。
- `ranges/`：推荐存放范围矢量及其完整 sidecar 文件；Shapefile 必须保持
  `.shp/.shx/.dbf/.prj` 等文件同组。

初始化脚本不会复制、扫描或修改输入数据。实际 raster 和 range layer
由使用者在 QGIS 中加载并选择。
