# 交互架构可视化

本目录保存当前项目的交互式架构图及其 Archify JSON 源规格。HTML 是可直接在
浏览器中打开的自包含交付物，支持明暗主题、缩放、搜索、路径追踪和导出。

## 当前图表

| 图表 | 用途 | 源规格 |
|---|---|---|
| [loess-qgis-runtime-architecture.html](loess-qgis-runtime-architecture.html) | 项目高层运行时、进程、数据流和双平台边界 | [JSON](loess-qgis-runtime-architecture.json) |
| [loess-qgis-database-connection-before-after.html](loess-qgis-database-connection-before-after.html) | PostgreSQL-only 改造前后的 Run 数据库连接对比，并区分 GeoPackage 数据层 | [JSON](loess-qgis-database-connection-before-after.json) |

## 维护合同

- 架构事实以当前代码和 [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) 为准；图表不得
  覆盖或替代书面合同；
- 修改图表时编辑对应 JSON，再使用 Archify `validate` 和 `deliver` 重新生成
  HTML，不直接手改生成后的 HTML；
- 正式交付必须通过 showcase 9 项校验，并检查 1440×900、1600×1000、
  1920×1080 和 2048×1320 桌面尺寸；
- `*.visual-check.*` 截图、联系表和收据属于本地验收证据，不进入项目仓库；
- 临时研究图、一次性截图和无来源导出不得放入本目录。
