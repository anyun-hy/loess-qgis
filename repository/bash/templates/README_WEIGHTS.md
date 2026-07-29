# 模型权重目录

本目录由项目使用者放置部署资产，初始化脚本不会下载、复制或创建假的权重文件。

所需文件及语义模型 SHA256 记录在项目根目录 `project_manifest.json` 的
`required_assets` 中。放置完成后执行：

```bash
<repository>/bash/init_project.sh --project-root <project_root> --check-only --check-assets
```

缺少文件或已登记 SHA256 不一致时，QGIS 环境检查和正式 Run 必须停止。
