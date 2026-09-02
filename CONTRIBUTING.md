# Contributing

感谢参与 Loess QGIS。项目采用受保护的 `main`：所有修改必须从功能分支通过
Pull Request 合并。

## 工作流

1. 从最新 `main` 创建 `feature/<topic>` ；
2. 不提交模型、checkpoint、输入、输出、数据库、QGIS 工程或本机配置；
3. 在 `qgis` Conda 环境运行相关测试和完整回归；
4. 推送分支并创建 Pull Request；
5. 等待 `quality` 检查通过后 squash merge，并删除功能分支。

## 验证

```bash
conda run -n qgis pytest -q
conda run -n qgis python -m compileall -q qgis_plugins inference_scripts tests
find bash inference_scripts -type f -name '*.sh' -exec bash -n {} +
git diff --check
```

GitHub Actions 只执行轻量质量门。涉及 QGIS、PostgreSQL、CUDA、MPS 或真实模型
资产的修改，PR 必须说明实际测试平台、输入边界和仍未取得的证据。

## 变更边界

- 插件和推理运行时必须继续共用 `qgis_plugins/labeling_tool/core/` 的合同源码；
- 不允许维护完整的 macOS/Linux 双份实现；
- 不得降低 gap、overlap、outside、Artifact 身份和恢复合同；
- 第三方代码或数据必须记录来源、版本和许可；
- 安全问题不要提交公开 Issue，请按 [SECURITY.md](SECURITY.md) 报告。
