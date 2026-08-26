# 当前实施状态

更新日期：2026-08-26。

## 当前结论

- macOS 与 Ubuntu 使用同一源码仓库和统一部署入口；
- PostgreSQL 是新正式 Run 的控制面状态库；
- Fragmentation V3 仍是当前生产碎片治理方案，生产入口未修改；
- V3.1—V3.4 已在独立实验分支完成同域全量比较，最终选择 V3.3；
- 主干候选只保留可配置规则系统和 V3.3 实现，V3.1、V3.2、V3.4 的
  源码、测试和实验脚本仅保存在实验分支历史；
- V3.3 尚未迁移到 approved Fusion 正式 Work Package，因此不能替代生产 V3；
- Generate 仍只作研究参考，失败 RAG 与空间联合解码仍保持归档状态。

## V3.3 当前代码

- 规则系统：`inference_scripts/fragmentation_policy/`；
- 冻结候选配置：`inference_scripts/fragmentation_policy/policies/v33.yaml`；
- V3.3 执行器：`inference_scripts/fragmentation_v33_candidate/`；
- 查询能力覆盖类别权限、源吸收、目标增长、面积阈值、单类包围、多类别
  包围、同类桥接、关系优先级和 proposal 冲突顺序；
- `inference_scripts/fragmentation_v3.py`、正式默认值和部署入口均未修改。

## 全域实验结果

Tencent 同域测试覆盖 140 个 Core、831,531,565 个 strict-valid 像元：

| 方案 | 动态碎片数 | 相对上一主要方案 | 结论 |
|---|---:|---:|---|
| V3 | 30,239 | 基线 | 当前生产方案 |
| V3.1-B | 25,983 | 比 V3 少 14.08% | 有效但提升有限 |
| V3.2 | 21,649 | 比 V3.1-B 少 16.68% | 继续改善 |
| V3.3 | 7,545 | 比 V3.2 少 65.15% | 最终选择 |
| V3.4 | 7,530 | 比 V3.3 只少 0.20% | 不采用 |

V3.3 相对 V3.2 减少 14,104 个动态碎片，动态碎片面积减少 69.65%，
组件减少 13,429，内部边界边减少 291,981。它改变 524,513 个像元，约占
完整有效范围 0.0631%。gap、overlap、outside、保护源类别损失均为 0。

V3.4 只比 V3.3 再减少 15 个动态碎片和 202 个像元，提升不足以支持采用。

## 本地验证

V3.3 收口后的规则、候选、现有 V3 后处理和权威栅格定向回归：

```text
46 passed in 2.71s
```

全仓测试：

```text
489 passed, 4 skipped, 1 failed in 88.35s
```

唯一失败是既有目录卫生检查发现工作区中的 `.opencode/` 与 ignored
`scratch/`。这两个目录包含用户工具和研究材料，本次没有删除，也没有把其中
的 `.npy`、栅格、日志或缓存提交到 Git。

## 尚未取得的证据

- 收口提交精确代码在 Tencent 上重新执行 140-Core 全量复算；
- V3.3 迁移到 approved Fusion 后的概率覆盖、单标签唯一性和 Core 唯一所有权；
- V3.3 接入正式 Work Package 后的栅格与矢量两阶段 gap、overlap、outside 验收；
- 最新提交对应的双平台部署回读和 QGIS 现场验收。

详细选择依据见
[decisions/FRAGMENTATION_V33_SELECTION_20260826.md](decisions/FRAGMENTATION_V33_SELECTION_20260826.md)。

## 历史证据

2026-07-15 至 2026-08-11 的完整测试、运行、部署和失败记录保存在：

- [archive/IMPLEMENTATION_HISTORY_20260715_20260811.md](archive/IMPLEMENTATION_HISTORY_20260715_20260811.md)
- [archive/PLUGIN_PLAN_V3_LEGACY.md](archive/PLUGIN_PLAN_V3_LEGACY.md)

历史文档不继续追加，也不能自动代表当前提交已通过相同验收。

## 更新规则

本文件只在已验证能力、部署状态或阻塞发生实质变化时更新。普通代码修改、
单次测试和会话交接不追加到这里；详细命令和日志由 commit、CI、Run Artifact
或历史归档承担。
