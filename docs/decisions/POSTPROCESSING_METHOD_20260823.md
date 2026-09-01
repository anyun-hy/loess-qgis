# 后处理方法决策（2026-08-23）

## 当前生产决定

- Fragmentation V3 继续作为当前生产碎片治理方案。
- Generate 只保留为激进几何去碎片的研究参考，不接入生产。
- 已测试的 RAG v1、完整 RAG quick v3 和单 Swin-B 空间联合解码 B 均未通过
  方法效果验收，不接入生产。
- RAG 详细源码、测试、设计文档和实验脚本不随公开主线分发。
- 全概率运行归档属于用户运行数据，不属于 Git 源码。

## 证据摘要

- 单 Swin-B 空间联合解码同域组件：214,728 降至 208,365，降幅 2.963%。
- 正式完整 Mask：831,531,565 像元、197,171 个 4 邻域组件、范围内 gap=0。
- 完整 RAG quick v3 六面板动态碎片减少 27.35%，但 P01–P04 归一化碎片仍
  约为 V3 的 2.76 倍。
- V3 现有全域图统计：336,689 个 raw Fusion 组件降至 140,694 个，完整范围
  内无无效像元。
- Generate 强制栅格诊断组件更少，但存在 1,157,933 个 gap 像元和 91,004
  个 overlap 像元，硬合同不通过。

上述输入基线并非完全相同，因此绝对组件数不能替代正式同源 A/B；但现有
证据足以否定当前 RAG 替代 V3 的资格。

## 从 Generate 保留的研究线索

后续 V3.1 候选可以研究，但不得直接复制 Generate 输出：

1. 分类别同类近距离 bridge；
2. 唯一外围类别的小孔洞/内部孤岛处理；
3. 分类别 MMU、距离和孔洞物理阈值；
4. 建设、水域、道路等敏感类别的明确保护；
5. 有冻结辅助证据时的大台地系统异常守卫；
6. 全域冲突裁决，而不是独立类别矢量相减。

必须继续保持 single-label、zero-gap、zero-overlap、完整范围、源组件不被
切碎和语义证据否决门。

## V3.1 启动边界

在当前工作树完成归属清理、通用 coverage 修改通过测试、并形成清晰 Git
基线前，不开始 V3.1 编码。

V3.1 应作为独立候选实现，不能直接覆盖 `fragmentation_v3.py`。只有在同一
模型、同一输入、同一完整 Mask 下同时通过碎片、语义、边界和完整性验收后，
才讨论替换生产 V3。

## 远程证据

```text
<project_root>/output/experiments/0817_swin_joint_b_20260822/full_b/audit.json
<project_root>/output/experiments/0817_swin_joint_b_20260822/full_audit_legacy_coverage/swin_joint_full_audit.json
<project_root>/output/experiments/0817_swin_joint_b_20260822/full_audit_strict_complete_mask/swin_joint_full_audit.json
```
