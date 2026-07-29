# 半自动标注工具 - 推理环境与配置检查设计

## 实现状态

本模块已于 2026-07-10 完成代码实现，包括环境检查脚本、JSON 报告、QGIS 检查表、配置持久化、启动拦截、配置向推理链路的真实传递和运行配置快照。第 2 节记录的是实现前发现的问题，用于说明本次修改原因；当前代码应按第 9 节验收。

## 1. 结论

“推理配置”不能只显示脚本目录和 `config.yaml` 中写了什么，而必须显示本次运行**实际会使用什么**。

本功能区采用以下原则：

1. 选择 `inference_scripts/` 后自动检查目录、配置、模型文件、Conda 环境、Python 依赖和计算设备。
2. 每个检查项同时显示“当前有效值”“状态”“值的来源”“出错后修改位置”。
3. 只有检查通过的配置才能启动推理；配置被修改后，旧检查结果立即失效。
4. UI 不自己复制一套模型配置。模型权重、版本、设备等仍以配置文件为准，避免“界面显示一套、脚本实际使用另一套”。
5. 环境检查必须在实际推理所用的 Conda 环境中执行，不能只检查 QGIS 自己的 Python 环境。

---

## 2. 实现前代码的真实状态

当前 `inference_scripts/config.yaml` 包含以下内容，但并非全部已经接入主流程：

| 配置项 | 当前状态 | 原因 |
|---|---|---|
| `model.semantic_weight` | 部分生效 | `predict_semantic.py` 会读取它，但 UI 没有验证文件是否存在、模型是否可加载 |
| `model.sam3_checkpoint` | 未完整生效 | `main_dock.py` 启动主流程时没有把该值传给 `run_full_pipeline()` |
| `model.device` | 未生效 | `predict_semantic.py` 当前按 CUDA 是否可用自行选择设备，没有读取该字段 |
| `model.tile_size` | 未生效 | 实际 Tile 尺寸来自 QGIS 面板的宽、高输入框 |
| `model.tile_overlap` | 未生效 | 实际重叠值来自 QGIS 面板的重叠输入框 |
| `classes.mapping` | 未生效 | `polygonize_mosaic.py` 当前使用代码内写死的类别映射 |
| `paths.scratch_dir` | 未生效 | 主流程当前使用输出目录下的 `tmp/` |
| Conda 环境 `geoai` | 生效但不可配置 | 4 个 `run_*.sh` 中均写死为 `conda run -n geoai` |

因此，现阶段不能因为“4 个脚本存在且 YAML 能读取”就显示“全部就绪”。那只能证明文件存在，不能证明配置正确，更不能证明该配置会被实际使用。

---

## 3. 功能区设计

```text
┌─ ③ 推理环境 ───────────────────────────────────────────────┐
│ 脚本目录: [/.../inference_scripts] [选择] [重新检查]        │
│ 配置文件: /.../inference_scripts/config.yaml [打开配置]     │
│                                                            │
│ 总体状态: [检查通过，可开始推理]                            │
│                                                            │
│ 检查项          当前有效值                 状态   修改位置   │
│ 管线脚本        4/4                        正常   脚本目录   │
│ Conda 环境      geoai                      正常   config.sh  │
│ Python 依赖     torch/rasterio/...         正常   geoai 环境 │
│ 语义权重        /weights/semantic.pt       正常   config.yaml│
│ 语义版本        suide-v1                   正常   config.yaml│
│ SAM3            已启用                     正常   config.yaml│
│ SAM3 权重       /weights/sam3.pt           错误   config.yaml│
│ 计算设备        cuda:0                     正常   config.yaml│
│ 类别映射        14 类                      正常   config.yaml│
│ Tile 参数       512 x 512 / overlap 64     正常   当前面板   │
│ 临时目录        /.../output/tmp            正常   程序规则   │
│                                                            │
│ 问题: SAM3 权重文件不存在                                  │
│ 修改: config.yaml -> model.sam3_checkpoint                 │
│ 当前值: /weights/sam3.pt                                   │
│ [查看详细检查日志]                                         │
└────────────────────────────────────────────────────────────┘
```

### 3.1 不在此处放“更改权重”按钮

第一版不提供语义权重、SAM3 权重的独立文件选择按钮，也不提供会直接覆盖配置的设备下拉框。

原因是这些控件会产生第二份配置来源：用户在 YAML 中写了一个值，又在 UI 中改成另一个值，最后很难判断实际使用的是哪一个。

本区域只做两件事：

- 只读展示本次运行的有效配置；
- 通过“打开配置”让用户修改唯一配置源，保存后自动重新检查。

以后如果增加“本次运行覆盖值”，必须在表格的“来源”列明确显示“本次 UI 覆盖”，并把覆盖值真实传给推理脚本。

### 3.2 状态定义

| 状态 | 含义 | 是否允许开始 |
|---|---|---|
| 未选择 | 没有选择脚本目录 | 否 |
| 检查中 | 正在读取并验证环境 | 否 |
| 错误 | 存在阻断问题，例如缺脚本、缺权重、环境不存在 | 否 |
| 警告 | 可以运行，但存在非阻断问题，例如使用 CPU 或 SAM3 被主动关闭 | 是，启动前提示一次 |
| 就绪 | 所有启用的步骤都检查通过 | 是 |

总体状态不能只放在 tooltip 中，必须直接显示文字。颜色只用于辅助，不作为唯一信息。

---

## 4. 配置内容与修改位置

### 4.1 `config.sh`：启动环境配置

`config.sh` 只负责推理进程如何启动：

```bash
CONDA_ENV="geoai"
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
```

所有 `run_*.sh` 必须读取同一个 `CONDA_ENV`，不再各自写死 `geoai`：

```bash
source "$SCRIPT_DIR/config.sh"
conda run -n "$CONDA_ENV" python ...
```

UI 显示：

- 有效值：`geoai`
- 来源：`inference_scripts/config.sh -> CONDA_ENV`
- 错误时修改位置：`config.sh` 的 `CONDA_ENV`

### 4.2 `config.yaml`：模型与推理配置

为减少本轮改动，第一版保留现有字段结构，并补全版本和 SAM3 开关：

```yaml
model:
  semantic_weight: "/absolute/path/to/semantic_14class.pt"
  semantic_version: "suide-semantic-v1"
  sam3_enabled: true
  sam3_checkpoint: "/absolute/path/to/sam3_checkpoint.pt"
  sam3_version: "sam3-v1"
  device: "auto"            # auto / cpu / mps / cuda / cuda:0
  sam_buffer_px: 32

classes:
  # Loess L2 输出 0..13 共 14 个有效地类，没有背景通道。
  # -1 仅表示未推理/未覆盖像素，不是模型类别。
  background_index: -1
  index_to_code:
    0: 12
    1: 13
    # 其余输出通道省略
  mapping:
    12: "水浇地"
    13: "旱地"
    # 其余 12 类省略
```

路径规则：

- 绝对路径直接使用；
- 相对路径以 `config.yaml` 所在的 `inference_scripts/` 为基准解析；
- UI 必须显示解析后的绝对路径，同时保留原始配置值；
- `/path/to/...` 这类模板占位符按“未配置”处理，不能显示就绪。

### 4.3 Tile、输出和临时目录

这些属于“本次任务参数”，不再放进模型配置：

| 参数 | 唯一来源 | UI 显示方式 |
|---|---|---|
| Tile 宽、高 | QGIS 面板 | `512 x 512` |
| Tile overlap | QGIS 面板 | `64 px` |
| 输出 GPKG | QGIS 面板 | 完整绝对路径 |
| 临时目录 | `<输出目录>/tmp` | 显示程序解析后的路径 |
| 跳过已确认区域 | QGIS 面板 | 开启/关闭 |

从 `config.yaml` 删除或弃用 `model.tile_size`、`model.tile_overlap` 和 `paths.scratch_dir`。环境检查发现这些旧字段时显示警告：

> 该字段当前不会被使用；请在 QGIS 面板修改对应参数。

### 4.4 类别映射

`classes.mapping` 必须成为语义矢量化的唯一类别映射来源，不能再与 `polygonize_mosaic.py` 内的 `CLASS_MAP`、`INDEX_TO_CODE` 各维护一份。

检查规则：

- 必须正好包含已确认的 14 个地类编码；
- 编码不得重复，名称不得为空；
- 模型没有背景类别，输出索引 `0..13` 全部计入 14 类；
- `background_index` 固定为 `-1`，只作为未推理/未覆盖像素的哨兵值，不能与模型输出类别混用；
- 模型输出通道索引到地类编码的顺序必须有明确配置或模型元数据，不能仅靠代码猜测；
- UI 展示“14 类”以及完整编码列表，点击后可查看全部映射。

---

## 5. 两级检查机制

### 5.1 第一级：QGIS 内静态检查

选择目录后立即执行，不启动 Conda：

1. 目录是否存在、是否可读；
2. `run_semantic.sh`、`run_mosaic.sh`、`run_polygonize.sh`、`run_sam3.sh` 是否存在且可执行；
3. `config.sh`、`config.yaml`、`check_environment.py`、`run_env_check.sh` 是否存在；
4. 配置文件是否在本次检查后被修改。

静态检查失败时直接显示具体缺失文件，不再继续做环境检查。

### 5.2 第二级：实际推理环境检查

新增：

```text
inference_scripts/check_environment.py
inference_scripts/run_env_check.sh
```

`run_env_check.sh` 与正式推理脚本使用相同的 `config.sh` 和相同 Conda 环境：

```text
QGIS -> run_env_check.sh -> conda run -n <CONDA_ENV>
     -> check_environment.py -> stdout 输出 JSON 报告
```

检查内容：

1. `conda` 命令是否存在；
2. `CONDA_ENV` 是否存在且能启动 Python；
3. `numpy`、`torch`、`rasterio`、`fiona`、`shapely`、`yaml` 是否能导入；
4. 若 SAM3 启用，`segment_anything` 或当前选定的 SAM3 实现是否能导入；
5. 语义权重、SAM3 权重是否存在、可读、扩展名合理；
6. `device=auto` 解析后的实际设备；
7. 若明确指定 CUDA/MPS，检查 `torch.cuda.is_available()`、MPS 可用性和设备编号；
8. 类别映射是否合法；
9. 权重能否完成最小加载验证；
10. 临时目录和输出目录是否可写。

环境检查应使用 `QProcess` 或后台任务执行，避免冻结 QGIS 主界面。路径输入变化时只做静态检查；用户选择完目录、点击“重新检查”或配置文件保存后，再执行完整环境检查。

---

## 6. 检查报告协议

`check_environment.py` 的标准输出只写一份 JSON，日志写标准错误。QGIS 不解析自然语言日志来判断结果。

```json
{
  "schema_version": 1,
  "status": "error",
  "config_fingerprint": "sha256:...",
  "effective": {
    "conda_env": "geoai",
    "device": "cuda:0",
    "semantic_weight": "/abs/weights/semantic.pt",
    "semantic_version": "suide-semantic-v1",
    "sam3_enabled": true,
    "sam3_checkpoint": "/abs/weights/sam3.pt",
    "sam3_version": "sam3-v1",
    "sam_buffer_px": 32,
    "class_count": 14
  },
  "checks": [
    {
      "id": "sam3_checkpoint",
      "status": "error",
      "value": "/abs/weights/sam3.pt",
      "source": "config.yaml:model.sam3_checkpoint",
      "message": "文件不存在",
      "fix": "修改 config.yaml 的 model.sam3_checkpoint"
    }
  ]
}
```

字段要求：

- `status` 只能是 `ready`、`warning`、`error`；
- `effective` 是实际将传给推理主流程的值；
- `source` 必须精确到文件和字段；
- `fix` 必须告诉用户修改哪里，不能只写“配置错误”；
- `config_fingerprint` 用于判断检查后配置是否又被修改。

---

## 7. 插件代码职责

### 7.1 新增 `core/inference_config.py`

该模块负责：

- 静态目录检查；
- 启动和解析环境检查报告；
- 保存最后一次 `ConfigReport`；
- 判断报告是否过期；
- 给 `InferenceRunner` 提供唯一的有效配置对象。

不要把 YAML 解析、路径校验和状态判断全部堆进 `main_dock.py`。

### 7.2 修改 `gui/main_dock.py`

主要改动：

- 把“推理配置”拆成“③ 推理环境”和“④ 输出”；
- 增加检查表、总体状态、重新检查、打开配置、详细日志；
- 选择路径后刷新静态状态并启动完整检查；
- 检查为错误或过期时禁用“开始标注”；
- 开始前再次验证配置指纹；
- 将 `ConfigReport.effective` 传给 `run_full_pipeline()`；
- 默认目录优先读取 `QgsSettings`，没有历史值时再从当前 QGIS 项目目录推导；
- 保存并恢复 `inference_path`，不要求用户每次重新选择。

### 7.3 修改 `core/inference_runner.py`

`run_full_pipeline()` 必须接收并实际使用：

```text
semantic_weight
semantic_version
sam3_enabled
sam3_checkpoint
sam3_version
sam_buffer_px
device
class_mapping
```

当前必须修正的调用链：

- `run_full_pipeline()` 调用 `run_semantic()` 时传入 `semantic_weight` 和 `device`；
- SAM3 关闭时不进入精修阶段；
- SAM3 开启时传入 `sam3_checkpoint`、`sam3_version`、`sam_buffer_px` 和 `device`；
- `run_polygonize()` 使用同一份类别映射和 `semantic_version`；
- 不再让脚本自行选择另一套默认值覆盖已检查配置。

### 7.4 修改推理脚本

- 4 个 `run_*.sh` 统一读取 `config.sh` 中的 `CONDA_ENV`；
- `predict_semantic.py` 接收主流程传入的权重和设备；
- `polygonize_mosaic.py` 移除写死的类别表，读取传入的类别映射快照；
- `sam3_refine.py` 使用检查报告中的 SAM3 权重和设备；
- 推理失败时返回非零退出码，不允许静默退回模拟结果；
- 每次运行在结果目录保存 `run_config_snapshot.json`，记录本次真正使用的配置和版本。

---

## 8. 启动前拦截规则

点击“开始标注”时按以下顺序处理：

1. 没有检查报告：提示“请先检查推理环境”；
2. 报告状态为错误：显示第一条阻断问题和修改位置；
3. 配置指纹已变化：自动重新检查，检查完成前不启动；
4. 报告为警告：列出警告，用户确认后允许继续；
5. 报告就绪：把 `effective` 配置对象原样传入主流程；
6. 在输出目录写入 `run_config_snapshot.json` 后开始切片和推理。

这样可以保证 UI 中看到的配置、检查通过的配置和真正运行的配置是同一份。

---

## 9. 验收测试

### 9.1 UI 与静态检查

- 首次打开插件：自动推导项目根目录下的 `inference_scripts/`；
- 再次打开插件：恢复上次选择的目录；
- 选择空目录：明确列出缺失文件；
- 删除任意管线脚本：总体状态变为错误，“开始标注”禁用；
- 修改 `config.yaml`：旧检查立即标记为过期并自动重新检查；
- “打开配置”定位到当前正在使用的 `config.yaml`。

### 9.2 配置错误定位

- 语义权重不存在：显示当前路径及 `model.semantic_weight`；
- SAM3 权重不存在：显示当前路径及 `model.sam3_checkpoint`；
- `device=cuda:1` 但只有一张 GPU：显示设备编号错误及 `model.device`；
- `device=mps` 但当前 PyTorch 不支持 MPS：显示 MPS 不可用及 `model.device`；
- 类别少于或多于 14 类：显示实际数量及 `classes.mapping`；
- YAML 格式错误：显示文件、行号和解析错误，不得静默忽略；
- Conda 环境不存在：显示环境名及 `config.sh -> CONDA_ENV`；
- Python 缺少依赖：显示缺失包和需要修改的 Conda 环境。

### 9.3 配置是否真正生效

- UI 显示的语义权重路径与语义脚本日志一致；
- UI 显示的设备与 PyTorch 实际设备一致；
- `config.yaml` 关闭 SAM3 后，主流程不调用 `run_sam3.sh`；
- UI 显示的 14 类映射与 `semantic_polygons` 中的类别一致；
- Tile 尺寸和 overlap 与当前面板一致，不受 YAML 旧字段影响；
- 结果目录中的 `run_config_snapshot.json` 与 UI 的有效配置一致。

### 9.4 QGIS 行为

- 环境检查过程中 QGIS 地图仍可平移、缩放；
- 检查失败不会改变当前地图工具；
- 插件关闭或重载时能终止未完成的检查进程；
- 通过安装脚本部署后热重载，配置状态仍能正确恢复。

---

## 10. 实施顺序

1. 统一 `config.sh` 和 4 个启动脚本的 Conda 环境来源；
2. 新增 `check_environment.py`、`run_env_check.sh` 和 JSON 报告协议；
3. 新增 `core/inference_config.py`；
4. 完成推理环境 UI、状态表和错误定位；
5. 把有效配置完整接入 `InferenceRunner`；
6. 移除类别映射和设备选择的重复默认值；
7. 保存运行配置快照；
8. 安装插件、热重载并按第 9 节逐项验收。

第一版完成标准不是“配置能显示”，而是：用户能明确看到本次会用哪个模型、哪个环境、哪个设备和哪套类别；任何问题都能直接定位到具体文件和字段，并且 UI 显示值与实际推理值一致。
