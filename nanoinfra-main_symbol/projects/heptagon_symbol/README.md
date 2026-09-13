# Heptagon 三圈 MHV 数据、训练与结果

本项目已完成 weight-6 原始 WXF 的无损转换与逐样本随机切分。
2026-09-07 已接回服务器的配置校验修复与正式训练记录：73,008 次更新，
全量 val/test 各 46,725 项的 exact 均为 1.0（random-row split）。
best 与最终 checkpoint 存放在仓库外，20 个记录的源码哈希均与更新后的本版一致。
此次只核对文件和运行记录，未在 Mac 重跑模型或单测。完整设计见
[HEPTAGON_W6_PLAN.md](../amplitude_symbol/docs/HEPTAGON_W6_PLAN.md)。

最新状态与工程复用边界见 [PROGRESS.md](PROGRESS.md)；服务器执行顺序见
[SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)。本项目最初在 HZQ-git 中开发，
现在以 `heptagon_w6_MHV` 为主开发仓库，复用其 `core` 和旧 amplitude 的 attention 工具。

服务器原始报告见 [reports/ACCEPTANCE_REPORT.md](reports/ACCEPTANCE_REPORT.md)；
本次同步范围、证据核对和模型位置见
[reports/SERVER_IMPORT_2026-09-07.md](reports/SERVER_IMPORT_2026-09-07.md)。
报告生成脚本已收录为 `make_report.py`，从本版根目录运行
`python -m projects.heptagon_symbol.make_report RUN_DIR NEW_REPORT_DIR`。
它读取已有 JSON/JSONL 并需要 matplotlib；此次未安装依赖或重新绘图。

从 `nanoinfra-main_symbol/` 运行，转换仅依赖 NumPy 与 Python 标准库：

```bash
python -m projects.heptagon_symbol.convert_data \
  --source /absolute/path/to/hep_w6_phy.wxf \
  --output /absolute/path/to/heptagon_symbol/w6/v1 \
  --seed 42
```

转换器只接受计划中已核查 SHA-256 的文件，不执行 Wolfram 代码。
输出目录必须不存在；先写 staging，完整读回校验后发布。
若转换失败，已创建的 staging 保留用于排查，原始输入和已有输出不会被覆盖。

输出：

| 文件 | 内容 |
| --- | --- |
| `words.npy` | `uint8[467250,6]`，字母按 `a11..a17,a21..a27,...,a67` 编为 0..41 |
| `coefficients.npy` | `int16[467250]`，原始系数 −48..48 |
| `splits.npz` | `train/val/test` 的 int32 行索引，373800/46725/46725 |
| `metadata.json` | 固定编号、来源及产物哈希、转换器源码哈希、环境与切分协议 |
| `audit.json` | 完整数据审计、各 split 分布、示例、耗时与进程峰值内存 |

加载不需要 Wolfram 或 pickle：

```python
from pathlib import Path
import numpy as np

root = Path('/absolute/path/to/heptagon_symbol/w6/v1')
words = np.load(root / 'words.npy', mmap_mode='r', allow_pickle=False)
coefficients = np.load(root / 'coefficients.npy', mmap_mode='r', allow_pickle=False)
with np.load(root / 'splits.npz', allow_pickle=False) as splits:
    train_indices = splits['train']
```

转换包含全局 word 查重、系数与字母频数核验、数组保存后读回及切分覆盖检查。
还从全部输出行重新构造源文件的 WXF 表达式字节，并比对解压内容 SHA-256，
验证字母、顺序、显式系数和隐含 +1 均未改变。

切分为 seed=42 的 PCG64 随机行切分，不做轨道分组、对称性压缩或增强。
相关 word 可以跨 split，结果只按同圈随机留出任务解释。

所有生成数据与运行产物保存在 Git 仓库外。生产转换是用户授权的本地数据处理；
模型推理、训练及项目单测仍遵循仓库的 AutoDL 执行规则。

## 已完成的本地转换

首版数据位于：

```text
/Users/hzq/hep_th/Building Intelligent Models from Scratch/nanoinfra-artifacts/heptagon_symbol/w6/v1/
```

467,250 项全部通过完整性审计，train/val/test 为 373,800 / 46,725 / 46,725。
训练集覆盖全部 33 种系数值。输出总计 5,624,185 bytes；实际转换约 6.90 秒，
进程峰值 RSS 约 121.11 MiB。未运行模型训练或模型测试。

此次使用的现成 Python 解释器为：

```text
/Users/hzq/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

完整调用参数记录在 `metadata.json` 的 `converter.argv` 中。
如需重新转换，选择不存在的新版本目录；转换器不会覆盖已发布的 `v1`。

## 数据加载与 tokenizer 接口

新增 `dataset.py` / `tokenizer.py`；使用本版既有 NumPy、PyTorch 和 `core`，
不依赖旧 amplitude 项目的固定字母表、轨道代码或 Wolfram。

- `HeptagonDataset(data_dir, split)`：有限长度、只读 split 视图；`dataset[i]`
  返回六个 letter ID 和 Python 整数系数，`dataset.indices[i]` 是原 WXF 行号。
  启动时核对 metadata、四个文件哈希、数组结构、全局 word 唯一性及切分互斥/覆盖。
  不重新切分。传入 `expected_metadata_sha256` 可额外锁定可信 manifest。
- `HeptagonDataLoader(dataset, ...)`：仅接收 train，单进程/单卡，无限循环；
  CPU 上一次预编码，每次只转移一个 batch。完整 batch 可跨 epoch，末尾不丢样本。
  固定切分 seed 与训练洗牌 seed 是不同概念；洗牌只改变 train 内部顺序。
- `encode_word` 接收六个独立标签，如 `['a11', ...]`；`assemble_sequence`、
  `assemble_batch`、`encode_prompt` 接收整数 letter ID，不接收拼接字符字符串。
- `encode_coefficient` / `decode_coefficient` 支持一般 base-1000 多块整数；
  `assemble_batch` 是三圈专用的单块快速路径，要求 `|c| < 1000`。
  Tokenizer 支持规范的 `+0`，但非零数据集加载器拒绝零样本。
- `decode_generated` 要求 EOS，拒绝负零、多块前导零、非法 token 或 EOS 后的非 PAD。
  非法输入抛出 `ValueError`；之后评估器应捕获它并计入 invalid rate。

以下是服务器上的调用示意，路径由服务器实际存储位置决定：

```python
from projects.heptagon_symbol.dataset import HeptagonDataset, HeptagonDataLoader

train = HeptagonDataset('/absolute/server/path/heptagon_symbol/w6/v1', split='train')
loader = HeptagonDataLoader(train, batch_size=512, sequence_len=16,
                            seed=42, shuffle=True, device='cuda')
batch = next(loader)
```

Batch 中 `idx/targets/token_types/target_types/loss_weights` 均为 `[B,15]`：
只监督 sign、magnitude、EOS；其余 `targets=-1`。`token_types` 与输入位置对齐，
`target_types/loss_weights` 与 shift 后的目标对齐。另有 JSON 可序列化的 `state_dict`。
预编码 train 的三份常驻张量合计约 114 MiB（估算，不是运行峰值；另有临时张量）。

使用 `loader.state_dict()` 保存读取进度，`loader.set_state(state)` 或
`loader.load_state_dict(state)` 恢复。每个 batch 自带下一未读位置状态，以兼容
现有 Trainer/checkpoint manager。状态绑定数据/切分哈希、词表、序列长度、
batch size、训练 seed、洗牌算法和 NumPy 版本；不匹配时报错，不静默重开。
epoch 顺序由 `PCG64(SeedSequence([seed, epoch]))` 重建，无须重放之前的 batch。
这只处理数据读取状态；新 `checkpoint.py` 在此之上封装模型、优化器、调度配置和 RNG。

**尚不能直接套用旧 word-bidi 训练入口。** 新 prompt 长度为 8，word 双向区间
必须为 `[1,7)`；旧默认 `[1,11)` 会覆盖真实答案位置。当前模块只提供位置常量和
编码 metadata。新 `model.py` 在模型组装后、编译前显式安装该 mask；
`evaluator.py` 与 `eval_checkpoint.py` 使用相同的协议进行生成和恢复。

## 服务器验证（保留复验命令）

按仓库约定，测试须在服务器经确认后运行。选定 `nanoinfra-main_symbol/` 为工作目录，
使用服务器现有环境（Python >=3.12、NumPy、PyTorch、pytest）。不要把两个 edition
同时放到 `PYTHONPATH`，不要直接调用本地 Mac 的 Python 路径。

第一阶段只跑 CPU 合成数据测试，无需真实 WXF、GPU、下载模型或网络；
测试包含微型随机模型的前向、优化更新和 checkpoint 恢复：

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q -p no:cacheprovider projects/heptagon_symbol/tests \
  -k 'not optional_real_converted_dataset'
```

合成数组由 pytest 写入系统临时目录，不修改真实产物。测试检查真实示例编码、
全部 token 类型、base-1000 边界和非法格式、loss shift、
数组篡改/重复 word/错误 split、路径迁移、跨轮读取和恢复后逐 batch 一致性。

第二阶段读取已传到服务器的转换产物，仍然仅用 CPU，不训练模型：

```bash
CUDA_VISIBLE_DEVICES='' HEPTAGON_W6_DATA_DIR=/absolute/server/path/heptagon_symbol/w6/v1 \
  python -m pytest -q -p no:cacheprovider \
  projects/heptagon_symbol/tests/test_dataset.py::test_optional_real_converted_dataset
```

该测试校验真实 467,250 项、train 数量与系数覆盖，并预编码训练集、读取小 batch。
未设置 `HEPTAGON_W6_DATA_DIR` 时该测试会 skip，不能把 skip 当成真实数据验证通过。
本次产物包含正式训练及独立评估结果，但没有 CPU pytest 的逐项输出；
因此不能据此把整个单测套件标记为通过。服务器报告的 smoke、恢复对照等结论
与本次可核对的原始文件范围，见同步审计。

## 训练与评估工程（2026-09-07，已接回首轮运行产物）

新增 `train.py`、`model.py`、`evaluator.py`、`checkpoint.py`、`eval_checkpoint.py`
及 `configs/{smoke,tiny_overfit,w6_random}.yaml`。
详细命令、预算和验收条件见 [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)。

- 保留现有 GPT、AdamW 参数组、学习率调度、KV-cache 和 DCP 底层实现。
  使用项目内单卡循环管理样本预算、分层验证和完整 checkpoint；没有修改共享 Trainer，
  也不依赖其有限 GPU 型号表。实际仍要求支持 bf16 的 CUDA GPU。
- 正式配置为 4 层 / 512 维 / 8 heads、batch=512、73,008 步，约 100 次 train 遍历。
  smoke 20 步；tiny-overfit 只取 train 中固定随机 32 项，最多 1000 步。
- 所有 step 目录与日志使用 **已完成 optimizer updates 数（1-based）**。
  DCP 内部兼容字段 `step=completed_steps-1`；不要和旧日志的零基 step 混淆。
- 每 3650 步评估固定随机 val 子集 4096 项；每 18250 步及每次调用结束时评估全量 val。
  `best` 仅由全量 val exact 改善触发保存；smoke/tiny 不做全量生成验证，`best` 为 null。
  始终另存 train 诊断指标和由 train 选择的常数基线，不自动评估 test。
- 输出目录必须是仓库与数据目录之外的**新目录**，恢复也使用新目录。
  保存 checkpoint 先写 staging 再发布，不覆盖或自动删除已发布 checkpoint。
  正式预算默认约 21 个周期/末尾 checkpoint，另可能有 best 保存点；请预留数 GB 磁盘。
- 每个 checkpoint 记录模型/优化器状态、下一未读数据位置、Python/NumPy/Torch CPU/
  训练 GPU RNG、完整超参、数据/代码哈希、编码/mask 及 best 记录。
  调度器由固定 horizon 和 completed_steps 恢复，不重新 warmup。
- 恢复严格拒绝更改数据、训练超参、代码、NumPy/PyTorch 版本；允许改变本次停止步数、
  输出位置和日志/保存频率。恢复只加载可信的本项目 checkpoint，不支持旧模型热启动。
  best 路径可能指向上一次运行目录；跨机器迁移时需一起保留对应 checkpoint。
- 不承诺跨 GPU 或编译环境位级一致；相同环境下的连续/分段训练对照仍是上线验收项。

新增测试覆盖 `[1,7)` 无泄漏（含旧 `[1,11)` 的负对照）、缓存/完整前向生成一致性、
生成器 EOS 补齐、输入无真实标签、评估前后模式恢复、配置组装、参数量、
checkpoint 模型/优化器/采样器/RNG 恢复及下一步更新一致性。
单测文件予以保留；本次没有重跑，随包也没有完整 pytest 结果，测试状态单独记录。
