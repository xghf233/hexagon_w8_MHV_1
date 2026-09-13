# Heptagon 三圈 MHV — 当前进度

最后更新：2026-09-07。

当前状态：**已从用户下载的服务器压缩包接回配置修复、正式训练结果和 best/最终 checkpoint。**
随包记录显示 73,008 次更新，全量 val/test 各 46,725 项 exact=1.0，采用 random-row split。
本次核对文件、哈希和 JSON 记录，未在 Mac 重跑单测、模型加载、推理或训练。
详细来源、导入清单与证据边界见
[SERVER_IMPORT_2026-09-07.md](reports/SERVER_IMPORT_2026-09-07.md)。

## 1. 工程位置与复用边界

本项目最初在 `HZQ-git/nanoinfra-main_symbol` 中开发，随后将完整 Symbol edition
发布到 `heptagon_w6_MHV`。现在本仓库作为主开发入口，本地编辑，经 GitHub 同步到
服务器运行副本。保留 `nanoinfra-main_symbol/` 层级及旧 amplitude 依赖，不重建框架。

服务器包基于本仓库的 `9bae12d` 加未提交的 `train.py` 修复。第一阶段已将它接回
HZQ-git 并核对产物；本阶段将核对后的代码、报告和状态说明合入本仓库。
本阶段没有修改 HZQ-git、standard 版或仓库外的模型数据，也没有暂存、提交或推送。

| 部分 | 本次处理 |
| --- | --- |
| GPT 主体、模型组装、AdamW 参数组、LR 调度、KV-cache、DCP | 直接复用既有 `core`，未修改其实现 |
| word-bidirectional mask 工具 | 复用旧 `amplitude_symbol/blocks/attention.py` 的通用函数，显式传入 `[1,7)`；未修改该文件 |
| heptagon alphabet、转换器、tokenizer、dataset | 新增项目专用模块，不改旧十字母输入和词表 |
| 训练入口、评估和 checkpoint 封装 | 新增项目专用实现；训练循环未直接调用旧 `core.training.trainer.Trainer` |
| 配置与测试 | 新增 `configs/` 和 `tests/`，不替换旧实验配置 |
| Symbol 版 `pyproject.toml` | 前序工作已增加 heptagon 包发现和 YAML 配置打包；未新增依赖 |
| README、计划、进度导航 | 更新，明确两条实验线和验证状态 |
| standard 版、旧训练结果、原 WXF | 本次 heptagon 工作未修改 |

单独项目化的原因：三圈 heptagon 的 42-letter alphabet、六字母 word、1048 词表、
随机行切分与旧项目的六字母 alphabet、十字母 word、1012 词表、D3 轨道切分不同。
不能仅替换旧数据路径或直接沿用旧 mask 默认值。

项目内单卡训练循环负责显式样本预算、全量/子集验证区分和完整 checkpoint。
仍复用原模型、优化器、调度和存储机制；没有改写共享 Trainer 或强行迁移旧实验。
由于 mask 工具仍从旧项目导入，heptagon 文件夹不是可脱离整个 Symbol edition 独立运行的包。

## 2. 已完成的执行工作：数据转换

- 输入：`hep_w6_phy.wxf`，仅处理三圈 weight 6 MHV；未启动 Wolfram 或符号求值。
- 保留 467,250 个唯一非零项，word 长度 6，42 个字母，33 种系数，范围 −48～48。
- 固定 PCG64 seed=42 随机行切分：train 373,800 / val 46,725 / test 46,725。
- 不做轨道分组、对称压缩或增强；训练集覆盖全部33种系数值。
- 数组保存后读回、全局查重、切分覆盖/互斥，以及重建完整 WXF 表达式字节的哈希核验均通过。
- 实测转换约6.90秒，峰值RSS约121.11MiB；五个产物总计5,624,185 bytes。

数据目录（仓库外）：

```text
/Users/hzq/hep_th/Building Intelligent Models from Scratch/nanoinfra-artifacts/heptagon_symbol/w6/v1/
```

输入/输出哈希和详细审计分别在产物 `metadata.json` 与 `audit.json`。
该审计证明转换数据完整，不替代后来新增 tokenizer、训练或 checkpoint 的运行验证。

## 3. 工程与导入后的验证状态

| 模块 | 内容 | 执行验证状态 |
| --- | --- | --- |
| `dataset.py` | 数组/哈希检查、固定 split、CPU 预编码、逐轮洗牌、恢复采样位置 | 随包正式运行使用；源码哈希一致 |
| `tokenizer.py` | 1048词表、base-1000、长度16序列、sign/magnitude/EOS监督、严格解码 | 随包正式运行使用；源码哈希一致 |
| `model.py` | 复用GPT，编译前安装 `[1,7)` mask，检查运行时配置 | 随包正式运行使用；源码哈希一致 |
| `train.py` | 单卡bf16训练、日志、显式步数预算、验证和保存 | 合入 Hydra 顶层键豁免；73,008步记录已接回 |
| `evaluator.py` | 长度8无标签prompt、贪心自由生成、exact/sign/magnitude/invalid及分组指标 | 全量 val/test 评估 JSON 已接回 |
| `checkpoint.py` | 模型/优化器、数据进度、RNG、训练配置和数据/代码哈希 | 两份 DCP 已保存并核对元数据；未本地加载 |
| `eval_checkpoint.py` | 独立恢复评估；默认val，test需显式启用 | 独立 val/test JSON 已接回；未本地复评 |
| `tests/` | 编码、无泄漏负对照、缓存生成、配置、参数计数、恢复后下一步更新等 | 本次包没有完整 pytest 输出；未重跑 |

服务器报告称 smoke、恢复对照、tiny-overfit 与 compile smoke 已通过；
相关阶段的完整原始产物没有随包提供。正式训练记录与独立评估结果可直接核对，
但不能仅凭报告将所有单测或“无泄漏”结论标记为独立验证通过。
本次只修改本地文件，没有连接服务器、安装依赖、运行模型、提交或推送。

## 4. 运行配置与首轮记录

| 配置 | 模型 | batch | 预算 | 用途 |
| --- | --- | ---: | ---: | --- |
| `smoke` | 4层 / 512维 / 8 heads | 8 | 20步 | 检查训练、评估和保存路径 |
| `tiny_overfit` | 同上 | 32 | 1000步、固定32项train | 要求全32项自由生成exact=1，否则报失败 |
| `w6_random` | 同上 | 512 | 73,008步，约100次train遍历 | 首轮正式实验 |

正式 run.json 记录模型参数量13,657,600，837.925秒，峰值显存941,300,224 bytes。
周期验证固定4096项val；全量val为46,725项，只有全量val exact改善才更新best。
训练不自动评估test，不将随机切分结果与旧D3隔离实验当成同难度比较。

best 为 `step_073000`，最终为 `step_073008`，两者全量 val exact=1.0。
独立 best-val 与 best-test 各记录46,725/46,725项正确；test JSON 含逐项预测。
summary 中 `test_evaluated=false` 仅表示训练阶段没有评估 test，后续独立结果
以 `best-test.json` 为准；`acceptance=not_requested` 不是全套验收状态。
本地仓库外产物位置及保留文件见导入审计；中间19个 checkpoint 未包含在下载包中。

## 5. 后续工作

1. 审阅本仓库的合入差异，获准后暂存、提交，再单独确认推送 GitHub；日常开发集中在本仓库。
2. 如需完成逐阶段证据归档，从服务器补回 CPU pytest、smoke、恢复对照等原始记录。
3. 后续复评或恢复前核对 CUDA 环境、数据与代码哈希；不能修改旧 checkpoint 的哈希来绕过校验。
4. 新实验使用新输出目录；多 seed、orbit-grouped 或跨 weight 实验需另行确定协议。
5. test 已有一次独立评估记录，后续复评应明确标注为复评，不能再次称为首评。

每阶段命令、读写范围和验收条件见 [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)。
接口见 [README.md](README.md)，完整设计和实施历史见
[HEPTAGON_W6_PLAN.md](../amplitude_symbol/docs/HEPTAGON_W6_PLAN.md)。
后续实际运行结果应记录命令、代码版本、环境、退出状态和产物路径，再更新本进度。
