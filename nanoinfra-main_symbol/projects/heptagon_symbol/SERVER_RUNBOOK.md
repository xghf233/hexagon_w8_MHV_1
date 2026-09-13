# Heptagon w6 服务器运行手册

状态：2026-09-07 已接回73,008步正式训练、独立 val/test 评估和服务器代码修复。
本页保留后续复验流程，原始记录及证据边界见
[导入核对](reports/SERVER_IMPORT_2026-09-07.md)。本次未在 Mac 重跑模型或单测。
每阶段运行前按仓库 AGENTS.md 核对命令、输入输出和预算，取得用户确认。
文档里的路径是模板，不能原样执行，也不代表获准启动远端计算。

## 1. 环境与数据交接

在服务器干净工作树上通过 Git 获取已审阅代码；若服务器有未提交或分歧修改，先停下协调。
本仓库只包含 `nanoinfra-main_symbol`；不要混入其他 checkout 的 PYTHONPATH。
原始 WXF 不必上传；单独传输已经生成的五个 `w6/v1` 文件，约 5.62 MB，保留原样。
代码同步不包含数据、venv 或日志；不要使用 `rsync --delete`。

服务器 checkout 位置尚未确认。工作目录模板（将前缀替换为实际绝对路径）：

```text
/ABSOLUTE/SERVER/PATH/heptagon_w6_MHV/nanoinfra-main_symbol
```

历史 run.json 记录的数据目录为 `/root/autodl-tmp/datasets/heptagon_symbol/w6/v1_upload`，
输出目录为 `/root/autodl-tmp/runs/heptagon/w6-random-seed42`，PyTorch 2.12.1+cu130、
NumPy 2.4.6。这些是已下载文件中的运行记录，不代表当前服务器连接检查。
原运行目录应保留，新运行必须使用新输出目录。

运行前检查现成环境中的 Python >=3.12、PyTorch、NumPy、Hydra/OmegaConf、pytest。
不自动安装或升级依赖。CUDA 步骤须有 bf16 支持；使用 plain `python`，不使用 torchrun。
转换端 NumPy 为 2.3.5，但固定 split 文件跨环境读取无需相同生成环境；
真正开始训练后，恢复会锁定服务器当时的 NumPy/PyTorch 版本和代码哈希。

下面用 `/DATA/heptagon/w6/v1` 和 `/RUNS/heptagon/...` 作为待替换的绝对路径。
输出目录必须尚不存在，位于 Git 和数据目录之外。Hydra 不在源码目录创建运行产物。

## 2. CPU 单测与真实数组验证

第一步只用 CPU，读取源代码并在系统临时目录写合成小数据和测试 checkpoint；不训练大模型。
测试含微型随机模型前向和一次优化更新，按约定仅在服务器执行。

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q -p no:cacheprovider \
  projects/heptagon_symbol/tests -k 'not optional_real_converted_dataset'
```

再读真实数据：校验全部文件/切分，CPU 预编码 train，并读取小 batch。

```bash
CUDA_VISIBLE_DEVICES='' HEPTAGON_W6_DATA_DIR=/DATA/heptagon/w6/v1 \
  python -m pytest -q -p no:cacheprovider \
  projects/heptagon_symbol/tests/test_dataset.py::test_optional_real_converted_dataset
```

验收：两步均退出 0；真实数据测试不能是 skip。记录环境、命令、测试数量和输出。
不要把本地之前的转换审计当成这些新增代码的运行验证。

## 3. GPU smoke：20 次更新

一张 GPU，4/512/8 模型，batch=8；20 步，验证子集 8 项、train 诊断 8 项，禁用 compile。
会写 JSON 日志、模型/优化器 checkpoint；不是正式训练结果。耗时和显存由这次运行实测。

```bash
CUDA_VISIBLE_DEVICES=0 python -m projects.heptagon_symbol.train --config-name smoke \
  data_dir=/DATA/heptagon/w6/v1 output_dir=/RUNS/heptagon/smoke-eager
```

验收：退出 0、所有 loss/梯度有限、summary.completed_steps=20、rows_consumed=160，
存在 `checkpoints/step_000020/meta.json`、`history.jsonl` 和 `summary.json`。
20 步不要求准确率达到某个值；smoke 不证明模型已学会任务。

## 4. 恢复对照：同一 20 步 horizon 分成 10+10

以下是两次独立调用，仍使用 smoke 配方，使用新的输出目录。
不能把 max_steps 改成 10，否则学习率 horizon 会变，无法比较。

```bash
CUDA_VISIBLE_DEVICES=0 python -m projects.heptagon_symbol.train --config-name smoke \
  data_dir=/DATA/heptagon/w6/v1 output_dir=/RUNS/heptagon/resume-part1 \
  stop_after_steps=10
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m projects.heptagon_symbol.train --config-name smoke \
  data_dir=/DATA/heptagon/w6/v1 output_dir=/RUNS/heptagon/resume-part2 \
  resume_from=/RUNS/heptagon/resume-part1/checkpoints/step_000010
```

验收：第二段 initial_completed_steps=10，结束为20、rows_consumed=160；
对照连续20步运行的后10步 LR、数据顺序、loss、末尾权重及固定 prompt 输出。
数据顺序和 LR 应完全一致；数值结果需记录实际差异及容差，不默认宣称位级一致。
服务器 AI 应报告结果，不擅自放宽数据/模型/代码恢复约束来绕过错误。

## 5. Tiny overfit：32 个 train 样本，最多1000步

```bash
CUDA_VISIBLE_DEVICES=0 python -m projects.heptagon_symbol.train --config-name tiny_overfit \
  data_dir=/DATA/heptagon/w6/v1 output_dir=/RUNS/heptagon/tiny-overfit
```

要求最后全32个训练样本的**自由生成 exact=1.0**，否则明确失败并保留 summary/checkpoint。
低训练 loss 不是替代条件；不能把真实 sign/magnitude 喂给生成器。1000步并不保证能过，
若失败先分析错误，调整预算或优化器须另行说明并获准。

另经确认用 `smoke` 加 `compile=true`、新的输出目录运行一次编译 smoke；
编译有额外启动成本。它只证明编译路径能运行，不等同于吞吐基准或收敛结论。

## 6. 正式训练（必须单独批准）

```bash
CUDA_VISIBLE_DEVICES=0 python -m projects.heptagon_symbol.train --config-name w6_random \
  data_dir=/DATA/heptagon/w6/v1 output_dir=/RUNS/heptagon/w6-random-seed42
```

预算：373,800 train，batch=512，73,008 次更新，共37,380,096个样本呈现，约100.00026轮。
每步真实输入7680 tokens（512×15）、有监督1536 tokens；8192只是名义配置预算。
本次 run.json 记录参数量13,657,600；单卡 bf16 + compile。
正式运行前根据 smoke 实測耗时、显存、磁盘和全量验证成本估算资源，不凭空给小时报价。

每3650步：固定4096项 val 子集；每18250步和结束时：全量46,725项 val。
只有全量 val exact 改善才能更新 best（同分保留更早的 checkpoint）。
每3650步、best 改善和结束时保存；不自动删除旧 checkpoint。
由 train 决定的常数基线保存在 baseline.json，test 不参与训练或模型选择。

产物：run.json、baseline.json、evaluation_rows.npz、history.jsonl、
val_subset/val_full/train_diagnostic 报告、checkpoints 和 summary.json。
完整数据分布、代码及超参哈希随 checkpoint 保存。best 路径在 summary 和 checkpoint metadata 中。
加载 checkpoint 只应使用已知来源的本项目产物。

## 7. 独立验证与最终 test

先对 summary 中记录的最佳 checkpoint 做全量 val 恢复评估：

```bash
CUDA_VISIBLE_DEVICES=0 python -m projects.heptagon_symbol.eval_checkpoint \
  --checkpoint /RUNS/heptagon/w6-random-seed42/checkpoints/step_BEST \
  --data-dir /DATA/heptagon/w6/v1 --split val \
  --output /RUNS/heptagon/reports/best-val.json
```

`step_BEST` 是占位符；替换为实际 best 路径。确认与训练中全量 val 结果一致。
协议锁定并获用户确认后，才可把 `--split val` 改为 `--split test --allow-test`，
使用新的输出文件；可加 `--save-predictions` 保存原始行号和逐项预测。
训练脚本不会自动触发 test；独立评估默认也是 val。

## 8. 已有证据与后续验证边界

下载包包含正式训练日志、两个 DCP checkpoint 和独立 val/test JSON。
服务器报告称 smoke、恢复对照、tiny-overfit 与 compile smoke 已通过，
但对应完整原始输出及 CPU pytest 结果未随包提供，本次文件核对不替代复跑验收。
后续代码变更仍需按影响范围验证恢复、缓存生成、环境兼容性及训练行为。
当前设计限定单卡，不包含 DDP、轨道泛化、零样本学习或跨圈泛化。
共享 `core` 和旧 amplitude 项目没有被修改；如发现需要修改框架，先说明原因。
