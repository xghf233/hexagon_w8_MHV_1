# M2/38 服务器交接：先 GPU smoke

2026-09-14：本地候选代码已写、人工静态核对；**未执行、未上传、未提交/推送**。
用户尚未提供服务器与环境信息。以下是待另行授权的命令，不是当前已运行的步骤。

## 1. 环境、代码与数据

新入口为 `projects.hexagon_c2_a3`，不是旧 0y 训练器。共享 core/GPT 类，不共享权重、优化器或 checkpoint。
新包与 configs YAML 已加入 pyproject；代码目前仅在本地，服务器 GitHub checkout 不会自动包含本次修改。

使用服务器已有的 Python>=3.12、PyTorch、NumPy、Hydra>=1.3、OmegaConf>=2.3 环境。
旧项目的 PyTorch 2.12.1+cu130 / NumPy 2.4.6 仅是历史线索，不是当前服务器探测结果。
公共框架使用较新的 DCP/FSDP API，不保证任意 torch 版本兼容。不要在 Mac 安装环境或运行 Python 检查，
也不要未经确认升级服务器依赖。可选 Liger 的实际头类型会记录并参与恢复校验。

只暴露一张支持 bf16 的 CUDA GPU，plain python，不用 torchrun；compile=false。
CUDA 门禁先于数据发布和模型构建，没有 CPU/MPS 回退。必要的 gzip/NumPy/哈希/格式检查/RNG/日志在主机上处理，
属于 GPU 工作流，不是另设 CPU 模型核验阶段。

两个源文件须同目录存放，独立于代码与输出目录：

| 文件 | 固定 SHA-256 |
| --- | --- |
| four_loop_mhv_symbol_native_x32.jsonl.gz | 363bc50186af77f7a8b2f726964ff305ab2522b576a8a47f2b9811419dd69de2 |
| scaling_manifest.json | fdd7c25ec846fd32426055dc26500536c86d4fdbd9c5aff9c5db9bb0c358a00a |

不拿旧训练数组或旧 checkpoint 替代 M2 数据；不重新提取、不再次乘32。
新数据与运行目录须彼此独立、位于代码和源目录之外，输出尚不存在；失败不自动删除产物。

## 2. 首次命令：仅 smoke

替换路径并获得服务器执行授权后：

```bash
cd /SERVER/PATH/hexagon_w8_MHV/nanoinfra-main_symbol
CUDA_VISIBLE_DEVICES=0 python -B -m projects.hexagon_c2_a3.gpu_smoke \
  --source /SERVER/PATH/Symbol_Data/hexagon_mhv_4loop_native_x32/four_loop_mhv_symbol_native_x32.jsonl.gz \
  --data-dir /SERVER/PATH/datasets/hexagon/m2_adjacent38/random_v1 \
  --output /SERVER/PATH/runs/hexagon/m2_adjacent38/smoke_001
```

重用已有发布数据：省略 `--source`，传入 `--metadata-sha256 TRUSTED_SHA256` 并选择新运行目录。
首次可信摘要保存在 environment.json 和成功报告中；不要为绕过错误而重算一个摘要。
若失败发生在 smoke 目录建立之前，发布器打印并保留 staging 目录以供诊断。

smoke 执行范围：

1. 完整源校验后构造40,776条相邻非零2y目标，验证13,338个输入及零冲突。
2. random-row PCG64 seed42，32,620/4,077/4,079；位置不参与输入分组。
3. 发布8个数组、splits、metadata/audit；逐数组读回，重新读取源并逐目标独立复查36项。
4. CUDA核对全池格式/编码；val/test不进入模型诊断或梯度更新。诊断前向仅用train。
5. 非零残差诊断模型检查答案/PAD不泄漏、错误mask负对照、4-token平均loss、cache/full生成。
6. 全新初始化连续20步，再独立全新初始化10步、恢复同一20步horizon继续10步，共40次更新。
7. 比较末尾参数、AdamW状态、RNG、实际下一批数据，以及固定8行val的自由生成报告。

参数/优化器张量：rtol=1e-5、atol=1e-6；sampler/RNG/生成token须一致。
cache/full logits 与可选 fused CE：rtol=1e-2、atol=1e-2，容纳 bf16 内核舍入，greedy仍须完全相同。
无泄漏对照要求受保护hidden完全相同；错误mask必须导致可观测差异。失败不能自动放宽容差。
无学习基线会对完整val计算，但smoke不选正式best、不运行tiny/pilot、不评估test。

## 3. 产物与验收

数据：conditions、y_types、y_positions、coefficients、words、source_row_ids、family_ids、input_group_ids
共8个 .npy，另有 splits.npz、metadata.json、audit.json。所有多字节整数为little-endian，禁用pickle。
位置/source行号为零基；位置/背景/ID只用于审计，不输入模型。input_group键仅为(C36,y_types)。

```text
environment.json
cuda_contract_checks.json
continuous/  prefix/  resumed/
  run.json / history.jsonl / summary.json
  evaluation_rows.npz
  baseline_val_full.json
  val_subset_*.json / train_diagnostic_*.json
  checkpoints/step_*/       DCP + meta.json
gpu_smoke_report.json
```

只有命令退出0且最终报告为 `status=passed_gpu_smoke`，才能记录GPU smoke通过。
audit.status=passed只代表数据检查；“已编写检查”不等于检查已通过。
失败保留产物并报告，不换切分/改标签/缩batch/放宽容差/自动继续pilot。
实际耗时显存尚未测量；暂至少预留2 GiB磁盘供多个DCP使用，以实际报告为准。

## 4. 后续阶段：分别授权

Tiny-overfit：

```bash
CUDA_VISIBLE_DEVICES=0 python -B -m projects.hexagon_c2_a3.train \
  --config-name tiny_overfit \
  data_dir=/SERVER/PATH/datasets/hexagon/m2_adjacent38/random_v1 \
  metadata_sha256=TRUSTED_SHA256 \
  output_dir=/SERVER/PATH/runs/hexagon/m2_adjacent38/tiny_001
```

确定性选择32个不同train输入，尽量覆盖位置/y配对/符号；选择规则及行ID存档。
最多1000步，全部自由生成exact=1才通过，之后不迁移这些权重。

Pilot另行从头开始：

```bash
CUDA_VISIBLE_DEVICES=0 python -B -m projects.hexagon_c2_a3.train \
  --config-name m2_adjacent_random_pilot \
  data_dir=/SERVER/PATH/datasets/hexagon/m2_adjacent38/random_v1 \
  metadata_sha256=TRUSTED_SHA256 \
  output_dir=/SERVER/PATH/runs/hexagon/m2_adjacent38/pilot_001
```

4/256/4/4、vocab117、预计3,206,400参数；batch64、5000步、平均呈现9.81次。
warmup200、末20%线性退火，是完整5000步计划，不是35024步前缀；末次实际更新倍率0.001。
每250步固定512行val，每1000步及结束完整4077行val；只有full-val exact选best，并列保留较早步数。
检查overall与seen/unseen-input38、输入组宏平均、train-only查表/众数基线的差距。

恢复：显式 `resume_from=/ABSOLUTE/PATH/checkpoints/step_XXXXXX`，使用新output_dir。
数据、依赖代码哈希、词表/mask、模型/头、优化器、batch、horizon、评估协议和环境须匹配。
stop_after_steps只限制该进程额外更新数，不改变horizon；不能无说明地改变max_steps续接已退火完的pilot。

独立评估入口 `projects.hexagon_c2_a3.eval_checkpoint`，必传
`--checkpoint --data-dir --metadata-sha256 --output`，默认完整val。
冻结模型及协议后，最终test须加 `--split test --allow-test`；不能用test选best或调参。
