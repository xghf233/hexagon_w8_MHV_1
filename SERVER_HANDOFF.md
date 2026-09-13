# GPU 交接：先冒烟，暂不正式训练

状态：2026-09-13，命令已准备，**尚未执行**。当前只完成本地代码配置；服务器 SSH、
checkout、Python 环境路径和运行时间由用户之后确认。不要按本页自动连接服务器或安装依赖。

## 1. 环境与传输范围

参考旧 w6 的 `projects/heptagon_symbol/SERVER_RUNBOOK.md` 和
`reports/SERVER_IMPORT_2026-09-07.md`：历史记录为 Python >=3.12、
PyTorch **2.12.1+cu130**、NumPy **2.4.6**、RTX 5090 / CUDA 13.0。
这些不是对当前服务器的探测结果，旧文档也没有确认可直接复用的 venv 路径。

优先使用服务器已有环境，另需 Hydra >=1.3 / OmegaConf >=2.3。
公共框架使用较新的 PyTorch FSDP/DCP API，不承诺任意旧版 torch 兼容。
本入口不需要 pytest、Wolfram、AIAmplitudes_common_public、绘图库或新模型权重。
不要为此在 Mac 新建/激活 `.venv`；也不要未经确认升级服务器依赖。

代码仓库为 `https://github.com/xghf233/hexagon_w8_MHV_1`。
本地目录已更名，代码交接保留 `hexagon_w8_MHV/nanoinfra-main_symbol/` 层级。
服务器克隆时可显式将目标目录命名为 `hexagon_w8_MHV`；若沿用仓库名
`hexagon_w8_MHV_1`，相应替换下面命令的 checkout 前缀。数据单独传：

| 必须同目录保留的源文件 | SHA-256 |
| --- | --- |
| `four_loop_mhv_symbol_native_x32.jsonl.gz`（879,345 bytes） | `363bc50186af77f7a8b2f726964ff305ab2522b576a8a47f2b9811419dd69de2` |
| `scaling_manifest.json` | `fdd7c25ec846fd32426055dc26500536c86d4fdbd9c5aff9c5db9bb0c358a00a` |

不传旧 Heptagon 数据、训练产物、`.git`、`.venv` 或凭据，不重新下载论文数据。
训练数组和运行目录必须在本代码目录之外，彼此分开；输出目录必须尚不存在。
保留原始 symbol 和提取工具，转换器不会修改它们。

## 2. 唯一首轮入口（全部路径待替换）

用户确认服务器与执行范围后，在 Linux、单张支持 bf16 的 CUDA GPU 上运行：

```bash
cd /ABSOLUTE/SERVER/hexagon_w8_MHV/nanoinfra-main_symbol
CUDA_VISIBLE_DEVICES=0 python -B -m projects.hexagon_symbol_w8_0y.gpu_smoke \
  --source /ABSOLUTE/SERVER/Symbol_Data/hexagon_mhv_4loop_native_x32/four_loop_mhv_symbol_native_x32.jsonl.gz \
  --data-dir /ABSOLUTE/SERVER/datasets/hexagon/w8_0y/random_v1 \
  --output /ABSOLUTE/SERVER/runs/hexagon/w8_0y/gpu_smoke_001
```

使用已发布且未改动的训练数组重跑时，换一个新输出目录；省略 `--source`，
改传 `--metadata-sha256 <可信的64位小写metadata摘要>`。
首次摘要保存在 `environment.json` 和最终报告中；复验时从先前可信记录取得，
不要通过修改摘要绕过错误。

不运行 CPU pytest/CPU 模型前向，也不使用 `torchrun`。Linux/CUDA/bf16 门禁先于转换和运行产物写入。
文件读取、gzip 解压、NumPy 数组整理、格式检查、日志和 RNG 序列化仍由主机完成，
这是 GPU 工作流必需的数据处理，不是独立 CPU 模型核验。

## 3. 入口实际执行的工作

1. 检查原 gzip 与 manifest 的固定哈希，筛选 0y；保存紧凑整数数组和固定 random-row split。
   完整读回，并重新读取原始源文件逐行对照。审计 59 个系数、稀有数字 token、S3 轨道重叠。
2. 在 CUDA 上核对全部 11,208 行编码的符号/高低块/EOS、shift 后监督位置，以及参数量和 logits。
   全表检查仅检查格式，不把 val/test 用于梯度更新或模型评估；诊断前向/反向只用 train。
3. 检查 `[1,9)` 无答案泄漏，并用错误 `[1,11)` 做负对照；比较缓存/无缓存生成。
   诊断模型显式使用非零残差投影以让泄漏测试有效，随后丢弃，不用于训练。
4. 重新从头初始化：连续训练 20 步，对照另一条相同 horizon 的 10+10 步恢复路径。
   一共 **40 次优化更新**，batch=8、compile=false；另有一次诊断反向但不更新权重。
5. CUDA 比较末尾参数（rtol=1e-5、atol=1e-6）、恢复游标、RNG 和固定 val 子集生成指标。
   任何差异超过标准即失败，不自动放宽容差或继续正式训练。

smoke 的 val 子集为 8 行，train 诊断 8 行，不选择 full-val best，不评估最终 test，
20 步也不要求准确率达到指定值。格式检查和恢复对照不能替代后续收敛实验。
CUDA 不可用、数据错误、依赖不兼容或任一步失败时直接退出，保留已写文件以便诊断。

## 4. 产物与验收

数据目录：`words.npy`、`coefficients.npy`、`source_row_ids.npy`、`orbit_ids.npy`、
`splits.npz`、`metadata.json`、`audit.json`。orbit IDs 只用于审计，**不参与切分**。

运行目录：

```text
environment.json                  本次实际环境、代码哈希、可信 metadata 摘要
cuda_contract_checks.json          仅 CUDA 检查通过后生成
continuous/                       连续20步的日志、评估、checkpoint
prefix/                           同一20步horizon的前10步
resumed/                          恢复后到第20步
gpu_smoke_report.json              所有阶段通过才生成 status=passed_gpu_smoke
```

仅当命令退出 0 且最终报告存在，才能记录“GPU smoke 已通过”。数据 `audit.status=passed`
只代表数据发布检查通过，不代表模型通过。正式训练、tiny-overfit、compiled smoke、test
在本次入口中均不自动执行。

资源：尚未测量本任务耗时/显存，不承诺分钟完成。冒烟暂建议预留至少 2 GiB 可用磁盘
存放多个 DCP checkpoint，实际占用由服务器报告确认；不自动删除旧输出。
正式候选配方已写为 batch=128、35,024 步（约500次 train 呈现遍历），但不在这里启动。

## 5. 后续配置（准备好，不自动运行）

- `train --config-name smoke`：独立20步小运行；数据目录、metadata 哈希、输出目录必填。
- `train --config-name tiny_overfit`：32条 train、最多1000步；全部自由生成 exact=1 才通过。
- `train --config-name w8_0y_random`：正式候选；需要单独确认预算与优化器。
- `eval_checkpoint`：默认 val；最终 test 必须另行冻结模型并显式 `--allow-test`。

恢复必须保持数据、代码、PyTorch/NumPy、base100 fixed2、模型、优化器和 horizon 相同。
若准备只跑正式配方前5000步，应保持 `max_steps=35024`，使用 `stop_after_steps=5000`；
不能把 horizon 改5000后又要求无变化地继续。代码变化后先处理版本差异，不直接绕过检查。
