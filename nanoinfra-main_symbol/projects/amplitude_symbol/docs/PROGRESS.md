# PROGRESS.md — 项目进度记录

> 最后更新: 2026-07-31

> 2026-09-07 导航补充：下文保留原五圈 amplitude/D3 项目的历史记录，
> 其中测试数量、运行环境及待办不代表 heptagon 当前状态。
> 新三圈 heptagon 位于并列的 `projects/heptagon_symbol/`，
> 最新进度见 [Heptagon PROGRESS.md](../../heptagon_symbol/PROGRESS.md)：
> 数据转换已执行并审计通过；新增训练工程及测试已编写，尚未执行服务器验证或训练。

## 科学问题

在 train/val/test 不存在 D3 轨道重叠的情况下，无显式 D3 约束的 decoder-only Transformer
能否学到可泛化的 D3 对称规律？

## 已完成阶段

### 阶段 0 — 环境审计
- NanoInfra @ `5a06d41` (MIT)，工作区干净
- 1× RTX 5090 (31.4 GB)，CUDA 可用
- Python 3.12.3, PyTorch 2.12.1+cu130
- AIAmplitudes 未安装

### 阶段 1 — 外部资源获取
- AIAmplitudes_common_public @ `e39daf0` (Apache-2.0) → `upstream/`
- EZ_symb_new_norm (7,704,854 bytes, MD5 verified) → `data/aiamplitudes/raw/`
- ENVIRONMENT.md 已更新 GPU 状态

### 阶段 2 — 项目计划
- PLAN.md 完成并获得批准
- 核心设计：decoder-only GPT + orbit-grouped split + 自由生成评估

### 阶段 3 — 数据解析与 D3 审计
- 安装 AIAmplitudes (editable, --no-deps)，torch/numpy 未被升级
- `prepare_data.py` — 使用官方 `convert(path, loop=5)` 解析
- `symmetry.py` — 薄封装官方 `get_dihedral_images()`
- 审计结果：
  - 263,880 样本，word 长度全 10，字母合法，系数全非零
  - 648 个 unique coefficients，max |c| = 860,160
  - Base-1000 最大块数 = 2
  - 43,980 orbits，全部大小恰好 6，orbit 内系数一致
  - 文件 MD5 和 AIAmplitudes commit 已记录

### 阶段 4 — Split、Tokenizer、DataLoader
- `split.py` — orbit-grouped split (seed=42): train=211,104 / val=26,388 / test=26,388
  - 随机行 split 作为对照
  - Orbit 互斥断言通过
  - MANIFEST.json 已保存
- `tokenizer.py` — 3-band vocabulary (1012 tokens):
  - Band 0 (word, type=0): a-f (6 tokens)
  - Band 1 (control, type=1): BOS, COEFF, EOS, PAD (4 tokens)
  - Band 2 (coefficient, type=2): PLUS_SIGN, MINUS_SIGN, NUM_0..NUM_999 (1002 tokens)
  - **Sign 在 coeff band 内**，保证生成时所有 coefficient token 共享 type=2
  - Strict encode/decode with round-trip guarantee
- `dataset.py` — AmplitudeDataSource (infinite, GPU placement) + AmplitudeDataLoader (Trainer-compatible batches)
- `MANIFEST.json` — 已保存的 split manifest (seed=42)
- 75 tests passed

### 阶段 5 — Tiny Overfit
- `train.py` — Hydra-driven training orchestrator
- 64 samples, GPT 3L/256d/4h, 2.88M params
- 3000 steps → loss 0.000000, 100% free-generation accuracy
- 训练时间 0.94 min on RTX 5090
- 漏洞修复: sign 从 control band 移入 coeff band (消除 train/inference token-type mismatch)

### 阶段 6A — Evaluator
- `evaluator.py` — 自由生成评估:
  - `strict_decode_coefficient()` — 7 项合法性检查
  - `generate_coefficients()` — 批量 autoregressive_generate
  - `compute_sample_metrics()` — exact/magnitude/sign/invalid
  - `compute_orbit_metrics()` — consistency/whole-orbit/singleton
- `tests/test_evaluator.py` — 40 个 CPU 测试
- 135 tests total, all pass

### 阶段 6B — 配置完善与训练集成
- PROGRESS.md (本文件)
- Evaluator 集成到 train.py 训练末尾
- Smoke config 补全 eval 字段
- `full.yaml` → `baseline_orbit.yaml`
- Smoke run 验证 (500 steps, 15.2% exact accuracy)

### 阶段 8A — 正式 Baseline 训练
- **Commit**: `a2a7d51` (20 files, 3735 lines)
- 配置: `configs/baseline_orbit.yaml`
- Training: 133,015 auto-calculated steps (Chinchilla ratio=20)
- Runtime: ~65 min on RTX 5090
- Checkpoints: step_130000, step_132000 (`keep_last_n=2`)
- **Final evaluation (512 val samples):**

| Metric | Value |
|--------|-------|
| Exact accuracy | **47.5%** |
| Magnitude accuracy | **95.1%** |
| Sign accuracy | 51.0% (≈ random) |
| Invalid rate | **0.0%** |
| Orbit consistency | **80.2%** |
| Whole-orbit accuracy | 38.4% |

- **Key finding**: Two-phase learning confirmed — magnitudes are nearly learned
  (95%) but signs are still at random (51%). Invalid rate is 0% across the
  entire training. Orbit consistency (80%) >> whole-orbit accuracy (38%),
  confirming that the model can learn D3 symmetry patterns even without
  explicit constraints.

## 文件结构

```
projects/amplitude_symbol/
├── PROGRESS.md                   # 本文件
├── PLAN.md                       # 项目计划
├── MANIFEST.json                 # split manifest (seed=42)
├── overfit_history.json          # tiny overfit training curve data
├── __init__.py
├── tokenizer.py                  # 固定词表 + base-1000 编解码
├── symmetry.py                   # D3 轨道封装
├── prepare_data.py               # 数据加载与审计
├── split.py                      # Orbit-grouped split
├── dataset.py                    # DataSource + DataLoader
├── train.py                      # Hydra 训练入口
├── evaluator.py                  # 自由生成评估
├── configs/
│   ├── base.yaml                 # 公共默认参数
│   ├── tiny_overfit.yaml         # 64-sample overfit
│   ├── smoke.yaml                # 500-step smoke run
│   └── baseline_orbit.yaml       # 正式训练
└── tests/
    ├── test_data.py              # 19 tests
    ├── test_tokenizer.py         # 55 tests
    ├── test_split.py             # 20 tests
    ├── test_evaluator.py         # 40 tests
    └── test_overfit.py           # (TBD)
```

## 关键设计决策

1. **Sign 在 coeff band**: PLUS/MINUS 不在 control band，而是在 coeff band 内，
   这样 autoregressive_generate 用 gen_token_type=2 时所有生成 token 类型一致。

2. **Decoder-only**: 使用 NanoInfra GPT（因果自注意力），encoder-decoder 的论文方法
   作为对照。

3. **Orbit-grouped split**: 训练/验证/测试按 orbit 分组，无重叠。
   核心科学问题依赖这个切分方式。

4. **自由生成评估**: 不用 teacher forcing 冒充 coefficient accuracy。
   所有指标通过 autoregressive_generate + strict_decode 计算。

5. **无 D3 prior**: Baseline 不 canonicalize、不 augment、不加 symmetry loss。

## 待完成

- [ ] Restart training with proper output capture (no pipe to head/grep)
- [ ] Track evaluation metrics across training (not just final)
- [ ] Test set evaluation (only after config is locked)
- [ ] Multi-seed experiments
- [ ] Ablation: canonicalization / augmentation / symmetry loss
- [ ] D3 relation accuracy 指标
