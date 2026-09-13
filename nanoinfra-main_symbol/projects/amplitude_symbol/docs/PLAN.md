# PLAN.md — 五圈散射振幅 Symbol 系数预测 Baseline

## 目录

1. [科学问题](#1-科学问题)
2. [D3 使用边界](#2-d3-使用边界)
3. [数据溯源与许可证](#3-数据溯源与许可证)
4. [数据解析与 Orbit-Grouped Split](#4-数据解析与-orbit-grouped-split)
5. [Token Vocabulary 与 Base-1000 编码](#5-token-vocabulary-与-base-1000-编码)
6. [序列格式与 Loss Mask](#6-序列格式与-loss-mask)
7. [模型配置](#7-模型配置)
8. [评估指标](#8-评估指标)
9. [里程碑与验收标准](#9-里程碑与验收标准)
10. [GPU 依赖划分](#10-gpu-依赖划分)
11. [预计文件结构](#11-预计文件结构)
12. [预计计算预算](#12-预计计算预算)
13. [已知风险与防泄漏断言](#13-已知风险与防泄漏断言)

---

## 1. 科学问题

> 在训练集和测试集不存在 D3 对称轨道重叠的情况下，一个没有显式加入 D3 约束的 decoder-only Transformer，能否从其他轨道中学到可泛化的 D3 对称规律？

关键点：
- **轨道隔离**：train/val/test 之间不能有彼此相关的 D3 轨道成员
- **无显式约束**：架构不内置 D3 不变性，loss 不监督 D3 一致性
- **泛化测试**：在未见轨道上评估 D3 指标

---

## 2. D3 使用边界

### 允许用途

| 用途 | 说明 |
|------|------|
| 计算 orbit_id | `orbit_id = min(get_dihedral_images(word))` |
| 轨道级数据切分 | 保证 train/val/test 无轨道重叠 |
| 训练后诊断指标 | D3 orbit consistency rate、D3 relation accuracy |

### 禁止用途（Baseline 中）

| 用途 | 说明 |
|------|------|
| D3 data augmentation | 不能通过旋转/翻转扩充训练数据 |
| Canonicalization | 不能强制 word 到某种规范形式作为输入 |
| Symmetry consistency loss | 不能添加 D3 一致性作为训练目标 |
| Equivariant architecture | 不能使用群等变网络层 |
| D3 关系作为额外标签 | 不能把轨道关系作为辅助训练信号 |
| 推理时轨道平均 | 不能手工复制或平均轨道成员的预测 |

后续可作为**独立消融实验**添加，不与 baseline 混淆。

---

## 3. 数据溯源与许可证

| 资源 | 来源 | 许可证 | 说明 |
|------|------|--------|------|
| NanoInfra | 本地仓库 | MIT | 训练框架 |
| AIAmplitudes_common_public | `https://github.com/AIAmplitudes/AIAmplitudes_common_public` | Apache-2.0 | 提供 `convert()` 和 `get_dihedral_images()` |
| EZ_symb_new_norm | `https://zenodo.org/records/11218272` | CC BY 4.0 | 五圈 Symbol 数据 |

**记录（阶段 1 已确认）**：

```
AIAmplitudes commit: e39daf0b1c19194da6911f9fb68c5504f321dcde
EZ_symb_new_norm size: 7,704,854 bytes
EZ_symb_new_norm MD5:  71bb1949a160a7501c4b942fc329b64f
五圈非零项预期数量:    263,880
```

**许可证分离**：代码（MIT / Apache-2.0）和数据（CC BY 4.0）必须分开标注，不能把数据声明为 MIT。

---

## 4. 数据解析与 Orbit-Grouped Split

### 4.1 解析

使用 AIAmplitudes 的 `convert()` 函数：

```python
from aiamplitudes_common_public.file_readers import convert
# 或直接将 convert() 的纯 Python 实现复制到项目中以避免完整依赖

symb = convert("data/aiamplitudes/raw/EZ_symb_new_norm", loop=5)
# 返回: {"abcd...": 12334, "badc...": -5678, ...}
```

`convert()` 的核心逻辑：
1. `readFile(f, "Esymb[5]")` — 从文件中读取 `Esymb[5]` 块
2. 正则拆分 `SB(...)` 或类似语法，提取 word 和系数
3. 返回 `{key_string: int_coefficient}` dict

**不需要** `download_all()`、`relpath` 或 GitHub token。

### 4.2 D3 轨道计算

```python
from aiamplitudes_common_public.rels_utils import alphabet, get_dihedral_images

def get_orbit_id(word: str) -> str:
    return min(get_dihedral_images(word))
```

`get_dihedral_images` 是纯字符串函数，不依赖外部文件。也可以直接复制实现。

### 4.3 轨道级数据切分

```
所有样本按 orbit_id 分组
→ 随机选取 N_test 个完整轨道 → test set
→ 剩余中随机选取 N_val 个完整轨道 → val set
→ 其余 → train set
```

**断言（写入 split manifest 构建代码）**：

```python
train_orbits = set(oracle.orbit_ids[train_indices])
val_orbits   = set(oracle.orbit_ids[val_indices])
test_orbits  = set(oracle.orbit_ids[test_indices])

assert train_orbits & val_orbits == set(), "train/val orbit leak!"
assert train_orbits & test_orbits == set(), "train/test orbit leak!"
assert val_orbits & test_orbits == set(), "val/test orbit leak!"
```

### 4.4 对照 Split

同时保存一个**论文式 random-row split**（忽略 orbit，对样本随机切分），用作对照但**不作为主要泛化结论**的来源。

### 4.5 Split Manifest

每次切分固定随机种子（如 `seed=42`），并保存：

```json
{
  "seed": 42,
  "split_type": "orbit",
  "data_file": "data/aiamplitudes/raw/EZ_symb_new_norm",
  "data_md5": "71bb1949a160a7501c4b942fc329b64f",
  "aiamplitudes_commit": "e39daf0b1c19194da6911f9fb68c5504f321dcde",
  "loop_order": 5,
  "total_samples": 263880,
  "train": { "n_samples": ..., "n_orbits": ... },
  "val":   { "n_samples": ..., "n_orbits": ... },
  "test":  { "n_samples": ..., "n_orbits": ... }
}
```

---

## 5. Token Vocabulary 与 Base-1000 编码

### 5.1 Token Types（3 个模态带）

按 NanoInfra 的 `VocabLayout` 三带设计：

| Type ID | 名称 | 内容 |
|---------|------|------|
| 0 | `word` | 6 个字母 token：`a`, `b`, `c`, `d`, `e`, `f` |
| 1 | `control` | 特殊控制 token：`BOS`, `COEFF`, `EOS`, `PAD`, `PLUS`, `MINUS` |
| 2 | `coeff` | 1000 个数字 token：`NUM_0` … `NUM_999` |

### 5.2 Base-1000 系数编码

与论文一致的 base-1000 整数编码：

```
系数  →  编码
 12334 → [PLUS,  NUM_12, NUM_334]
-12334 → [MINUS, NUM_12, NUM_334]
     0 → [PLUS,  NUM_0]
```

编码规则：
- 第一个 token 是符号（PLUS 或 MINUS）
- 后续是表示绝对值的 base-1000 块
- 不足 3 位的块（最高位块）不需要补零
- 0 编码为 `[PLUS, NUM_0]`

### 5.3 Token ID 分配

```
word band:     offset=0,  size=6
control band:  offset=6,  size=6
coeff band:    offset=12, size=1000
total vocab:   1012
```

Control tokens 具体 id：
```
BOS=6, COEFF=7, EOS=8, PAD=9, PLUS=10, MINUS=11
```

`VocabLayout.IGNORE_INDEX` 用于无监督位置的 targets。

### 5.4 Tokenizer 实现

在 `projects/amplitude_symbol/tokenizer.py` 中实现 `SymbolTokenizer`：
- `encode_word(word: str) -> List[int]`：10 个 letter → token ids
- `encode_coefficient(coeff: int) -> List[int]`：int → sign + base-1000 tokens
- `decode(tokens: List[int]) -> str`：仅用于调试
- 总 vocab_size = 1012

---

## 6. 序列格式与 Loss Mask

### 6.1 完整序列

```
[BOS] [w1] [w2] ... [w10] [COEFF] [sign] [block1] ... [blockN] [EOS] [PAD...]
```

| 位置 | 内容 | Type ID | Loss Mask |
|------|------|---------|-----------|
| 0 | BOS | 1 (control) | 0（无监督） |
| 1–10 | 10 个 word 字母 | 0 (word) | 0（无监督） |
| 11 | COEFF 分隔符 | 1 (control) | 0（无监督） |
| 12 | sign (PLUS/MINUS) | 1 (control) | 1（有监督） |
| 13…13+N-1 | base-1000 块 | 2 (coeff) | 1（有监督） |
| 13+N | EOS | 1 (control) | 1（有监督） |
| 剩余 | PAD | 1 (control) | 0（无监督） |

### 6.2 变长处理

不同系数的 base-1000 块数量不同。统一 pad 到固定 `sequence_len`。在 batch 中，PAD 位置的 targets 设为 `IGNORE_INDEX`。

### 6.3 与 NanoInfra 的集成

使用 **AlignedSupervision**（无 shift）+ **loss_weights**：

- `tokens`: 完整序列，长度 `sequence_len`
- `loss_weights`: 同长度，0 表示无监督，1 表示有监督
- `targets`: 与 tokens 相同（aligned），但无监督位置替换为 `IGNORE_INDEX`

仿照 `train_t2m.py` 的 `SourceLoader` 模式：

```python
# 在 __next__ 中：
tokens = ...  # [B, L]
loss_weights = ...  # [B, L], 0 for input positions, 1 for output
targets = torch.where(
    loss_weights[:, 1:] > 0,
    tokens[:, 1:],
    torch.full_like(tokens[:, 1:], VocabLayout.IGNORE_INDEX)
)
# 注意：这里用 shift-by-1，遵循 NextTokenPrediction 约定
# 或者直接用 AlignedSupervision 无 shift
```

**最终决定**：使用 `NextTokenPrediction`（shift-by-1），因为：
- 标准因果 LM 训练方式
- position i 预测 token i+1
- loss_weights 在 shifted targets 上控制哪些位置有监督

---

## 7. 模型配置

### 7.1 Baseline 配置

```python
gpt_config = GPTConfig(
    sequence_len=32,          # 足够容纳: 1 BOS + 10 letters + 1 COEFF + sign + blocks + EOS + padding
    vocab_size=1012,          # 6 word + 6 control + 1000 coeff
    n_layer=4,                # 第一版 baseline 从小模型开始
    n_embd=512,
    n_head=8,
    n_kv_head=8,              # 无 GQA（n_head == n_kv_head）
    n_token_types=3,          # word, control, coeff
)
```

### 7.2 训练配置

```python
config = {
    "max_steps": -1,          # 自动计算（Chinchilla ratio=20）
    "sequence_len": 32,
    "device_batch_size": 256, # 初步估计，根据 GPU 内存调整
    "total_batch_size": 32768,# 初步估计
    "optimizer": {
        "lr_max": 3e-4,
        "max_grad_norm": 1.0,
        "scheduler": {"type": "linear", "warmup_steps": 500, "warmdown_ratio": 0.2},
    },
}
```

### 7.3 单卡训练

- 使用 `bf16` 混合精度（`torch.amp.autocast`）
- 不需要 DDP/FSDP
- 训练脚本直接运行，不通过 `torchrun`

### 7.4 集成方式

完全仿照 `train_t2m.py` 的组装模式：
1. 自定义 DataSource（yield samples）
2. SourceLoader 封装（batch + targets masking）
3. `build_system(GPT, gpt_config)` 组装 trunk + head
4. `Trainer(system, optimizers, loader, config, ...)` 执行训练
5. 不需要子类化 Trainer

---

## 8. 评估指标

所有指标在 **未见过的 test orbits** 上计算。

### 8.1 基础预测指标

| 指标 | 说明 |
|------|------|
| Exact coefficient accuracy | 预测系数 == 真实系数（完全匹配） |
| Magnitude accuracy | `abs(pred) == abs(true)` |
| Sign accuracy | `sign(pred) == sign(true)` |
| Invalid output rate | 输出的 token 序列无法解码为合法系数 |

### 8.2 D3 对称指标

| 指标 | 说明 |
|------|------|
| D3 orbit consistency rate | 同一 orbit 内各 word 的预测系数是否全部一致（模型可能全部猜错但彼此一致） |
| D3 relation accuracy | 对于给定的 relation（如 triple adjacency），模型预测是否满足 relation 方程的百分比 |
| Per-orbit all-correct rate | 每个 orbit 的**所有**成员都预测正确的比例 |

### 8.3 训练效率指标

| 指标 | 说明 |
|------|------|
| Validation cross-entropy | 仅在监督 token 上计算 |
| Training time to reach target accuracy | 达到指定准确率所需时间（分钟）和 step |
| MFU | 模型 FLOPs 利用率 |

### 8.4 重要区分

> **区分 1**：预测是否等于真实值 ≠ 同一 orbit 内预测是否彼此一致。
>
> **区分 2**：Orbit consistency（同一轨道内预测一致）≠ D3 relation satisfaction（满足线性关系方程）。
>
> 模型可能：全部预测错误但轨道内一致；或每个轨道都对但不满足某条 relation。

### 8.5 评估器实现

为每类指标实现独立的 `Evaluator` 子类（仿 `SupervisedCEEvaluator`）：

```python
class CoefficientAccuracyEvaluator(Evaluator):
    """Exact / magnitude / sign accuracy on test set."""

class D3ConsistencyEvaluator(Evaluator):
    """Orbit consistency and relation accuracy on test orbits."""

class SupervisedCEEvaluator(Evaluator):
    """Mean CE over supervised tokens on val set."""
```

---

## 9. 里程碑与验收标准

### 里程碑 0：数据准备和 Tokenizer（无 GPU）

**目标**：可重建的数据处理 pipeline

**验收条件**：
- [ ] `convert()` 能正确解析 EZ_symb_new_norm，返回 ~263,880 个非零项
- [ ] 每个 word 长度 = 10，字母 ∈ {a,b,c,d,e,f}
- [ ] `get_dihedral_images()` 对任意 word 返回 6 个不同的 dihedral images
- [ ] orbit_id = min(images) 正确
- [ ] Tokenizer encode/decode 正确往返
- [ ] 轨道级 split 通过三个断言（train/val/test 无 orbit 重叠）
- [ ] 训练集 ~200K samples，验证集 ~30K，测试集 ~30K
- [ ] Split manifest 已保存，包含所有元数据
- [ ] Random-row split 已保存作为对照

### 里程碑 1：Tiny Overfit（需要 GPU）

**目标**：证明 pipeline 完整且模型可学习，但只做极小的过拟合测试

**配置**：取 64 个样本、用极小模型训练到严重过拟合。

**验收条件**：
- [ ] 64 样本训练 loss 趋近于 0
- [ ] 64 样本 exact accuracy > 0.95
- [ ] 输出合法率 100%（无 invalid tokens）
- [ ] 无训练崩溃（NaN loss, OOM 等）
- [ ] 检查点正常保存和恢复

### 里程碑 2：Smoke Run（需要 GPU）

**目标**：用真实 split 做短时间训练，确认没有静默泄漏或 pipeline bug

**配置**：完整 train split，训练 ~500-1000 steps。

**验收条件**：
- [ ] 训练 loss 在稳态下降
- [ ] val CE 同步下降（未过拟合）
- [ ] train/val/test orbit 重叠断言持续保持
- [ ] 随机 baseline（多数类）被超越
- [ ] 所有评估指标正常产出数值（即使很低）

### 里程碑 3：单次正式 Training Run（需要 GPU）

**目标**：获得可用来回答核心科学问题的 baseline 结果

**配置**：完整 train split，n_layer=4, n_embd=512，训练到收敛或 Chinchilla optimal。

**验收条件**：
- [ ] 所有指标在 test orbits 上合理
- [ ] D3 consistency rate 明显超过随机 baseline
- [ ] D3 relation accuracy 明显超过随机
- [ ] 区分「轨道内一致」和「预测正确」—— 两个维度的独立诊断
- [ ] 报告训练时间、MFU、最终 val CE
- [ ] 所有结果保存在 outputs 中，可从 manifest 完全重建

---

## 10. GPU 依赖划分

### 无卡模式下可完成

| 任务 | 说明 |
|------|------|
| 数据解析与校验 | 纯 CPU |
| D3 orbit 审计 | 纯 CPU |
| Tokenizer 实现与单测 | 纯 CPU |
| Split 生成与断言 | 纯 CPU |
| 代码实现 | 纯 CPU（编辑 .py） |
| PLAN.md 撰写 | 纯 CPU |

### GPU 挂载后才能执行

| 任务 | 说明 |
|------|------|
| Tiny overfit 测试 | 需要 CUDA |
| Smoke run | 需要 CUDA |
| 正式训练 | 需要 CUDA |
| 推理和评估 | 推断时可用 CPU，但建议 GPU |

**当前状态**：RTX 5090 已挂载可用（31.4 GB），所有 GPU 任务可以执行。

---

## 11. 预计文件结构

```
projects/amplitude_symbol/
├── PLAN.md                      # 本文件
├── MANIFEST.json                # Split 元数据（阶段 3 生成）
├── __init__.py
├── tokenizer.py                 # SymbolTokenizer（encode/decode word 和 coeff）
├── data_parser.py               # 对 AIAmplitudes convert() 的薄封装
├── orbit_utils.py               # D3 orbit 计算（引用 AIAmplitudes rels_utils）
├── split.py                     # Orbit-grouped split + random-row split
├── dataset.py                   # DataSource 实现
├── config.py                    # 配置常量
├── evaluate/
│   ├── __init__.py
│   ├── accuracy.py              # CoefficientAccuracyEvaluator
│   ├── d3_metrics.py            # D3ConsistencyEvaluator
│   └── ce_eval.py               # SupervisedCEEvaluator
├── train.py                     # 训练入口（仿 train_t2m.py）
└── generate.py                  # 推理和指标产出脚本
```

**参考但不修改的代码**：
- `upstream/AIAmplitudes_common_public/` — 提供 convert() 和 get_dihedral_images()
- `core/` — 不修改

**数据和产物（不提交 Git）**：
- `data/aiamplitudes/raw/EZ_symb_new_norm`
- `models/amplitude_symbol/`
- `outputs/amplitude_symbol/`

---

## 12. 预计计算预算

| 里程碑 | 预计时间 | FLOPs 估算 |
|--------|---------|-----------|
| Tiny overfit | < 5 分钟 | 极小 |
| Smoke run | < 30 分钟 | ~10¹⁵ FLOPs |
| 正式训练（Chinchilla optimal） | 1–3 小时 | ~10¹⁷ FLOPs |

模型大小：
- n_layer=4, n_embd=512 → ~10 M params
- Chinchilla optimal tokens = 20 × 10M = 200M tokens
- total_batch_size = 32,768 tokens/step → ~6,100 steps

RTX 5090 bf16 dense ≈ 209.5 TFLOPS，10M 参数模型预计可接近较高 MFU。

---

## 13. 已知风险与防泄漏断言

### 13.1 防止数据泄漏

```python
# 在 split.py 中强制断言
train_orbits = set(oracle.orbit_ids[train_indices])
val_orbits   = set(oracle.orbit_ids[val_indices])
test_orbits  = set(oracle.orbit_ids[test_indices])

assert train_orbits.isdisjoint(val_orbits),  "Orbit leak: train ∩ val"
assert train_orbits.isdisjoint(test_orbits), "Orbit leak: train ∩ test"
assert val_orbits.isdisjoint(test_orbits),   "Orbit leak: val ∩ test"
```

### 13.2 已知风险

| 风险 | 缓解措施 |
|------|---------|
| 系数范围可能很大（超出 base-1000 块数预算） | 在解析阶段统计系数分布，确定 `sequence_len` 上限 |
| 某些 orbit 可能只有 1 个非零样本 | 轨道级切分可能导致 split 不平衡 — 先统计后切分 |
| 模型可能学会靠 word embedding 做最近邻而不是 D3 泛化 | 通过 test orbit 上的一致性指标区分这两种行为 |
| AIAmplitudes 完整依赖很多（torch、wandb 等） | 只复制需要的纯函数（convert、get_dihedral_images），不 pip install 完整包 |
| 序列格式变长 | 设置充足的 sequence_len 并用 PAD 填充 |
| 训练时全部预测同一值 | Evaluator 检查预测分布，报告 per-class accuracy |

### 13.3 不允许的操作

- 不修改 `core/`
- 不修改 `upstream/AIAmplitudes_common_public/`
- 不在 baseline 中使用 D3 augmentation/canonicalization/symmetry loss
- 不在推理时手工复制或平均轨道预测
- 不 push 到远程仓库
