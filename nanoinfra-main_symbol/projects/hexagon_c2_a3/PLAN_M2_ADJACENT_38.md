# PLAN — 四圈相邻 y、38 项输入的 M2 小实验

日期：2026-09-14。状态：**已按本计划编写首版候选代码并人工静态核对；尚未运行验证。**
训练数组/切分尚未生成，模型和 GPU smoke 尚未执行。当前交接见 [SERVER_HANDOFF_M2.md](SERVER_HANDOFF_M2.md)，
静态检查范围见 [STATIC_REVIEW_M2.md](STATIC_REVIEW_M2.md)；“已写检查”不等于“检查通过”。

本计划落实用户选择：先做 M2 的相邻 y 子任务，不同时启动完整 word 版本、非相邻 y、零目标或 20M 模型。
已有 `audit_representation.py` 是独立数据审计工具，不是本实验的训练实现。

本次修订：根据相邻子集去掉位置信息后的精确汇总，将主输入由 40 项改为 **38 项**。
两个 y 的位置仅用于构造条件、来源核验和分层评估，不进入模型、查表基线的键或主输入重复分组。
39 项（再给一个位置）/40 项版本只保留为可能的后续对照；本版不同时实现。

## 1. 实验问题与边界

固定普通 BDS-like 四圈 Hexagon MHV、weight=8，使用带帽字母表和已缩放的整数系数 `C4=32*c4`。
目标 word 恰好有两个 y，位置相邻，目标系数非零。

设从左到右的两个 y 为 `y_alpha, y_beta`，其位置为 `r < s`，且 `s=r+1`。
普通字母顺序固定为 `E=(hat_a,hat_b,hat_c,hat_d,hat_e,hat_f)`：

```text
C36[6*i+j] = C4(W 中第 r 个位置替换为 E[i]、第 s 个位置替换为 E[j])
模型输入     = (C36[0:36], y_alpha, y_beta)
模型输出     = C4(W)
```

- 36 个条件均为同圈 0y 精确查表值，保留零、重复值和行优先顺序；不使用旧 0y 模型的预测。
- 目标 word 的其余六个字母不进入模型；完整 word 只用于来源追踪、核验和诊断。
- 两个位置均保留在数据的审计字段中，但不生成位置 token，不用位置选择输出头或推理路径。
- 圈数固定为 4，不逐样本增加圈数 token；圈数和归一化由 metadata/checkpoint 锁定。
- 两个项目共享 GPT 实现，不共享权重、优化器、RNG、checkpoint 或训练目录。
- 本实验是“已知非零支持集中的相邻 yy 系数补全”，不是零/非零分类或完整 A3 恢复。
- 相邻子任务是在已有全池审计后提出的探索性设定，需如实记录，不能包装成未经数据观察的预注册发现。

## 2. 已有证据与尚未验证的部分

以下统计来自已经完成的精确审计及其分组产物的只读汇总，不是训练结果：

| y 位置（从 1 开始） | 非零目标行数 | 不同 38 项输入数（在该位置内去重） | 冲突输入数 |
| --- | ---: | ---: | ---: |
| (3,4) | 7,200 | 2,421 | 0 |
| (4,5) | 6,684 | 2,517 | 0 |
| (5,6) | 8,184 | 2,439 | 0 |
| (6,7) | 7,788 | 3,021 | 0 |
| (7,8) | 10,920 | 3,060 | 0 |
| 全池联合去重 | **40,776** | **13,338** | **0** |

各位置内部的输入数相加为 13,458，但去掉位置后存在 120 个跨位置输入组，因此全池不同输入数为 13,338，不能直接相加。
这 120 组的目标系数仍一致；已知相邻非零子集的 38 项表示没有输入冲突。

| 相邻子集表示 | 不同输入数 | 冲突输入数 |
| --- | ---: | ---: |
| 旧 40 项（含两个位置，仅供比较） | 13,458 | 0 |
| 当前 38 项（不含位置） | 13,338 | 0 |

- 不同 C36 表共 2,067 张；36 项全零的目标数为 0。
- 目标有 76 种非零整数值，范围 `[-240,216]`；条件绝对值最大为 1,920。
- 13,338 个输入组中，3,570 个只出现一次，其余 9,768 个有重复；最大组有 32 行。
- “未发现冲突”只对这个固定、非零、四圈相邻子集成立，不证明包含零目标或其他圈数时仍充分。
- 本子集 38 项的经验表示上限是 100%；**全体 2y 的 38 项 95.5415% / 40 项 97.2966% 上限都不适用于这个子集**。
- 目前尚未生成 seed=42 的本实验切分，因此 train/val/test 的分布、重复覆盖率、稀有项归属都待实际生成。

证据路径（仅作本地追溯，不能硬编码为服务器路径）：

- `/Users/hzq/hep_th/Building Intelligent Models from Scratch/hexagon_c2_a3_audits/2026-09-14_nonzero_inputs_v1/REPORT.md`
- 同目录 `audit.json` 和 `c36_y_types_positions.groups.jsonl.gz`。

38 项子集统计的复算方法：从上述 40 项分组产物筛选 `s=r+1`，以 `(ordered_coefficients_C4,ordered_y_types)`
重新合并输入组及目标计数，忽略位置字段；检查总行数 40,776、不同键 13,338、冲突数 0。
这是对已有产物的只读汇总，不改动原审计脚本、历史报告或源数据；后续数据发布仍需从原 source 独立复核。

## 3. 工程组织：新增同级任务，不改已训练任务

位置：`nanoinfra-main_symbol/projects/hexagon_c2_a3/`。

复用共享 `core` 的 GPT、LMSystem/LMHead、next-token supervision、优化器、调度器和 DCP 机制；
新建任务级的数据、编码、模型适配、训练和评估组件。旧 `hexagon_symbol_w8_0y` 不作为可直接套用的通用接口。

禁止顺带改动：

- 旧 0y 项目的 Python/config 文件与已有运行；其 checkpoint 会校验相关代码哈希。
- 公共 `core/`、共享 amplitude attention 工具、源 Symbol_Data 和已有提取/缩放脚本。
- HZQ-git、Heptagon 项目及它们的历史计划；本文件是新小实验的执行计划，不隐式同步其他目录。

必要的仓库公共改动仅规划为：给 `pyproject.toml` 增加 `projects.hexagon_c2_a3*` 的包发现和 configs YAML 打包条目，
同步更新入口文档。现已增加这两个打包条目，未改变旧包条目或依赖版本。

## 4. 数据生成与精确来源

### 4.1 唯一真值源

已有 `Symbol_Data/hexagon_mhv_4loop_native_x32/four_loop_mhv_symbol_native_x32.jsonl.gz`：

| 文件 | 固定 SHA-256 |
| --- | --- |
| x32 JSONL gzip | `363bc50186af77f7a8b2f726964ff305ab2522b576a8a47f2b9811419dd69de2` |
| scaling_manifest.json | `fdd7c25ec846fd32426055dc26500536c86d4fdbd9c5aff9c5db9bb0c358a00a` |

每行是 `word`（8 个原子字母）、`numerator`（整数十进制字符串）、`denominator="1"`。
原生 `mu,mv,mw` 分别对应 `hat_d,hat_e,hat_f`；原生 `yu,yv,yw` 对应三个 y。
文件中的分子已经是 C4，不得再次乘 32，不用浮点数解析/比较标签。

### 4.2 生成顺序

1. 核对两个固定哈希、manifest 的对象/圈数/weight/字母表/缩放约定。
2. 完整读取 243,000 行；验证字段、原子字母、严格有序且唯一、非零整数格式、分层数量及完整系数直方图。
3. 从 11,208 个 0y 非零项建立精确字典。只有完成全源检查后，合法 0y word 查无记录才解释为零；解析错误直接失败。
4. 在全部 110,082 个 2y 非零项中按 `s=r+1` 筛选，保留原始 source 顺序，得到 40,776 行。
5. 每个目标替换两个 y，按 `slot=6*i+j` 查出 36 项。可按 family 缓存计算，但结果必须独立逐目标复核。
6. 构造真实 family ID 与完整 38 项输入 ID。精确元组判等，不能仅凭 hash 相同认定输入相同。
7. 验证五组位置计数、13,338 个不同 38 项输入和零冲突。若出现冲突先停止查错，不删除冲突行或改写标签。
8. 按下节规则生成一次固定切分、统计重复与分布，输出紧凑数组、metadata 和 audit；逐项读回核对。

family 键为把两个 y 换成同一占位符 ID=9 的八字母模板；按原生 ID 的字典序对唯一模板分配稳定整数 ID。
input38 键固定为 `(C36_tuple, ordered_y_types)`，也按精确键的字典序分配稳定 ID；**不得包含 y_positions 或完整 word**。
不要用 Python 进程相关的 `hash()` 直接充当持久化 ID。

可以通过新 `oracle.py` 封装现有审计脚本的纯数据 `read_source`，避免重复维护源格式校验；
不改变已经生成报告所对应的审计脚本。若采用此依赖，数据/checkpoint 的代码哈希必须包含它。
已有审计分组文件只作核对参考，不替代原始 x32 文件作为训练真值源。

### 4.3 存储格式

此小实验优先使用直观的逐行数组，而非为 210 万候选设计的多级 family 存储。
`conditions` 的全部整数仅 2,935,872 bytes，逐行保存足够小；不存 36 个完整源 word 或预编码的所有 token。

| 文件 | shape / dtype | 用途 |
| --- | --- | --- |
| conditions.npy | `[40776,36]`, little-endian int16 | 模型的 36 项输入 |
| y_positions.npy | `[40776,2]`, uint8 | 存储零基位置，仅作审计/位置分层；不编码成 token |
| y_types.npy | `[40776,2]`, uint8 | 原生字母 ID 6、7、8；保持从左到右顺序 |
| coefficients.npy | `[40776]`, int16 | 非零 C4 目标标签 |
| words.npy | `[40776,8]`, uint8 | 仅用于来源追踪与诊断，禁止进入 M2 prompt |
| source_row_ids.npy | `[40776]`, int32 | 原 gzip 的零基行号；报告行号为它加 1 |
| family_ids.npy | `[40776]`, int32 | 审计字段，不参与本轮切分或模型输入 |
| input_group_ids.npy | `[40776]`, int32 | 完整 38 项精确输入分组；不包含位置，本轮仅作诊断 |
| splits.npz | train/val/test 的 int32 行索引 | 固定源顺序上的 random-row 分区 |
| metadata.json / audit.json | 版本化 JSON | 来源、约定、分布、检查结果和全部数组哈希 |

所有多字节整数数组使用固定 little-endian dtype；不使用 object array 或需要 pickle 的存储。
模型输入接口只接收 `conditions, y_types`；标签单独传给训练组装器。
y_positions、完整 word、source ID、family ID、input_group ID 不得隐式混入 token、embedding 或 loss 特征。
仍须用 y_positions 在数据准备阶段检查相邻性和从左到右的 y 类型顺序；与模型输入接口分离。

派生数据和训练输出放代码仓库、不可变源目录之外的新目录；不覆盖已有数据或运行。
服务器传入显式 `--source/--data-dir/--output`，不要把 Mac 绝对路径写进默认配置。

### 4.4 数据入口和运行入口如何改变

| 层次 | 原 0y 任务 | 新相邻 y M2 |
| --- | --- | --- |
| 原始 source | 同一完整 x32 gzip | 不变，不重新下载/缩放 |
| 转换筛选 | 0y 非零、11,208 行 | 2y 非零且相邻、40,776 行 |
| Dataset 输入字段 | 八字母 word | conditions + y_types；位置仅附在审计元信息中 |
| 标签 | 0y 的 C4 | 目标相邻 2y 的 C4 |
| 模型 prompt | BOS + word + COEFF | 两个有序 y 查询 + 36 数字 + ANSWER |
| smoke 模块 | projects.hexagon_symbol_w8_0y.gpu_smoke | projects.hexagon_c2_a3.gpu_smoke |
| 训练模块 | projects.hexagon_symbol_w8_0y.train | projects.hexagon_c2_a3.train |
| data_dir / output_dir | 旧 0y 路径 | 新建、独立、明确指定；不自动搜索旧目录 |

新 Dataset 必须先验证自己的 dataset ID、数组 schema 和 metadata 哈希；误传旧 0y 数据目录直接报错。
不能仅因两个目录都有 `words.npy` 和 `coefficients.npy` 就把它们视为兼容。

以下 CLI 已有候选实现，但**未执行验证，须在另行批准的服务器 GPU 阶段使用**：

```bash
cd /SERVER/PATH/hexagon_w8_MHV/nanoinfra-main_symbol
CUDA_VISIBLE_DEVICES=0 python -B -m projects.hexagon_c2_a3.gpu_smoke \
  --source /SERVER/PATH/Symbol_Data/hexagon_mhv_4loop_native_x32/four_loop_mhv_symbol_native_x32.jsonl.gz \
  --data-dir /SERVER/PATH/datasets/hexagon/m2_adjacent38/random_v1 \
  --output /SERVER/PATH/runs/hexagon/m2_adjacent38/smoke_001
```

之后另行授权的 pilot 才使用 `projects.hexagon_c2_a3.train --config-name m2_adjacent_random_pilot`，
并显式传入 `data_dir`、可信 `metadata_sha256`、新 `output_dir`。新配置默认 `resume_from=null`。
依赖沿用项目的 Python>=3.12、PyTorch/NumPy/Hydra/OmegaConf；先读取服务器实际环境，不安装或猜测版本。

## 5. 首轮切分：保留 random-row，公开输入重复

### 5.1 固定算法

- 对按原始 source 顺序排列的 40,776 个目标行，使用 `numpy.random.Generator(PCG64(42)).permutation(N)`。
- train=`floor(0.8*N)=32,620`；val=`floor(0.1*N)=4,077`；test=剩余 `4,079`。
- 按 permutation 依次切片；每个 split 内保存升序数据行索引，训练 loader 再独立洗牌。
- 记录 NumPy 版本、算法、种子和切分文件哈希；不得为稀有项或指标重抽 seed。
- 保留全部原始目标 word，即使它们的压缩输入重复；每个原始目标行等权，不做输入去重、平衡采样、对称增强或 family/input-group 分组切分。
- metadata 明写 `split_type=random_row`，且 `input_grouped/family_grouped/orbit_grouped=false`。

这延续用户希望的“见过背景后补全其他 y 查询”实验，**不是 family-heldout**。
压缩会让不同 word 出现相同完整输入；本轮有意保留该现象，但必须单独度量，不能把总分全解释为新输入泛化。

### 5.2 必须生成的诊断

- val/test 的 `seen_input38_in_train`：完整 38 项键是否在 train 中出现；只用输入判定，不用标签挑选。
- `seen_family_in_train`、每个目标对应的训练 sibling 数；只作诊断，不作为模型特征。
- 每个集合的 y 配对、五组位置、目标符号/大小、高低数字块与条件数字块分布。
- 全局 38 项重复组大小、跨 split 覆盖率；验证没有同一原始目标行/word 跨 split。
- 家族数、不同 C36 表数、不同输入数；不同训练标签数与不同输入数分别报告。

禁止把全体 2y 的约 81.1% 重复覆盖估计或旧 40 项分组覆盖率套到本子集；本实验按实际 38 项键重新计算。
若后续采用 exact-input-grouped split，它是一个新版本实验，不能覆盖或暗改本次 random-row 数据。

## 6. Tokenizer：固定三 token 数字，128 槽就够

### 6.1 序列布局（零基下标）

```text
[BOS][QUERY][Y_alpha][Y_beta][CONTEXT]
  [sign high low] × 36，严格按 slot=0…35 排列
[ANSWER][target_sign][target_high][target_low][EOS]
  [PAD] × 10
```

| 下标 | 内容 | 是否监督 |
| --- | --- | --- |
| 0 | BOS | 否 |
| 1 | QUERY | 否 |
| 2,3 | 两个有序 y 类型 | 否 |
| 4 | CONTEXT | 否 |
| 5…112 | 36×3 个条件数字 token | 否 |
| 113 | ANSWER | 本位置 logits 预测目标符号 |
| 114…117 | 目标符号、高块、低块、EOS | 原始序列 loss 权重为 1 |
| 118…127 | PAD | 否 |

明确常量：`prompt_length=114`、`answer_length=4`、`sequence_len=128`，无 PAD 的完整样本长 118。
这里的 38 是逻辑输入量数量，不是 token 数；输入内容为 110 tokens，控制标记使 prefix 成为 114 tokens。
与旧 40 项计划相比只减少两个内容 token；固定 padding 到 128 的训练计算量基本不变，不把此次修改宣传成显著提速。

每个数字固定为 `sign + NUM_(abs(C)//100) + NUM_(abs(C)%100)`；允许条件零，统一编码 `+ 00 00`，禁止负零。
高块为零也不能省略；NUM_0 不是 PAD。保留完整 NUM_0…NUM_99，而不是用全池目标直方图制作输出值白名单。

**首版不增加 36 个 SLOT token。** 每个数字固定占 3 槽，slot 的语义由固定偏移和 RoPE 位置确定；
不排序、不去重、不跳过零，因此边界和语义仍明确。无须重复 36 个源 word。
这把此前为含槽位标记版本预留的 192 槽收紧为 128 槽，不改变 36 项物理数据。
如果以后比较显式槽位标记，另立 tokenizer 版本；不得在同一 checkpoint 中更换格式。
不输入 y 的绝对位置，不等于去掉序列位置编码：RoPE 仍保留，用于区分两个 y 查询和 36 个有序系数槽。

### 6.2 新词表与 token type

| token ID | 意义 | type |
| --- | --- | --- |
| 0…8 | 原生九字母，顺序 a,b,c,mu,mv,mw,yu,yv,yw | 0 |
| 9…14 | BOS,QUERY,CONTEXT,ANSWER,EOS,PAD | 1 |
| 15,16 | PLUS,MINUS | 2 |
| 17…116 | NUM_0…NUM_99 | 2 |

`vocab_size=117`，`n_token_types=3`。普通六字母 ID 保留在词表中，但 M2 prompt 不得出现它们。
删除原计划的全部 POS_1…POS_8 token；合法 y 位置仍在数据审计中验证，而非 token 化。
VocabLayout 必须使用三个连续区间 `[0,9),[9,15),[15,117)`，不能为同一个 type 重复添加不连续区间。
不要沿用旧 0y 的 vocab=115、token ID、prompt=10 或 sequence=16。

### 6.3 监督与推理

- 只实施一次 next-token shift：`x=tokens[:-1]`、`target=tokens[1:]`。
- shift 后有效 loss 下标为 `[113,117)`，仅四个目标 token；损失按这些有效目标 token 归一化。
- 不监督 36 个已知条件、查询信息或 PAD。
- `encode_prompt(conditions, y_types)` 不接收位置、目标系数或完整 word；推理只给 114-token prefix。
- 首版 greedy、最多四个生成 token，使用完整词表，不用目标白名单或事后改答案。
- 严格解析符号/两个数位/EOS；提前 EOS、缺 EOS、错误 token、负零等算 invalid。
- 合法生成 0 算数值错误但不算 invalid，因为目标池非零，而语法本身支持 0。

## 7. 模型接入：共享 GPT，不共享旧模型权重

| 参数 | 本实验首版 |
| --- | --- |
| GPT 层数/宽度/heads/KV heads | 4/256/4/4 |
| head_dim / MLP expansion | 64 / 4 |
| 位置编码/MLP/归一化 | 保留现有 RoPE / ReLU² / 无参数 RMSNorm |
| sequence_len / vocab / types | 128 / 117 / 3 |
| 估计参数量 | **3,206,400**，按现有 untied LMHead 计算，GPU 构建后必须实测断言 |
| 精度/设备 | Linux 单张 bf16 CUDA GPU；无 CPU/MPS 模型回退 |
| compile / head_softcap | false / 15.0 |
| 权重与状态 | 全部新初始化；只允许恢复同一新任务的兼容 checkpoint |

注意力为 **prefix-LM**：

- 已知 prefix `[0,114)` 内完全双向；不能看见序列下标 114 及之后的答案。
- 答案部分自回归；所有真实查询均不能读取右侧 PAD。
- 训练虽包含右侧 PAD，其 loss 为零；mask 必须保证有效位置不能借由 PAD 泄露未来信息，且 PAD 查询不能产生全屏蔽 NaN。
- 用共享 mask factory 的显式边界 `start=0,end=114` 生成矩阵，并通过现有 `_word_bidi_mask` 挂载点安装。
- 新任务的 `attention_config` 独立记录 `mode=prefix_lm,prefix_length=114,sequence_len=128`。
  可调用现有工具生成 mask，但不能把内部历史名称 `word_bidirectional` 当成新实验的科学定义。
- 所有启动、保存、恢复和评估都重新安装并验证 runtime mask；它不等于 tensor checkpoint 自动恢复的参数。
- 第一次推理整段 prefill 114 tokens，后续逐 token KV-cache decode；不先实现分块 prefix prefill 或 compiled static-cache 路径。

模型与 runner 都独立于旧 0y 硬编码适配器。首版没有必要改公共 GPT、增加模型到 20M 或引入新的 attention 架构。

## 8. 训练配置与预算

### 8.1 分阶段配置

下表为未来配置候选，不是本轮运行授权。默认入口仍是 GPU smoke。

| 配置 | batch（样本/更新） | nominal total_batch_size | 更新步数 | 用途 |
| --- | ---: | ---: | ---: | --- |
| smoke | 8 | 1,024 | 20 | 连续 20 与另一路 10+10 恢复对照，总共 40 次更新 |
| tiny_overfit | 32 | 4,096 | 最多 1,000 | train 中 32 个不同输入，检查自由生成能否全部正确 |
| m2_adjacent_random_pilot | 64 | 8,192 | **5,000** | 首轮独立、完整调度的学习 pilot |

首版单 GPU、gradient accumulation=1，严格满足 `total_batch_size = batch * sequence_len`。
不能带入旧任务的 2,048。若显存不允许 batch64，先报告并确定一个新配置，不静默 OOM 后缩 batch 或改累积。

5000 步、batch64 共呈现 320,000 条训练目标，约 `320000/32620=9.81` 次平均曝光。
这不是 500 次曝光，也不保证足够收敛；pilot 用于了解曲线，不以未达 99% 自动认定失败或需要扩容。
每个 epoch 采用无放回 shuffle，尾部与下一轮拼接成完整 batch，不丢尾部；持久化下一个未读行游标及 RNG。

### 8.2 优化器

首版沿用已用过的三组 AdamW 配方，保持几何不变；不是把所有参数的 LR 都设成一个值：

| 参数组 | 配置 LR | width=256 经 core 缩放后的峰值 LR |
| --- | ---: | ---: |
| matrix | 0.0002 | 0.0003464102 |
| embedding（含 token/type） | 0.14 | 0.2424871131 |
| unembedding | 0.0028 | 0.0048497423 |

core 的缩放因子是 `sqrt(768/256)=sqrt(3)`；必须在 run.json 中保存实际三组 LR，防止混淆。
其余：AdamW fused=true、betas=(0.9,0.95)、eps=1e-10、weight_decay=0.01、grad clip=1.0。
更长的 prefix 和新 token 分布可能影响优化；若修改 LR，作为明确的新配方记录，不能伪装成原配方续训。

### 8.3 退火和恢复

pilot 使用 `max_steps=5000,warmup_steps=200,warmdown_ratio=0.2,final_lr_frac=0`。
即 200 步 warmup、平台期、末段约 1000 步线性退火；**这是完整 5000 步计划，不是 35024 步计划的前段**。

保留 core 的零基更新语义：更新前用 `completed_steps` 查 LR。
步索引 0 的倍率为 1/200；索引 4000 仍为 1；索引 4999 为 0.001；理论 horizon 索引 5000 才为 0。
所以不能把“final_lr_frac=0”误写成最后一次实际更新的 LR 严格等于零。GPU smoke 需核对调度边界。

smoke 使用同一 20 步 horizon 完成连续/分段两条路径（建议 warmup=2、warmdown=0.2）。
tiny_overfit 单独 horizon=1000（建议 warmup=20、warmdown=0.2），不作为 pilot 的预训练权重。
恢复不允许更改数据、batch、tokenizer、mask、优化器或 horizon。
需要更长正式训练时重新确定预算并新建运行；不得把退火完的 pilot 不加说明地当作更长原计划继续。

## 9. 评估：必须区分查表覆盖和新输入预测

### 9.1 无学习基线

在 train 上构造，禁止使用 val/test 标签：

1. **全局众数**：只从 train 系数选取；并列时固定选数值较小者。
2. **完整 input38 查表**：从 train 的精确 `(C36,y_types)` 输入到系数建表，不含位置；本子集无冲突，重复标签必须一致。
   已覆盖则返回该系数，未覆盖时回退到 train 众数；单独报告覆盖率和覆盖部分准确率。

当前数据若查表已覆盖却不正确，是编码/分组/数据问题，不应解释为统计噪声。
查表基线使用完整 train 池，不是只使用模型当前已遍历的 minibatch；日志注明这点。

### 9.2 指标与选模

- 主表：自由生成的 exact、magnitude、sign、invalid、合法预测零比例；不是 teacher-forced token accuracy。
- 分层：seen_input38 / unseen_input38、五组 y 位置、九种有序 y 类型、稀有系数、seen_family。
- 同时报告行加权成绩及输入组宏平均；输入重复不能被误算成同等数量的独立推断成功。
- 空分层输出 count=0 与 null 指标，不填 100%；所有指标带分母。
- 保存训练诊断、loss、三组 LR、梯度范数、实际参数量、吞吐和峰值显存。
- 记录 raw rows、不同训练目标/输入数、平均曝光、输入 token 和监督 token，不能仅记 epoch。
  batch64 时 nominal tokens=8192、实际送入模型的 tokens=8128（含 PAD）、监督 tokens=256。

pilot 建议：每 250 步固定 val 子集 512 行；每 1000 步及末步完整 val 4077 行；生成 batch=64（smoke 为 8）。
train 诊断固定 256 行。subset_seed=42，和 split seed 的用途分别记录，不能在每次评估重新抽样。
只有 full-val overall exact 更新 `best`，并列保留较早 step；同时记录 unseen-input 成绩，不因结果好看后验换 best 规则。
checkpoint 建议每 1000 步及末步保存；日志每 50 步，smoke 更频繁。

test 4079 行冻结，不参与训练、基线拟合、超参或 checkpoint 选择；最终评估需显式 opt-in。
数据构造时全池的格式/冲突检查不等于在 test 上选模型，但它已经影响了子任务选择，科学报告必须披露。

解释标准：

- 高 overall exact 但只在已见输入上成功，说明 random-row 补全有效，不能宣称学到普适数值关系。
- 新输入上超越训练众数、以及相对查表总体基线的增益，是更有价值的学习证据；不设未经讨论的“必须99%”验收线。
- 不以关系残差小代替系数正确；不在首版增加未推导的带帽基底物理关系损失。

## 10. 首版已新增或接入的代码（待 GPU 验收）

| 文件/位置 | 计划职责 |
| --- | --- |
| audit_representation.py（已有） | 保持原审计逻辑与来源，不改成训练器 |
| alphabet.py / data_contract.py | 原生映射、相邻/非零谓词、固定源哈希、版本和数组协议 |
| oracle.py | 精确源读取及 0y 查询；复用纯数据校验，不依赖旧 0y 模型 |
| build_samples.py | 筛选相邻目标、C36、来源/家族/精确输入分组和审计 |
| splits.py | PCG64 random-row 固定划分与 train 覆盖诊断 |
| convert_data.py | 新数组发布、逐项读回、哈希与 metadata；GPU 工作流门禁 |
| tokenizer.py | 117 词表、38 项编码、128 槽布局、严格解码、监督边界 |
| dataset.py | 新模型输入接口、batch 编码、只读 mmap、可恢复 sampler |
| model.py | 共享 GPT 的 4/256/4/4 组装、prefix mask 安装/验证、参数计数 |
| train.py | 独立 runner；更新旧硬编码假设，保留任务级训练/评估/计账方式 |
| baselines.py | 仅从 train 拟合众数和精确输入查表 |
| evaluator.py / eval_checkpoint.py | 114-token prompt 生成、完整/分层指标、显式 test 门禁 |
| checkpoint.py | 独立任务版本、模型/数据/编码/mask/优化器/RNG/sampler 的严格恢复契约 |
| runtime.py / gpu_smoke.py | Linux bf16 CUDA 门禁、数据准备、模型核验和20 vs 10+10流程 |
| configs/ | base、smoke、tiny_overfit、m2_adjacent_random_pilot 及 YAML 打包 |
| README / SERVER_HANDOFF_M2.md | 新任务入口、阶段状态、服务器命令与输出解释 |
| pyproject.toml | 只增加新任务包和 YAML 发现条目，保留所有旧条目 |

不是机械复制旧 0y 项目后全局替换名字。必须逐一替换旧的 0y/nonzero 数据谓词、11208 行常量、
16 槽、115 词表、10-token prompt、`[1,9)` mask、old task/checkpoint ID 和 35024 步配置。
严禁通过导入旧 task runner 而意外触发旧路径、标签筛选或旧模型恢复。

建议独立版本：

- dataset：`hexagon-mhv-w8-2y-adjacent-nonzero-m2-38-x32-random-row-v1`
- tokenizer：`hexagon-m2-adjacent38-base100-fixed2-packed128-v1`
- checkpoint：`hexagon-m2-adjacent38-prefix128-training-v1`

旧 40 项设计的词表、prompt/mask 边界和输入分组语义均不兼容；未来 loader 必须拒绝把旧版 metadata/checkpoint 当作本版恢复。

checkpoint 的依赖哈希覆盖新任务实际使用的 Python/config 文件、共享 core 和 mask factory，
并包括用于源读取的审计模块；不因无关旧项目改动而无谓失效，也不能漏掉实际依赖。

## 11. 验证与执行阶段

### A. 本地实施与静态审阅（2026-09-14 已获准并实施）

- 只编写上述代码/配置/文档；核对接口、下标、计数、版本与旧模块隔离。
- 用户现已授权按计划写代码；不等于授权上传、提交、连接服务器或启动计算。
- 用户先前批准的本机 CPU 例外仅用于已完成的精确数据审计，不延伸为本地模型核验权限。
- 不安装/激活 Mac 环境，不启动 CPU/MPS 模型测试。额外独立本地数据执行也应明确确认范围。

### B. 服务器首次 GPU smoke（独立授权）

1. Linux、单 GPU、bf16 门禁先于数据发布和模型构建；使用服务器已有环境，记录实际版本，不猜测 SSH/venv。
2. 完整准备并核对数据、固定切分和精确查表基线；主机读写、gzip/数组整理是 GPU 工作流必要步骤。
3. CUDA 上核对全部编码：两个 Y 的顺序、36 槽顺序、符号/高低块、NUM_0/PAD、prompt、shift 和仅四项 loss；位置相邻性在数据侧检查。
4. 核对 prompt 接口不接收位置、word 背景或标签；固定 `(C36,y_types)` 后，仅替换附带的审计位置、追溯 word 或 row ID 不改变编码或模型路径。
5. 用非零残差投影的诊断模型做无答案泄漏检查，并用错误扩大到答案的 mask 做负对照；不能用零残差初始化造成假通过。
6. 检查完整 prefix 的无缓存生成与一次 prefill 后 KV-cache 生成一致、EOS 行为正确、无 PAD 路径泄漏；记录数值差异。
7. 检查 finite loss/梯度、参数量 3,206,400、固定 batch、学习率边界和 loader 游标。
8. 同一初始化和 horizon 下比较连续20步与10+10恢复；检查参数、优化器状态、RNG和实际下一批数据。
   参数恢复对照先沿用既有 `rtol=1e-5,atol=1e-6`，不为通过而放宽；其他比较在实现时逐项写明容差和理由。
9. smoke 不做最终 test、不选正式 best、不自动开始 tiny-overfit 或 pilot。

### C. Tiny-overfit（独立阶段）

从 train 选择 32 个不同完整输入，尽量覆盖五个位置、不同 y 查询和正负标签；选择规则与实际样本存档。
只用于工程核验，不改变正式 train 池分布。检查自由生成 exact=100%、invalid=0；失败先诊断，不删困难样本。
tiny 模型随后丢弃，不作为 pilot 的预训练 checkpoint。

### D. 5000 步 pilot（独立阶段）

重新初始化 M2，使用冻结 random-row 数据和完整5000步调度。
阶段末对比查表、众数、模型 overall/seen/unseen，报告耗时和显存；据此决定更长训练或后续 exact-input-grouped 实验。
如果模型只学会已见输入，按该结果如实总结，不自动上20M或将测试集换成更容易的切分。

## 12. 本次验收与后续边界

本次实现交付：独立数据/编码/训练/评估/checkpoint 模块、三阶段配置、GPU smoke 入口及交接文档。
**尚未生成数组/切分/checkpoint，未运行 Python 检查或模型，不能宣称 GPU 验收通过。**
当前范围只到本地实现与静态审阅；服务器 smoke、tiny-overfit 和 pilot 均是之后独立执行阶段。

以下均不在本小实验首版：

- 非相邻 y、零目标、其他圈数、约210万全候选训练；
- 显式提供一个位置的 39 项或两个位置的 40 项对照；
- M0/M1 或新的大型模型、迁移旧权重；
- 自动对称增强/轨道平均、family-heldout/输入分组切分、多个训练种子；
- SLOT-token 消融、可变长数字编码、联合零分类器、物理关系损失或两候选答案输出；
- 自动提交 Git、上传数据、连接服务器、安装环境、删除既有产物。

这些可以在 M2 pilot 结果明确后分别提出，不作为本轮计划的隐含工作。
