# Heptagon 三圈 MHV 系数预测计划

日期：2026-09-06

进度同步：2026-09-07。后续当前状态以
[Heptagon PROGRESS.md](../../heptagon_symbol/PROGRESS.md) 为入口；本文件保留设计与实施记录。

状态：本地数据转换、random-row 切分及完整性审计已完成；数据加载、tokenizer、mask、单卡训练、评估与 checkpoint 代码及测试已编写，待服务器验证。未执行单测、推理或训练。转换实测结果见第 10 节，代码进度见第 11、12 节。

## 1. 本次实验的目标与边界

在 heptagon 三圈（weight 6）MHV Symbol 的非零项中，学习

\[
(l_1,l_2,l_3,l_4,l_5,l_6)\longmapsto c\in\mathbb Z\setminus\{0\}.
\]

用户已选择使用 **4 层、512 维、8 heads** 模型。本次使用 **逐样本随机切分（random-row split）**，不按 D7 轨道分组。

- 输入保留完整六字母 word，不做 canonicalization、quad 压缩或对称性增强。
- 模型直接学习整数系数，不提供轨道编号、物理关系标签或一致性损失。
- train/val/test 中不得存在完全相同的 word；对称相关但不同的 word 允许跨集合出现。
- 保留验证集和测试集。“直接学习”在本计划中指随机划分后的系数监督学习，不指全部数据同时用于训练和评估。
- 本次衡量同圈随机留出 word 的预测能力，不作为未见 D7 轨道泛化或跨圈泛化的证据。
- 首轮仅处理非零项，不加入零样本；D7 变换表和物理关系验证不作为本轮实施的前置条件。

## 2. 已确认的数据格式

原文件：

```text
/Users/hzq/hep_th/Building Intelligent Models from Scratch/Symbol_Data/heptagon_symbol_weight_2_4_6_8_MHV/hep_w6_phy.wxf
```

2026-09-06 已通过只读、受限解压和表达式解析核查整个文件；未启动 Wolfram Kernel 或做符号求值。

| 属性 | 已核查值 |
| --- | ---: |
| 文件大小 | 1,823,955 bytes |
| 文件头 | `8C:`，压缩 WXF |
| 解压后字节数 | 42,915,610 |
| 顶层表达式 | `Plus` |
| 项数 / 唯一 word 数 | 467,250 / 467,250 |
| 每个 word 的字母数 | 6 |
| Alphabet | `a11`…`a17`, `a21`…`a27`, …, `a61`…`a67`，共 42 个 |
| 系数 | 非零整数，范围 −48～48，共 33 种带符号取值 |
| 带显式 `Times` 的项 | 290,850 |
| 直接以 `SB` 表示的项 | 176,400，隐含系数 +1 |

原始文件 SHA-256：

```text
8cae0bbcfa99b28bf5b20f102966db4125f26567b3f7fd975850cf037f9b70ba
```

真实样本：

```wolfram
-48 * SB[a11,a11,a11,a11,a11,a21]
-48 * SB[a11,a11,a11,a11,a11,a22]
+48 * SB[a11,a11,a11,a11,a11,a25]
```

文件中自定义符号带有 `Global\`` 上下文。解析器需准确识别 `Global\`SB` 与字母符号，只在建立已知字母映射时去掉该上下文。

33 种系数为：

```text
-48,-24,-16,-14,-13,-12,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,
1,2,3,4,5,6,7,8,9,10,11,12,14,16,18,24,48
```

这些统计用于转换审计，不用于决定哪些样本进入训练集。保留原始系数的数值和符号，不重新归一化。文件所代表物理对象的具体归一化约定应另补来源说明，不能仅根据文件名 `phy` 推断。

## 3. 本地数据转换

### 3.1 方法与资源

转换只需解析 WXF，不需数学求值。实现受限读取器或使用经过核查的 WXF 反序列化接口，并对支持的表达式结构进行白名单校验；不得调用 `eval` 或加载任意对象执行代码。

本机已核查为 16 GiB 内存，约 109 GiB 可用磁盘。三圈文件的转换预计为秒到分钟级、峰值内存几十至几百 MB，实际耗时和峰值需在正式转换时记录。

- 输入大小上限建议 4 MiB，解压上限 64 MiB；超限明确报错。
- 逐个解析顶层项，避免构造包含全部节点的 Python 表达式树。
- 限制嵌套深度，遇到未支持的表达式、非整数或非法字母立即报错。
- 三圈转换不读取 weight 8/10 文件，不启动 Wolfram Kernel。
- 原 WXF 保持只读，输出使用新目录和临时文件，完整校验后才发布为可训练版本。

### 3.2 固定字母编号

按下式编号：

\[
\operatorname{id}(a_{ij})=7(i-1)+(j-1),\quad i=1,\ldots,6,\ j=1,\ldots,7.
\]

即 `a11=0`、`a17=6`、`a21=7`、`a67=41`。编号表必须保存进 metadata，不能依赖字母在文件中的首次出现顺序。

转换规则：

```text
Times[integer, SB[l1,...,l6]] → (six letter IDs, integer)
SB[l1,...,l6]                → (six letter IDs, +1)
```

对固定输入版本，其余结构视为异常，不跳过样本、不静默截断、不用正则替代完整结构校验。保留源文件顶层项顺序，发现重复 word 时失败并报告，禁止字典覆盖。

### 3.3 输出格式与位置

建议输出至仓库外的新目录：

```text
/Users/hzq/hep_th/Building Intelligent Models from Scratch/nanoinfra-artifacts/heptagon_symbol/w6/v1/
├── words.npy          # uint8, [467250, 6]，每个元素为 0..41
├── coefficients.npy   # int16, [467250]，保留原始带符号整数
├── splits.npz         # int32 的 train / val / test 行索引
├── metadata.json      # 来源、编号、版本、形状、dtype、hash 与切分信息
└── audit.json         # 解析与保存后读回的审计结果
```

数组数据约 3.74 MB，切分索引约 1.87 MB，不含少量文件头与 metadata。`npy/npz` 使用普通数值数组，不使用 object dtype 或 pickle；读取时设置 `allow_pickle=False`。

存储的是 letter ID 与整数系数，不是展开后的 `[BOS] ... [EOS]` 序列。训练输入可在加载时一次性预编码成 CPU tensor，再按 batch 转移到 GPU。

### 3.4 转换验收

- 源文件 SHA-256、解析字节结束位置、项数与上述记录一致；完整消费解压流，拒绝尾随内容。
- 每条 word 恰好六个合法字母；无重复 word、无零项、无分数或浮点系数。
- 显式整数项与隐含 +1 项均完整保留。
- 记录所有系数频数和字母频数，与直接解析结果对照。
- 保存后重新读取数组，核对形状、dtype、内容与哈希；验证 letter ID 与字母标签的往返转换。
- 抽取固定示例及随机样本，对照原表达式、整数数组和 token 序列。
- 保存输入/输出哈希、转换器版本、依赖版本、生成时间及资源统计。

## 4. 逐样本随机切分

采用固定 `split_seed=42`，比例为 train/val/test = 80%/10%/10%。

| Split | 样本数 |
| --- | ---: |
| Train | 373,800 |
| Validation | 46,725 |
| Test | 46,725 |

可实现为 `numpy.random.Generator(PCG64(42)).permutation(467250)`，依次取上述数量的行索引。固定实现、NumPy 版本和最终索引文件哈希；不同训练 seed 复用同一切分。

- 不计算或使用 orbit ID，不导入旧 D3 切分工具。
- 三组行索引互斥、并集覆盖全部数据；原 word 已全局查重。
- 在 metadata 中记录 `split_type: random_row`。
- 切分时不按系数重采样；记录各 split 的系数分布和训练集中未出现的验证/测试系数。
- 训练过程中反复遍历 train，val 用于模型与预算选择，test 在协议锁定后用于最终评估。
- 对称相关 word 跨 split 是本实验允许的设定；论文式随机留出结果与旧 D3 orbit-grouped 结果不做同难度比较。

## 5. Tokenizer、序列与监督对齐

### 5.1 词表

| 类别 | 全局 ID | Token type |
| --- | --- | --- |
| 42 个 letter | 0..41 | 0：word |
| BOS / COEFF / EOS / PAD | 42 / 43 / 44 / 45 | 1：control |
| PLUS / MINUS | 46 / 47 | 2：coefficient |
| NUM_0..NUM_999 | 48..1047 | 2：coefficient |

总词表 1048，保留 base-1000 编码。每个 `aij` 是一个 letter token，不能逐字符拆分为 `a`、`i`、`j`。三圈每个系数只需一个 magnitude token。

### 5.2 序列

```text
位置： 0     1  2  3  4  5  6     7        8        9       10    11..15
内容： BOS   l1 l2 l3 l4 l5 l6    COEFF    sign    NUM_|c|   EOS     PAD
监督： 0     0  0  0  0  0  0     0        1        1        1       0
```

真实样本 `SB[a11,a11,a11,a11,a11,a21] → -48` 的完整 token 序列为：

```text
[42, 0, 0, 0, 0, 0, 7, 43, 47, 96, 44, 45, 45, 45, 45, 45]
```

沿用 next-token shift：`idx=tokens[:-1]`，`targets=tokens[1:]`。只对 sign、magnitude、EOS 的目标位置计算交叉熵，其余设为 `IGNORE_INDEX`。

`sequence_len=16` 时现有 DataLoader 实际返回长度 15 的 `idx/targets`。batch=512 时名义配置为 `8192` 位置/step，实际输入为 `7936` 位置/step，有监督目标为 `1536` tokens/step。训练日志应区分这些口径，以样本数和数据遍历次数作为主预算，不能把 PAD 算成新训练信息。

### 5.3 Attention 与评估

- 六个 word 位置 `[1,7)` 双向可见，其余位置因果可见。
- 必须替换旧 `[1,11)` 范围，否则新序列中的真实系数会进入双向区域，造成答案泄漏。
- prompt 为 `[BOS] + 六个 letter + [COEFF]`，长度 8；分隔符位置 7。
- 训练、自由生成、checkpoint 恢复共用词表和位置定义。
- 新 checkpoint 保存 alphabet、词表版本、word 长度、attention 区间和 dataset/split hash。
- 自由生成只提供 prompt，不提供真实 sign 或 magnitude；严格检查输出格式和 EOS。

## 6. 模型与首轮训练配置

| 项目 | 选择 |
| --- | --- |
| 主体 | 现有 decoder-only GPT + word-bidirectional attention |
| 层数 / 宽度 / heads | 4 / 512 / 8 |
| KV heads | 8，无 GQA |
| 词表 / token types | 1048 / 3 |
| 参数量 | 13,657,600，按当前无 bias、无 weight tying 实现计算，组装后再次核对 |
| 初始化 | 从头训练，首轮 `train_seed=42` |
| 序列长度 | 16 |
| 每步样本 | 512，单卡，gradient accumulation=1 |
| 名义 `total_batch_size` | 8192，与配置中的 512×16 匹配 |
| 精度 / 编译 | bf16 / torch.compile，完成最小远端验证后启用正式训练 |
| 首轮正式预算 | 约 100 次 train 遍历，即 73,008 optimizer steps |
| Warmup / warmdown | 500 steps / 最后 20% steps |

预算计算：`ceil(100 × 373800 / 512) = 73008`。100 轮是探索性起点，不预设能达到 99% 或完成 sign 学习；若仍有明显改善，再独立设计更长预算和对应退火。停止使用 Chinchilla 自动步数。

沿用旧工程的优化器参数组作为第一版起点，并在运行 metadata 中保存有效学习率：

- AdamW，`betas=[0.9,0.95]`，`weight_decay=0.01`，`max_grad_norm=1.0`。
- 矩阵组 `lr_max=3e-4`；embedding 组 `embedding_lr=0.2`；输出组 `unembedding_lr=0.004`。
- 按现有实现，各组再乘 `(512/768)^(-1/2)`，并使用相同调度乘子。
- 不将旧 checkpoint 的 embedding 或输出头直接用于新 alphabet。
- 不自动以训练 loss 显示为零作为早停充分条件；保留自由生成验证与最佳验证 checkpoint。

## 7. 工程组织与适配范围

在 `nanoinfra-main_symbol` 内新增独立项目 `projects/heptagon_symbol/`，复用本版 `core`。旧五圈工程保留作为已有结果的参照，不改其固定字母表或旧 checkpoint 语义。

建议结构：

```text
projects/heptagon_symbol/
├── README.md
├── alphabet.py            # 42-letter 固定映射和词表定义
├── convert_data.py        # 受限 WXF → 数值数组与审计
├── dataset.py             # 数组读取、random-row split、batch 组装
├── tokenizer.py           # letter IDs + base-1000 + special tokens
├── train.py               # Hydra 组装与 word-bidirectional 配置
├── evaluator.py           # 共用序列定义的自由生成评估
├── configs/
│   ├── smoke.yaml
│   └── w6_random.yaml
└── tests/                 # WXF 结构、映射、切分、监督、mask 与推理验证
```

优先复用既有纯函数和框架接口；若旧项目函数绑定六字母词表、D3、长度 10 或 prompt 位置 11，则适配成新项目接口，不能直接调用。数据路径、manifest 和输出目录均由显式配置提供，使用新的 heptagon 命名空间。

## 8. 评估与结果解释

首轮自然采样，不做系数平衡。全数据中 `|c|=1` 占 77.11%，`|c|≤2` 占 92.15%；永远预测 `-1` 的 exact 为 39.35%。正式基线应从 train 决定固定预测，再在 val 上测量，不能用总体分布选择最优预测。

记录：

- 自由生成 exact、magnitude、sign、invalid rate。
- 每种真实系数的样本数和准确率、按绝对值分组的结果。
- Train 诊断子集与 val 的表现，区分拟合不足与泛化差距。
- optimizer steps、累计样本、train 遍历次数、实际输入/监督 tokens、耗时及峰值显存。

周期验证约每 5 轮一次（约 3650 steps），使用固定随机抽取的验证子集，例如 4096 项，禁止直接截取按 word 排序的前缀样本。定期及终评使用全量 val=46,725 项；子集指标和全量指标明确分列，最佳模型选择使用固定协议。

本轮不要求轨道指标。由于对称相关样本允许跨 split 且样本之间存在结构相关性，不把普通逐行独立二项误差条作为严格统计证据；单 seed 的结果也不能证明架构因果优势。

首轮完成后，根据错误类型决定是否增大预算、增加 seed 或测试其他系数编码；不在首轮同时加入 quad、零样本、物理约束损失或混合圈数。

## 9. 实施顺序与验收

1. **本地转换实现与数据审计**：生成新的三圈数组目录，完成结构、哈希和写出后读回核对。
2. **随机切分**：按固定规则生成 373800/46725/46725 索引，验证无重复 word 跨集合。
3. **模型输入适配**：实现 1048-token vocab、长度 16 序列、正确 loss shift、`[1,7)` mask 与长度 8 的评估 prompt。
4. **远端最小验证**：验证转换器的隐含 +1/显式负数/截断或异常输入处理；验证 token 往返、split 互斥和 mask。特别确认改变目标系数不会通过 attention 改变前缀表示。
5. **远端 tiny overfit 与 smoke**：验证模型能拟合小样本，训练与恢复 checkpoint 的推理配置一致，且自由生成过程不读取真实标签。
6. **正式训练**：约 100 轮，保存周期指标、全量验证结果、最佳与最后 checkpoint。
7. **结果整理**：按随机切分口径报告实际训练量；配置确定后评估 test，不将其与旧 D3 隔离实验直接作难度等同的比较。

本地负责转换、代码和文档；Python 单测、模型推理、smoke 和 GPU 训练在 AutoDL 执行，遵循仓库 `AGENTS.md` 与 `LOCAL_REMOTE_WORKFLOW.md`。执行远端计算前再给出确切命令、输入输出和预算。本计划本身不代表已执行或已验证训练代码。

## 10. 本地转换完成记录

已按用户授权执行第 1、2 步。转换器位于
`projects/heptagon_symbol/convert_data.py`，使用现有工具运行环境中的
Python 3.12.14 / NumPy 2.3.5；未安装依赖、启动 Wolfram、修改源 WXF 或运行模型。

数据已发布至：

```text
/Users/hzq/hep_th/Building Intelligent Models from Scratch/nanoinfra-artifacts/heptagon_symbol/w6/v1/
```

| 实测项 | 结果 |
| --- | --- |
| 数据 | 467,250 个唯一六字母 word；42 个字母；33 种非零整数系数 |
| 切分 | PCG64 seed=42；train 373,800 / val 46,725 / test 46,725 |
| 分组或压缩 | 无轨道分组、无对称性压缩、无数据增强 |
| 训练集系数覆盖 | 33 种全部覆盖；val/test 无训练集未出现的系数值 |
| 输出总大小 | 5,624,185 bytes，约 5.62 MB，包含数组、索引与两个 JSON |
| 转换总耗时 | 约 6.90 秒 |
| 进程生命周期峰值 RSS | 126,992,384 bytes，约 121.11 MiB |
| 审计结果 | 通过 |

除逐项解析、全局查重、频数核对、保存后数组读回与切分互斥/覆盖外，
还从全部输出行重建了完整 WXF 表达式字节，其 SHA-256 与源文件解压内容相同：

```text
d74115d3e4bad4437da4dbf1a307c11652680f35f78fdea129ea73b6c04ec4cf
```

这同时核验了 word 顺序、全部字母与系数以及隐含 +1 项。发布后再次读回
`metadata.json` / `audit.json`，核查 metadata 中记录的四个产物哈希和文件大小，均一致。

详细分布、样本和资源统计在产物 `audit.json`；来源、固定字母映射、切分协议、
转换器源码哈希与运行环境在 `metadata.json`。当前存储仍为紧凑整数数组，
并未预展开训练序列；后续代码适配进度见第 11 节。

## 11. 数据加载与 tokenizer 代码进度（2026-09-07）

已新增 `projects/heptagon_symbol/tokenizer.py`、`dataset.py` 和对应测试：

- 固定 1048-token 词表、六字母 word、长度 8 的无标签 prompt、默认长度 16 序列。
- 通用 base-1000 整数编码/严格解码，以及 w6 单块系数的向量化 CPU 预编码。
- 复用 `VocabLayout` / `NextTokenPrediction`，只监督 sign、magnitude、EOS。
- 只读加载现成数组与 split，校验文件哈希、字母、结构、word 唯一性和切分覆盖。
- 单卡无限训练加载器；训练集内逐轮洗牌，跨轮 batch 不丢尾部样本；
  记录下一未读位置，并通过确定性 epoch 排列恢复，状态绑定数据与编码配置。
- 编写合成数据单测和可选真实产物集成检查，命令与接口见新项目 README。

当前仅完成代码编写与静态审阅，未在本地运行 Python 单测或模型，也未远端执行。
尚需服务器确认运行后才能声称接口通过验证。数据加载器状态支持不等同于完整训练恢复；
后续仍需接入训练入口、`[1,7)` attention mask、自由生成评估、模型/优化器/RNG checkpoint，
并执行相应无泄漏、tiny-overfit、smoke 和恢复验证。

## 12. 训练、评估与恢复适配（2026-09-07，代码完成待验证）

新增独立 `model.py` / `train.py` / `evaluator.py` / `checkpoint.py` / `eval_checkpoint.py`：

- 复用现有 GPT、模型组装、AdamW 参数组、LR 调度、KV-cache 生成和 DCP。
  项目内单卡循环负责预算计数、验证与完整 checkpoint，不改共享 core Trainer。
- 显式安装 `[1,7)` mask 后才编译；恢复与评估核查模型、数据、编码及 mask。
- 生成只接收长度8的 word prompt；适配通用生成器的重复 EOS 补齐，保留严格格式检查。
- 每3650步验证固定随机4096项，每18250步及每次调用结束时验证全量 val；
  只有全量 val 结果参与 best 选择，同时报告 train 诊断、分系数/绝对值指标与常数基线。
- checkpoint 保存模型、优化器、采样器、RNG、调度配置/步数、代码和数据哈希。
  恢复严格锁定原预算与训练配方；每次调用写新目录，不覆盖或自动清理既有 checkpoint。
- `smoke`：20步，batch8；`tiny_overfit`：固定32项，batch32，1000步且最终 exact=1；
  `w6_random`：4/512/8、batch512、73,008步，bf16 + compile。
- 新增 CPU 测试代码覆盖 mask 无泄漏及负对照、缓存生成、参数计数、评估输入/模式、
  配置与 checkpoint 恢复后的下一步更新。GPU 数值一致性仍需远端实测。

服务器 AI 的命令模板、读写范围、分阶段预算和验收要求见
`projects/heptagon_symbol/SERVER_RUNBOOK.md`。以上是实现记录，不是运行验证结论。
