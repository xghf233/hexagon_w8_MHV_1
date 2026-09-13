# Hexagon 四圈 0y 系数预测：基于 Heptagon 工程的完整改造计划

日期：2026-09-13。文档版本：implementation-smoke-v2。

**用户已确认 random-row，并授权本地准备 GPU 冒烟候选代码。本地框架快照、新模块和配置已写；只做人工静态核对，没有运行 Python/CPU 核验、转换训练数组、实际切分、安装依赖、推理或训练，也没有执行 GPU 服务器操作。用户随后授权目录更名为 hexagon_w8_MHV，并提交到 xghf233/hexagon_w8_MHV_1。**

当前已实现的范围与执行入口以 [静态核对](STATIC_REVIEW.md)、[GPU 交接](SERVER_HANDOFF.md)
为准：random-row 转换/审计集中在 `convert_data.py`，CUDA 检查与连续/恢复对照集中在
`gpu_smoke.py`；**不设置任何独立 CPU 测试阶段**。服务器信息暂不需要，用户之后提供。
后文保留完整研究路线；orbit-grouped、报告绘图和扩展指标等尚未实现，不表示已完成。
优化器和正式计算预算仍是候选配置，须在实际运行阶段确认。

## 1. 目标、已确定事项与范围

### 1.1 本阶段任务

输入八个不含 y 的 Hexagon 原子字母，预测普通 BDS-like 四圈 MHV 完整 weight-8 symbol 中对应项的整数编码系数：

\[
W\in\{\hat a,\hat b,\hat c,\hat d,\hat e,\hat f\}^{8},
\qquad \widehat C_4(W)=f_\theta(W),\qquad C_4(W)=32c_4(W).
\]

第一阶段沿用旧项目的**非零系数预测**任务，只使用已知非零的 0y 项。0y 指 word 不含 y 字母，不代表系数为零。仅在此数据集上训练，不能声称已学会识别任意 word 的零系数。

| 项目 | 已确定的方案 |
| --- | --- |
| 物理对象 | 六点 Hexagon、四圈、MHV、普通 BDS-like E 的完整 weight-8 symbol |
| 数据选择 | 0y 非零项，输入长度 8 |
| 系数 | 精确整数 C4 = 32 c4；不再乘一次 32 |
| 字母 | 作者原生带帽字母；全局保留九字母 ID，本阶段输入只允许 0–5 |
| 模型 | 4 层、宽度 256、4 query heads、4 KV heads |
| 框架 | 现有 GPT 主干，word 区域双向注意力，答案自回归 |
| 数值编码 | base-100，绝对值固定两个块，符号在前、EOS 在后 |
| 监督 | 仅 sign、high、low、EOS 四个答案 token 的等权交叉熵 |
| 初始化 | 从头初始化，不加载 Heptagon/base-1000 checkpoint |
| 切分 | random-row，PCG64 seed=42，8,966 / 1,120 / 1,122 |
| 实施方式 | 本地先准备代码；测试、GPU 验证和训练按授权在服务器执行 |

### 1.2 与旧 Hexagon 研究计划的关系

历史 [Hexagon C2 → A3 计划](../HZQ-git/nanoinfra-main_symbol/projects/hexagon_c2_a3/PLAN.md) 研究“含两个 y 的目标 word + 36 个真实 0y 系数 → 目标系数”。现在新增的是**先做独立的 0y word-to-coefficient 基线**，不是把旧计划的条件输入直接搬过来。

- 本阶段不构造 36 项条件输入，不处理 2y/4y/6y 标签。
- 不默认将本模型输出替换为后续实验的真实 36 项系数；那是另一个带误差传播的实验。
- 旧计划中“当前不先训练 C2”“四圈尚未整数缩放”等描述属于当时状态，不适用于本阶段。此次只在新计划说明优先级，不改写历史文档或哈希锁定的旧清单。
- 不做三/五圈混合训练、跨圈迁移、零样本生成、对称增强、约束 loss、59 类分类头或浮点回归。
- 不将 0y 投影、相同字母名称或项数相同，直接解释成与论文 form factor 完全等同；不照抄其可积性或邻接关系。

### 1.3 尚需确认但不阻塞本地准备的选项

1. 主切分已确认 random-row；第 4 节的 orbit-grouped 仅保留为未来独立对照。
2. 第 7 节建议的学习率、batch 和 500 次训练集遍历预算。
3. 首轮仅 seed=42，还是后续获准增加 seed=43、44 的重复实验。
4. 实际服务器 checkout、数据、环境和输出路径；GPU 与可用磁盘。
5. 框架复制、冒烟模块本地实现及独立 GitHub 代码发布已授权；GPU 服务器数据上传与执行仍需后续确认。

## 2. 工程基线与目录安排

### 2.1 选择 w6 成熟闭环为底座，参考 w8 的八槽适配

只读检查的基线：

- [heptagon_w6_MHV](../heptagon_w6_MHV/README.md)：主训练代码位于 `nanoinfra-main_symbol/projects/heptagon_symbol/`，已有服务器训练记录。当前 HEAD 为 `6f9aae9d5366b9a62e87c6b800360a8ead2926bc`。
- w6 工作区另有用户的未跟踪 `training_curves.png`；不得删除、覆盖或默认纳入复制快照。开始实施前重新核对 Git 状态，以上 commit 只记录本次观察。
- [heptagon_w8_MHV](../heptagon_w8_MHV/README.md)：独立 `heptagon_symbol_w8` 模块已写八槽 tokenizer/mask、分片等代码，但文档标注尚未运行验收，不能视为已验证的训练底座。
- 本目录已按用户要求由 `hexagon_w6_MHV/` 更名为 `hexagon_w8_MHV/`，包括隔离的框架快照与新冒烟模块，尚未运行。**本任务始终 loop=4、weight=8**，并作为独立仓库发布。

已经从上述 w6 commit 按白名单复制框架，并新增 `projects/hexagon_symbol_w8_0y/`。参考 w8 的 `WORD_LENGTH=8`、`[1,9)` mask，但不引入其一亿项数据的分片、pilot 索引和原始坐标模式。

### 2.2 已建立的代码布局（训练产物尚未生成）

```text
hexagon_w8_MHV/
  HEXAGON_W8_0Y_ADAPTATION_PLAN.md      本文
  tools/                              已有精确提取/缩放工具，原样保留
  README.md                           项目入口
  AGENTS.md                           本项目执行边界
  SNAPSHOT_PROVENANCE.md               复制来源、commit、比对记录及排除清单
  SERVER_HANDOFF.md / STATIC_REVIEW.md GPU命令模板、静态核对与未验收项
  nanoinfra-main_symbol/
    core/                             复用，不因本任务默认改写
    projects/amplitude_symbol/        保留 attention 等依赖
    projects/heptagon_symbol/         复制基线，保留并标为历史模块
    projects/hexagon_symbol_w8_0y/     新任务模块，所有适配优先在这里
```

数据/训练产物放在代码目录之外，建议分别为：

- 已有源数据：`Symbol_Data/hexagon_mhv_4loop_native_x32/`。
- 新训练数组：`Symbol_Data/hexagon_mhv_4loop_0y_training/<split-version>/`。
- 运行产物：`nanoinfra-artifacts/hexagon_symbol/w8_0y/<run-id>/`。
- 服务器目录在连接后另行确认；Mac 的绝对路径仅用于 provenance，不能成为服务器硬编码依赖。

复制不得携带 `.git`、`.venv`、数据、checkpoint、日志、缓存、凭据或用户临时文件；保留许可证、依赖清单与 `nanoinfra-main_symbol/` 层级。不能覆盖已有 `tools/`，也不隐式同步 HZQ-git 或更改旧项目的 remote。

## 3. 训练数据格式对接

### 3.1 锁定源文件与缩放语义

直接读取已生成的 [整数版 JSONL gzip](../Symbol_Data/hexagon_mhv_4loop_native_x32/four_loop_mhv_symbol_native_x32.jsonl.gz)，不是重新解析 WXF、Mathematica 或重新提取振幅。

| 文件 | SHA-256 |
| --- | --- |
| `four_loop_mhv_symbol_native_x32.jsonl.gz` | `363bc50186af77f7a8b2f726964ff305ab2522b576a8a47f2b9811419dd69de2` |
| 同目录 `scaling_manifest.json` | `fdd7c25ec846fd32426055dc26500536c86d4fdbd9c5aff9c5db9bb0c358a00a` |
| 原始分式 `four_loop_mhv_symbol_native.jsonl.gz` | `5eaa2ad35811e8a093540115ae56cba9dda84f118447bdf34212f820dac5b694` |

JSONL 每行字段为 `word`（8 个原子名称）、`numerator`（十进制整数字符串）、`denominator`（固定为字符串 `"1"`）。**这里 numerator 已经是 C4，不是原始 c4 的分子。** 解析须直接转为精确整数，不经过浮点，不二次缩放、不裁剪、不丢弃 1920。

一条已核对的实际源记录：

```json
{"word":["a","a","mv","mu","mu","b","b","mu"],"numerator":"1","denominator":"1"}
```

它应对应数组 `word_ids=[0,0,4,3,3,1,1,3]`、整数标签 `C4=1`；原物理系数为 `c4=1/32`。数组阶段保留整数 1，只有 tokenizer 阶段才转成 `PLUS, NUM_0, NUM_1, EOS`。

完整整数文件有 243,000 条非零记录，包含多个 y 扇区。转换器只按 word 中 y 数量筛选，不按系数大小或首尾模式额外删除数据。

### 3.2 字母映射

| 磁盘源名称 | 物理/训练别名 | 全局 ID |
| --- | --- | ---: |
| a | hat_a = U/(VW) | 0 |
| b | hat_b = V/(WU) | 1 |
| c | hat_c = W/(UV) | 2 |
| mu | hat_d = (1-U)/U | 3 |
| mv | hat_e = (1-V)/V | 4 |
| mw | hat_f = (1-W)/W | 5 |
| yu | y_U | 6 |
| yv | y_V | 7 |
| yw | y_W | 8 |

字母契约为 `hexagon_hat_native_v1`。本数据集每行的 8 个 ID 必须全部落在 0–5；模型保留 0–8 的词表空间，避免后续加入 y 时重排普通字母 ID。`mu` 是一个原子，不是 m/u 两个字符。没有基底变换、antipode 反序或轴局部坐标映射。

### 3.3 必须重现的 0y 审计结果

以下来自对当前源版本的完整只读统计，不是训练结果：

| 指标 | 预期值 |
| --- | ---: |
| 0y 唯一非零 word | 11,208 |
| word 长度 | 8 |
| 不同非零 C4 | 59 |
| 最小/最大 C4 | -156 / 1920 |
| 正数/负数记录 | 5,610 / 5,598 |
| 系数绝对值的最大公约数 | 1 |
| 绝对值小于 100 | 11,142 |
| 绝对值大于等于 100 | 66 |
| C4 = 1920 | 6 |
| S3 轨道数/每轨道项数 | 1,868 / 6 |

保留完整系数直方图、字母/位置频数、high/low 数值块频数。校验 first entry 属于 0–2、last entry 属于 3–5，并核对六个 S3 图像的支持和系数一致性；这些检查不是完整可积性证明。

### 3.4 拟议训练数组

采用 w6 的小数据集方案，不使用 w8 的大规模分片。以源文件顺序过滤后形成稳定的训练数据行号。

| 文件 | dtype/形状 | 内容 |
| --- | --- | --- |
| `words.npy` | uint8 [11208,8] | 原子字母 ID，first → last |
| `coefficients.npy` | little-endian int16 [11208] | C4；当前范围完全可容纳 |
| `source_row_ids.npy` | little-endian int32 [11208] | 完整 243,000 行 JSONL 的零基行号 |
| `orbit_ids.npy` | little-endian int32 [11208] | 确定性 S3 轨道 ID，仅用于划分/审计 |
| `splits.npz` | train/val/test 三个 int32 数组 | 对上述 11,208 行的索引 |
| `audit.json` | JSON | 全量一致性、分布、覆盖和稀有值审计 |
| `metadata.json` | JSON | 数据契约、来源、哈希、切分、转换器版本 |

数组不保存 token 化的答案作为唯一数据来源；整数标签与 tokenizer 解耦。NPY 用 `allow_pickle=False`、只读 mmap；无需 PyTorch 对象序列化。数组有效载荷合计约 247 KB，不含头信息和 JSON。

`metadata.json` 至少记录：数据契约 `hexagon-mhv-w8-0y-x32-v1`、loop_order=4、weight=8、sector=MHV、y_count=0、nonzero_only=true、普通 BDS-like 完整 symbol、scale=32、两个缩放方向公式、字母映射/版本、source/manifest 哈希、数组形状/dtype/哈希、转换器源码哈希、切分版本/seed/实际计数、验证状态。tokenizer 如写入 metadata，必须明确是约束/推荐，不能伪装成磁盘系数的进制。

### 3.5 转换和加载验收

1. 读取前验证数据与缩放清单的可信哈希，检查 scale=32、loop=4、weight=8 和字母契约；禁止把任意文件仅凭名字当成本版源。
2. 流式解压至 EOF，检查 243,000 条源记录、字段类型、规范整数、分母 1、word 长度、合法原子和排序唯一性；未知格式或重复 word 在此已规范化源上必须报错，不能静默合并。
3. 过滤 0y，得到恰好 11,208 条；验证第 3.3 节统计与完整直方图。
4. 新建 staging 输出目录，写数组，再逐行读回并利用 `source_row_ids` 对照源 word/C4；验证映射可逆、次序无错位、int16 无溢出。
5. 按获准的第 4 节协议生成 splits，验证互斥、覆盖、索引界限和切分哈希。
6. 所有检查通过后发布为不可变新版本；失败不发布 passed，不覆盖旧版本，保留 staging 供排查。
7. 加载器启动检查 manifest、数组哈希、shape、数据语义和 split；不在每次启动重新随机划分。对这个小数据集，全量查重和扫描是合理的。

完整稀疏源表允许合法但缺失的 word 系数解释为零；但本阶段有限训练表只包含非零项，不实现“未知输入自动当零”的预测器。读取失败、截断和未知字母绝不能当成零。

## 4. 切分、稀有系数与实验边界

### 4.1 两个独立实验版本

用户已指定首轮 random-row，本版只实现此协议。orbit-grouped 保留为未来更严格的独立对照；random-row 允许轨道跨 split，不能论证未见轨道泛化，也不自动执行两套训练。

| 协议 | 固定生成顺序 | 拟议 train/val/test |
| --- | --- | --- |
| `random_row_v1` | PCG64(seed=42) 对全部行排列；先 floor(0.8N)，再 floor(0.1N)，其余 test | 8,966 / 1,120 / 1,122 行 |
| `s3_orbit_v1` | canonical word 排序获得轨道 ID，PCG64(seed=42) 排列轨道；同样 floor 分配 | 1,494 / 186 / 188 轨道，即 8,964 / 1,116 / 1,128 行 |

这些是按规定取整方式计算的计划数量，**不是已生成的 split**。两种版本使用独立数据目录与 manifest。划分后每个 split 内索引采用固定排序规则持久化；训练洗牌是另外的过程。

S3 映射为对 (a,b,c) 和 (mu,mv,mw) 同时施加同一个三元素排列，不反转 word 槽位。取六个图像中字母 ID 字典序最小的 word 为 canonical key。轨道 ID 只是分组键，不能输入模型；不同轨道仍可能受其他物理关系约束，不是物理独立性证明。

### 4.2 稀有值不可隐式处理

- 1920 的 6 个 word 构成一个完整轨道。base-100 的数值 token 19 在当前全表中只出现在这 6 项；改进制并未消除稀有值问题。
- 必须分别记录 train/val/test 中的 C4、abs(C4)、high、low，以及跨 high/low 共享数值词表的 token 覆盖。
- orbit-grouped 下，这一轨道只能全部位于一个 split。若留出，则 token 19 无训练正例，应单列为未见数值 token/未见系数情况，不伪称普通插值。
- 默认协议不为追求系数覆盖而反复挑 seed、搬样本、拆轨道或保证 1920 入 train；任何覆盖约束都需成为预先定义的另一个切分版本。
- 首轮不加 rare oversampling 或 class weights，不删除/截断 1920，也不把它硬编码成特殊规则。
- train/test 的文件、标签、轨道伙伴不能作为模型条件输入；诊断子集只能从对应 split 选择。
- test 封存，模型选择只使用 val；最终协议锁定后再经批准评估 test。分组评估的有效独立单元是轨道，不能无条件使用行独立假设给出过窄置信区间。

## 5. base-1000 → base-100 固定两块：完整编码规范

### 5.1 数值语法

令 q=floor(abs(C4)/100)、r=abs(C4) mod 100。固定输出 `SIGN, NUM_q, NUM_r, EOS`，q/r 的范围均为 0–99。首轮数据全部满足 abs(C4)<10000。

| C4 | 固定两块答案 |
| ---: | --- |
| 4 | PLUS, NUM_0, NUM_4, EOS |
| -156 | MINUS, NUM_1, NUM_56, EOS |
| 1920 | PLUS, NUM_19, NUM_20, EOS |
| 0（仅语法测试） | PLUS, NUM_0, NUM_0, EOS |
| 9999（仅边界测试） | PLUS, NUM_99, NUM_99, EOS |

解码按 sign × (100q+r) 精确恢复 C4。数字块不是十进制字符串拼接；例如 q=1、r=4 表示 104，不是 14。

- **高位 NUM_0 是有效监督目标，不能当 PAD 或忽略。**
- 不允许省略高位、增加第三块、负零、错误位置 EOS 或数值块中出现控制 token。
- abs(C4)>=10000 必须拒绝，不能取模截断；未来更高圈需要新的编码版本。
- 通用语法可编码正零以便测试/未来扩展，但当前非零数据加载器拒绝零标签。
- 编码版本建议 `hexagon-w8-base100-fixed2-v1`。旧变长解码器的“禁止多块前导零”规则必须替换，不能只把 BASE 常数从 1000 改成 100。

### 5.2 固定 token ID 与 type

| 名称 | 新 ID/区间 | token type |
| --- | --- | --- |
| 九个字母 | 0–8 | 0（word） |
| BOS | 9 | 1（control） |
| COEFF | 10 | 1（control） |
| EOS | 11 | 1（control） |
| PAD | 12 | 1（control） |
| PLUS | 13 | 2（coefficient） |
| MINUS | 14 | 2（coefficient） |
| NUM_0 … NUM_99 | 15–114 | 2（coefficient） |

常量目标：WORD_SIZE=9、CONTROL_SIZE=4、COEFF_SIZE=102、BASE=100、NUM_OFFSET=15、VOCAB_SIZE=115、N_TOKEN_TYPES=3。`build_layout` 的三个半开区间为 [0,9)、[9,13)、[13,115)。

token ID 与序列位置是两件事，例如 COEFF 的 ID 是 10，但位置是 9。词表整体改变，旧 embedding/head 不能按同一 ID 解释或恢复。

### 5.3 序列位置、attention 和 loss 对齐

序列长度保持 16；prompt 长度为 10，包含 BOS、八个 word 字母和 COEFF。

| 零基序列位置 | 内容 | 原序列 loss weight |
| --- | --- | ---: |
| 0 | BOS | 0 |
| 1–8 | 八个 word 字母 | 0 |
| 9 | COEFF | 0 |
| 10 | SIGN | 1 |
| 11 | HIGH | 1 |
| 12 | LOW | 1 |
| 13 | EOS | 1 |
| 14–15 | PAD | 0 |

常量为 WORD_START=1、WORD_END=9（exclusive）、COEFF_POSITION=9、PROMPT_LENGTH=10、ANSWER_LENGTH=4、SEQUENCE_LENGTH=16。

第 3.1 节实际样本的完整 token ID 序列应为 `[9,0,0,4,3,3,1,1,3,10,13,15,16,11,12,12]`。这为源记录 → 数组 → token → 精确整数往返提供一个不含占位系数的端到端验收样例。

- word 双向区域严格为 [1,9)，其余按因果 mask。COEFF 可以看到整个 word，但 word 不能看到答案。
- 禁止复用 w6 的 [1,7) 或 amplitude attention 工具的默认 [1,11)；后者会把当前 sign 位置纳入双向区域，造成泄漏。
- 经 `NextTokenPrediction` shift 后，`idx/targets/token_types/target_types/loss_weights` 的 shape 都是 [B,15]。
- 仅 targets 的零基位置 9、10、11、12 有效，分别预测 SIGN/HIGH/LOW/EOS；其他 targets 为 -1。
- token_types 与输入位置对齐，target_types 与右移后的目标对齐。EOS 类型仍是 control，不能因为答案有四个 token 就把所有 target_types 强制置为 coefficient。
- 只使用 0/1 loss mask 和现有标准 CE，不引入 label smoothing、sign 重加权、物理约束 loss 或类别加权。

### 5.4 批量编码与自由生成

`assemble_sequence` 与 `assemble_batch` 必须产生完全一致的 ID、types、weights。批量路径统一计算 q/r，不再只接受 abs(C)<1000，也不再按 3 个答案 token 分配位置。

推理只接受 word，构造十 token prompt，不接受真实系数、高位块、轨道 ID 或其他标签。首轮保持全词表 greedy 自由生成，不使用真实标签约束输出；最多生成 4 个 token，并要求首个 EOS 恰好在第四个生成位置。

缺 EOS、提前 EOS、数量不符、非数值块、负零均计 invalid，不能修补成正确答案或将 invalid 当 0。若底层批量生成器在已终止行后填充重复 EOS，只允许按其明确协议去掉终止后填充；不能修补终止前的错误。数值生成步的 type 必须与训练一致；EOS 是终止控制符，不能带着错误类型作为下一步有效输入。

可以以后另做语法约束解码对照，但必须标记为不同 decoding 协议，不能用它替代首轮 unconstrained 指标。

## 6. 模型与数据加载改造

### 6.1 模型结构

| 项目 | 旧 Heptagon w6/w8 基线 | 本计划 |
| --- | --- | --- |
| 层数 | 4 | 4 |
| hidden width | 512 | 256 |
| query/KV heads | 8/8 | 4/4 |
| head dimension | 64 | 64 |
| FFN hidden | 2048 | 1024 |
| 词表 | 1048 | 115 |
| token types | 3 | 3 |
| 位置/归一化 | RoPE、QK norm、无参数 RMSNorm | 保留 |
| FFN | ReLU²、两层无 bias 投影 | 保留 |
| 输入/输出权重 | 不共享 | 保留 |
| head softcap | 15.0 | 保留 |
| Dropout | 当前实现没有 | 首轮不新增 |
| 参数量 | 13,657,600 | 3,205,376（静态计算，待实测确认） |

参数公式为 12 L d² + (2V+3)d；这里 L=4、d=256、V=115。仅修改项目级模型组装参数和 tokenizer，预期不需要更改 core GPT/attention/MLP。

不增加 encoder–decoder、CLS 分类头或 59 类输出。论文将 word 与系数作为 token 序列，用 CE 学习，并使用 base-1000；本方案保留这一任务表达而改变进制。论文中的 epoch 定义为 300,000 样本呈现，不能与本文的完整 train 遍历直接比较。[论文 §3](https://arxiv.org/html/2405.06107v2#S3)

### 6.2 小数据加载器

优先复用 w6 的有限 Dataset、train-only 无限循环 DataLoader、按 epoch 确定性洗牌和恢复游标。修改旧的 weight=6、loop=3、42 字母、范围 ±48、`V6` 查重等硬编码；本任务为 weight=8、loop=4、九字母契约中的 0–5、已验证的当前系数范围和 `V8` word。

服务器 GPU 入口首次启动执行必要的文件/格式审计并在主机预编码 train；这不是独立 CPU 模型核验阶段。以 8,966 条 train、16 槽计算，tokens(int64)、types(int64)、weights(float32) 三份张量约 2,869,120 bytes，另有临时张量；不需要一亿数据量的分片管线或 worker 池。

只允许单进程/单卡、梯度累积为 1。train 每轮采用 PCG64(SeedSequence([train_seed, epoch])) 洗牌，不丢末尾样本，允许一个 batch 跨越 epoch。loader state 绑定数据/split/tokenizer 哈希、batch、seed、shuffle 算法和 NumPy 版本，记录下一未读位置。

训练加载器不得直接读取 gzip 作为每步来源；gzip 只在一次性转换/审计中使用。字母和 C4 以紧凑 CPU 数组保存，GPU 每次只接收当前 batch。

## 7. 训练参数与配置修改清单

本节除 4/256/4 与编码几何外，均为**首轮建议值，尚未经训练验证或计算授权**。目标是提供可直接审核的配置，不把旧 w6 的训练轮数或吞吐照搬为新任务保证。

### 7.1 主运行参数

| 配置/行为 | 旧 w6 | Hexagon 建议值/改法 |
| --- | --- | --- |
| 默认模块入口 | heptagon_symbol.train | hexagon_symbol_w8_0y.train |
| 默认配置名 | w6_random | smoke，20步；正式候选须显式选 w8_0y_random；本版不提供 orbit 配置 |
| `data_dir` | w6 数据目录 | 必填的新 Hexagon split-version 目录 |
| `metadata_sha256` | 旧训练入口未显式传入 | 新入口必填可信 metadata 哈希并传给 Dataset |
| `output_dir` | 新的仓库外目录 | 同样必填，必须在代码、数据和已有 checkpoint 之外且尚不存在 |
| `resume_from` | null | null；只允许恢复相同新协议 |
| `sequence_len` | 16 | 16 |
| `device_batch_size` | 512 | 128 个样本 |
| `total_batch_size` | 8192 | 2048 = 128×16，仅为旧接口的名义 token 数 |
| 单卡/累积 | 单卡、累积 1 | 保留，不把 total_batch_size 误当 2048 个样本 |
| `seed` | 42 | 首轮 42；与数据 split_seed 分别记录 |
| `compile` | 正式配置 true | 首次 false；通过单独 compiled smoke 后再决定是否用 true |
| `head_softcap` | 15.0 | 15.0 |
| `overfit_n_samples` | null | 正式训练 null；诊断配置单独指定 |
| `require_overfit_exact` | false | 正式训练 false；tiny 配置为 true |
| `max_steps` | 73,008 | 由实际 N_train 和获准遍历预算计算，持久化为显式整数 |
| `stop_after_steps` | null | 初段可用 5,000；含义保持“本次额外完成的更新数”，不是全局绝对停止编号 |

第一版项目级 `validate_config` 的允许字段、类型检查、配置组合、checkpoint recipe 和 eval 重建逻辑必须同步更新。不要为了容纳新字段放松成“忽略未知 key”。`metadata_sha256`、任务和 split 必须与数据契约一致，不能从目录名猜测。

如果保留旧 `total_batch_size` 字段，明确它不参与自动 batch/Chinchilla 计算；本任务每步实际处理 128 个样本、1,920 个送入模型的位置、512 个有效监督 token。

### 7.2 优化器：配置学习率不等于实际学习率

复用 core 的两组 AdamW optimizer/三个参数角色，不改底层实现。现有实现会统一乘以 `(d/768)^(-1/2)`：宽度 512 时为约 1.224745，宽度 256 时为约 1.732051。同一份原始 YAML 在缩宽后实际学习率会增大 sqrt(2)。

建议将三项原始学习率约除以 sqrt(2)，采用容易记录的近似值，以接近旧实验的实际步长，而不是无意识放大：

| 参数 | 旧 YAML 值 | 新建议 YAML 值 | 新模型实际峰值（乘 sqrt(3)） |
| --- | ---: | ---: | ---: |
| `lr_max`（主干矩阵） | 0.0003 | 0.0002 | 约 0.000346410 |
| `embedding_lr` | 0.2 | 0.14 | 约 0.242487113 |
| `unembedding_lr` | 0.004 | 0.0028 | 约 0.004849742 |

这些是继承项目分组配方的起点，尤其 embedding 的实际学习率较大，不是论文给出的通用建议。只有数值稳定性和 tiny/pilot 检查通过后才能接受；若需调整，用新的配置/run 版本，不能修改旧 checkpoint 的恢复配方。

| 其他优化器项 | 首轮建议 |
| --- | --- |
| type | adamw |
| weight_decay | 0.01，保持原参数组施加规则 |
| betas | [0.9, 0.95] |
| fused | true，只在已确认支持的服务器运行 |
| eps | 现有 core 硬编码 1e-10；记录实际值，不新增不存在的 YAML 字段 |
| max_grad_norm | 1.0 |
| scheduler.type | linear，保持 warmup → constant → linear warmdown |
| warmup_steps | 200（旧正式值 500） |
| warmdown_ratio | 0.2 |
| final_lr_frac | 0.0，保留 core 现有零基 schedule 的端点语义 |

日志给每个参数组增加明确角色名。现有 optimizer 的遍历顺序是 unembedding、embedding、matrix，不能把按位置取出的第一个学习率标成 matrix。梯度裁剪函数返回的 norm 是裁剪前的总范数，报告不得把它错误标为“裁剪后梯度”。

### 7.3 训练预算与停止规则

建议正式候选配置预设 **500 次完整 train 遍历的样本呈现预算**，不是保证需要或只需 500 轮。计算式为：

\[
\mathrm{max\_steps}=\left\lceil\frac{500N_{\rm train}}{128}\right\rceil.
\]

| 切分 | 计划 N_train | 500 次遍历的 max_steps |
| --- | ---: | ---: |
| random_row_v1 | 8,966 | 35,024 |
| s3_orbit_v1 | 8,964 | 35,016 |

batch 允许跨 epoch，最终可能比预算多呈现不足一个 batch，报告真实 `rows_consumed/N_train`。不套用旧 73,008 步、旧 100 轮，也不把论文的 300,000 样本/epoch 当成此处的一轮。

执行建议：先检查新协议和 tiny，再用正式固定 schedule 的前 5,000 步观察学习情况；若获准继续，按同一 horizon 恢复，不重新 warmup。不能先按 max_steps=5000 衰减到尾部，再把 checkpoint 当成 max_steps≈35000 的同一训练无缝恢复。

首轮不自动使用短 patience early stopping。分别观察 train/val 的符号、幅值和 exact；若只学会高位零或幅值而符号停滞，不立刻据此增加层数。非有限 loss/梯度、数据校验失败、磁盘不足必须停止；达到调用步数上限则正常保存并退出。调整总 horizon、batch、学习率、seed 或 compile 的后续实验应新建 run，不能伪装成完全一致恢复。

没有本任务吞吐测量前，不给出分钟/小时训练完成承诺。运行预算、GPU 时间和磁盘按每阶段实际测量与授权控制。

### 7.4 评估、日志与 checkpoint 频率

| 配置项 | 旧正式值 | 新正式建议 |
| --- | ---: | ---: |
| evaluation.subset_size | 4096 | 512，兼容子集路径；不得超过对应 val 数 |
| evaluation.subset_seed | 20260907 | 20260913 |
| evaluation.interval_steps | 3650 | 500 |
| evaluation.full_interval_steps | 18250 | 500 |
| evaluation.full_validation | true | true；val 仅约 1,100 条，可每次全量 |
| evaluation.gen_batch_size | 64 | 64 |
| evaluation.train_diagnostic_size | 256 | 256，tiny 时覆盖全部 tiny train |
| checkpoint.save_every | 3650 | 2500；另保存 best 改善点与调用结束点 |
| logging.log_every | 100 | 50 |

正式配置两个评估间隔同为 500，因此周期评估直接用 full val，不重复跑同一时刻的 subset。`best` 只按 full-val exact 严格提升更新；平分保留较早 checkpoint，不用 test、train loss 或有标签 teacher forcing 准确率选择 best。

日志改为从编码元数据推导计数：每样本 15 个模型输入位置、4 个监督 token，完整非 PAD 序列 14 个位置。对 B=128，每步分别记录 1,920（含输入 PAD 位置）、512 和 1,792（非 PAD 位置）；不能保留旧 `B×3` 监督计数。`rows_consumed` 是样本呈现次数，不是独立样本数量。

checkpoint 只写新目录，不自动删除历史文件。参数量约 3.2M，但 optimizer/RNG/元数据也占空间；首轮建议预留约 10 GiB 运行目录空间，实际根据首个完整 checkpoint 的字节数及保存事件上界修正。每 500 步均可能刷新 best，磁盘估算不能只按每 2,500 步保存计算。

### 7.5 诊断配置，均与正式结果分开

| 配置 | batch / total_batch_size | max_steps / warmup | 评估与验收 |
| --- | --- | --- | --- |
| smoke | 8 / 128 | 20 / 2 | compile=false；每 10 步评估/保存；subset=8，gen_batch=4，train_diag=8，log=1；只验通路和有限数值 |
| tiny_overfit | 32 / 512 | 1000 / 20 | 固定 32 条 train；每 100 步评估、每 500 步保存；subset/gen_batch/train_diag=32，log=10；全 tiny 自由生成 exact=1，invalid=0 |
| 正式初段 | 128 / 2048 | 正式完整 horizon，stop_after_steps=5000 | full val；不是一次已完成的 500 轮训练 |
| compiled_smoke（可选） | 同 smoke | 同 smoke | 只在未编译路径通过后独立执行，并验证数值、保存恢复和生成一致性 |

smoke/tiny 的 full_validation=false、best=null，不能误写成“正式验证已通过”。tiny 只从 train 取样，最好覆盖正负和 high=0/high>0；若规定 train 没有稀有 1920，不从 val/test 借入。1920 和固定前导零的边界由合成单元测试覆盖，不因此污染训练集。

## 8. 评估指标、基线和报告

### 8.1 主指标与诊断

主指标是**自由生成后完整带符号整数 C4 的 exact accuracy**。每条预测必须先通过第 5 节语法验证。invalid 在整体 exact/sign/magnitude 中均计错误；语法正确但预测为 0 的非零目标也是错误，不把它误记 invalid。

同时输出：

- magnitude accuracy、sign accuracy、invalid rate 及分子/分母计数。
- 自由生成的 high/low 块准确率、EOS 位置正确率；这些不能替代整数 exact。
- C4 与 abs(C4) 分组的数量和准确率；记录宏平均与整体按样本平均，避免多数小系数掩盖稀有值。
- high=0（11,142 项）与 high>0（66 项）、1920 的单独指标；若某组在 split 中为空，显示 n=0/不可用，不伪造 100%。
- 训练已见/未见的系数值和数值 token 分组，明确 token 覆盖是整个数据集词表统计还是 train-only 覆盖。
- orbit-grouped 时，增加每轨道平均以及“六个图像全对”的轨道 exact；验证/测试内的对称一致性只能作为诊断，六项同时错也可能一致。
- train 诊断和 full val 学习曲线；当固定 256 条 train 诊断无法解释问题时，再获准做全 train 评估，不能把子集 exact 当成全 train exact。

不把“预测高位 0 的准确率约 99.41%”当成模型成功，也不使用把所有 token（含 prompt/PAD）计入的 token accuracy 作为成果。

### 8.2 简单基线

常数系数预测器必须仅由 train 的最高频 C4 决定，再在 val/test 上按同一完整整数指标计算。另记录仅预测高位 0 的诊断基线。

random-row 可额外报告使用已知 S3 对称变换、只查 train 标签的 orbit lookup 基线，披露其覆盖率和 exact。这有助于解释模型是否超过直接对称复制；不是将查表结果并入模型预测。orbit-grouped 应验证与 train 轨道交集为零。

不调用整份真值表给测试预测补答案，不用类别候选表根据测试真值纠正输出。源数据全量审计与训练时可用信息严格区分。

### 8.3 报告产物

拟议保留 `run.json`、`history.jsonl`、`baseline.json`、`summary.json`、逐次 val 指标、checkpoint 元数据与显式最终 test 报告。字段必须写入 task、loop/weight、0y/nonzero_only、scale=32、base=100、fixed_blocks=2、115-token 词表、模型参数量、split/seed、实际组学习率、样本呈现数和训练遍历数。

报告脚本从 run metadata 读取 batch、验证样本数、学习率角色、warmup、任务名、日期和训练步数；清理旧报告中写死的“w6”“batch512”“46,725”“4096”和学习率图例。动态记录硬件与代码版本，不复制历史成绩、耗时或 GPU 型号为新实验结果。

第一阶段不把未完成的全可积性验证列为已通过，也不凭高 exact 声称恢复完整函数或证明了关系。论文、原始 symbol 审计、工程验证和新训练实测应分别引用。

## 9. Checkpoint、恢复与版本隔离

现有 DCP 底层可以保留，但项目 envelope 必须独立版本化，建议使用 `hexagon-w8-0y-training-v1` 和新 `hexagon_contract` 字段，不继续写 `heptagon_contract` 伪装兼容。

checkpoint 必须绑定：

1. 数据 schema、metadata/源数据/split 哈希、0y 与非零范围、scale=32。
2. `hexagon_hat_native_v1`、tokenizer 版本、base=100、fixed_blocks=2、所有 token ID/type 区间、word 长度、prompt 和生成长度。
3. attention mode 与 [1,9)、模型 4/256/4/4、词表 115、head 类型和 softcap、精度与 compile。
4. 优化器原始配置与实际参数组、固定 schedule horizon、batch、训练 seed、完整配置。
5. 数据读取游标、下一未读位置、洗牌算法与 NumPy 版本、Python/NumPy/Torch CPU/CUDA RNG、model 和 optimizer state。
6. 完成的 optimizer updates（1-based）、旧底层 step=completed_steps-1 的对应关系、best 与当前调用范围、代码文件哈希和环境版本。

恢复前先校验元数据，不加载完权重才发现协议错。拒绝旧 Heptagon、base-1000、变长 base-100、其他 split/缩放版本的 checkpoint；即使碰巧 tensor shape 一样也不允许混用。模型几何不同更不能自动截断/补零迁移 embedding 或 head。

恢复只允许白名单中的运行位置/本次追加步数等非科学配方变化。训练数据、词表、scale、模型、学习率、batch、horizon、compile、seed 不能静默变更。保存路径和输出目录可迁移，但不得把 provenance 中的旧绝对数据路径当成必需运行位置。

runtime attention mask 不是普通可学习参数，加载后必须重建并逐层验证。恢复评估模式、RNG 和下一 batch；验证过程不能改变后续训练序列。保存用 staging 原子发布，已有 checkpoint 不覆盖，best 指向的旧调用目录需一并保留。

## 10. 逐文件改造清单

以下均指新模块 `nanoinfra-main_symbol/projects/hexagon_symbol_w8_0y/`，不修改原 heptagon 项目。
这是完整路线的清单，不是全部已实现声明。当前转换/切分/审计合并在 `convert_data.py`，
GPU检查在 `gpu_smoke.py`；没有单独 `split.py`、`audit_data.py`、`tests/`、`make_report.py`
或 orbit 配置。其余当前实现见 STATIC_REVIEW；扩展功能按后续需求实施。
复制快照只是底座；新类名、导入、文档和元数据统一使用 Hexagon 新任务。

| 文件/区域 | 主要改造 | 特别核对 |
| --- | --- | --- |
| `alphabet.py` | 九字母映射、word 长度 8、0y 允许集合 | 删除 42 字母及轴坐标假设 |
| `convert_data.py` | 读取已缩放 JSONL gzip，筛选 0y，生成紧凑数组/清单 | 不读 WXF；不二次乘 32；可信哈希与全量读回 |
| `split.py`（新增） | random-row 与 S3 两种显式协议、canonical key、轨道 ID | 与旧 Heptagon/旧 form factor 的对称映射隔离 |
| `audit_data.py`（新增） | 统计、源-数组对照、split 互斥、稀有 token 覆盖 | 失败不可标记通过；输出只写新目录 |
| `tokenizer.py` | base100 fixed2、115 个 ID、8 槽、答案 4 token、严格解码 | 前导 NUM_0 合法、负零非法、越界拒绝、单条与批量一致 |
| `dataset.py` | 新 schema/哈希/8 字母查重、int16 C4、0y/nonzero/split 检查 | 删掉 weight6、±48、random-only、`V6` 等假设；沿用小数据加载 |
| `model.py` | 4/256/4/4 与新 vocab/type，显式 [1,9) mask | mask 安装后才允许 compile；不改 core |
| `train.py` | 新配置校验、可信 metadata、预算、日志、四 token 监督计数 | 旧 allowed keys、默认模块配置、B×3、4096 子集必须改 |
| `evaluator.py` | 固定四步自由生成、严格语法、exact/sign/magnitude/invalid、稀有块和轨道指标 | 无真标签输入；0 与 invalid 分开；EOS 和 type 协议 |
| `checkpoint.py` | 新 envelope、版本、哈希覆盖、tokenizer/split/scale 绑定 | 不能漏掉新 split/audit 文件及影响训练的配置文件 |
| `eval_checkpoint.py` | 新 contract 重建、generation length=4、test 显式授权 | 只载入可信新版本；恢复 encoding 不能使用旧默认常量 |
| `make_report.py` | 元数据驱动的新报告与曲线 | 学习率组顺序、grad norm 含义、旧日期/样本数/任务名 |
| `configs/base.yaml` | 本计划通用参数，不指定未确认 split 或服务器路径 | 几何由 tokenizer 固定，未知 key 必须报错 |
| `configs/w8_0y_random.yaml` | 锁定 random_row_v1 对应数据与正式预算 | 实际 metadata/数量确认后填 max_steps |
| `configs/w8_0y_orbit.yaml` | 锁定 s3_orbit_v1 对应数据与预算 | 不能加载 random split 冒充分组结果 |
| `configs/smoke.yaml` / `tiny_overfit.yaml` | 第 7.5 节诊断参数 | 不自动标 best/test 或用验证标签补 tiny |
| `tests/` | 数据、tokenizer、mask、模型、指标、恢复与报告的合成测试 | 在获准服务器执行，不把“代码已写”当通过 |
| `README.md` / `PROGRESS.md` / `SERVER_RUNBOOK.md` | 入口、任务契约、状态、分阶段授权和操作说明 | 不复制旧验收结论为本项目成绩 |

复制后的 `core/`、`projects/amplitude_symbol/blocks/attention.py` 首轮应保持源码哈希不变，只通过显式参数复用。若实施中发现真正的共享框架问题，单独描述影响和需要的改动范围，再决定是否调整；不顺手重构。

旧大数据 WXF reader、raw_coordinates/physical_letters 双模式、分片 sampler、1m/5m pilot、全数据动态数据池不属于这个 11,208 项任务的必要实现。

## 11. 分阶段实施与验收门槛

### 阶段 0：计划审阅（已完成）

初次仅交付计划；随后用户确认 random-row 和本地实现范围。现已准备 A–C 的冒烟候选子集，
未生成训练数组，完整研究功能与执行验收仍未完成。

### 阶段 A：工程隔离与数据准备代码

获准后核对源 w6 commit/工作区，按白名单复制到 Hexagon，不覆盖已有工具；建立新模块、项目文档、数据契约和转换/切分/审计代码。

验收：来源清单完整；原两个 Heptagon/HZQ-git 不变；旧工具及数据哈希不变；新模块不引用旧 WXF 路径、42 字母或 weight6 的运行入口。Git 初始化/提交/上传分别处理，不从计划推定授权。

### 阶段 B：tokenizer、loader、模型与合成测试代码

实现第 3–6 节及对应测试，但本地只做源码审查。服务器测试至少包括：

- 全范围 -9999…9999 的规范整数编码/解码往返；正零语法单测与 dataset 拒零分开。
- 重点边界 0、±1、±99、±100、±156、1920、±9999，以及 ±10000 拒绝。
- 单/批量 token、type、weight 一致；NUM_0 与 PAD 不混淆。
- 提前/缺失 EOS、负零、额外数值块、非法 token、错误固定长度拒绝。
- 数据篡改、重复 word、非 0y/零标签、错误 scale、错误源/manifest 哈希、split 越界/重叠/遗漏均报错。
- S3 canonical key、六图像系数一致、orbit split 无交叉、稳定 ID 和可复现划分。
- 8 槽 attention 的正例与 [1,7)/[1,11) 负例；改变答案不应影响对应更早预测位置，不能出现标签泄漏。
- 参数量实测 3,205,376；输出 logits 最后一维 115；token types 与生成协议一致。
- 有缓存/无缓存的确定性自由生成一致，生成仅接收十 token prompt。

测试用合成数据覆盖 1920 不意味着向训练集增加新的物理真值样本。

### 阶段 C：训练、评估、恢复与配置闭环

实现第 7–10 节。检查所有配置组合和强校验，覆盖四 token 监督计数、full-val best、test opt-in、稀有/空组指标、原始与实际学习率、save staging 和不可覆盖路径。

验收测试包括：相同环境下连续运行与“保存→恢复”在下一 batch、RNG、学习率及下一次参数更新上相符；错误 base/width/split/scale/source/code/environment 恢复被拒绝；评估/保存前后 RNG 和 train/eval 模式恢复；日志/报告从元数据读取真实参数。GPU/编译环境不同不承诺位级可复现。

### 阶段 D：获准环境中的数据发布与运行前验证

先确定服务器路径、环境版本、GPU/bf16 支持、磁盘、上传范围和费用。源 gzip 只有约 0.84 MiB，不需要重新下载科学数据或上传大型 Heptagon WXF。

本版遵循用户“全程不用 CPU 核验”：不运行 CPU 单测、CPU 微型模型或 MPS 回退。
首轮统一调用 GPU 门禁入口：必要的主机转换/全量源数据读回与发布 → CUDA 真实编码/监督/mask/生成检查
→ GPU 未编译20步 smoke → 同horizon的10+10步 checkpoint恢复对照。
tiny-overfit 与 compiled smoke 留待之后获准执行，不被首轮入口自动触发。

转换在哪台机器执行需单独确认，默认在获准服务器；之前对本地 symbol 提取的授权不自动延伸到训练工程测试。任何一步失败先报告问题，不把后续训练当作调试替代。

### 阶段 E：首轮正式初段与完整预算

锁定 split、数据哈希、tokenizer、4/256/4、建议或修订后的优化器/预算，使用全新 run 和 seed=42。先执行获准的前 5,000 步，保留完整固定 horizon 以便正常恢复。

检查：真实吞吐/显存/磁盘、加载等待、loss/梯度有限、free-generation exact/sign/magnitude、high/low、invalid 和 train/val 差距。无数据/实现错误且得到继续授权后，再完成剩余预算；不因状态正常而自动多跑 seeds 或第二种切分。

工程验收要求流程与可复现性正确；科学验收报告准确率和误差类型，不预设模型必达 98% 或 100%。若 train 拟合好但 val 差，先分析轨道和稀有 token，不直接增加模型；若 train 也差，先核对编码、梯度和优化器。

### 阶段 F：冻结模型选择并独立测试

基于 full val 冻结 best 与协议，经批准对 test 进行一次独立最终评估，保存完整预测、分组指标与结论边界。小数据可保存全部预测，但不得把报告写入 checkpoint 或代码目录覆盖既有文件。

以后如增加 42/43/44 三个训练 seeds，固定同一已发布 split，报告每个 seed 和均值/离散性，不只挑最优种子。切分泛化需另设 split seeds，对应新的实验版本。

## 12. 风险与完成判据

| 风险 | 必要防护 |
| --- | --- |
| 旧目录名造成错误圈数 | 已更名 hexagon_w8_MHV；所有 runtime metadata 明确 loop4/weight8，模块名 w8_0y |
| 把 243,000 项全当作 0y train | 必须过滤并核对 11,208 |
| 原始分式、C4 和二次乘 32 混用 | hash + scale 契约 + 逐行回查 |
| 把 NUM_0 当 PAD 或禁止固定前导零 | fixed2 独立版本与完整边界测试 |
| 旧 mask 或 loss shift 泄漏答案 | 显式 [1,9)、四位置监督和负对照测试 |
| 把两块 base100 当字符串拼接 | 解码严格使用 100q+r |
| token19/1920 训练样本稀缺 | 覆盖审计、单轨道处理、单列指标；不暗改 split |
| 旧学习率隐式随宽度变化 | 同时记录配置/实际 LR 与角色 |
| 验证子集 4096 超过新 val | 新配置 512；所有选择有界检查 |
| token accuracy 被高位零/PAD 抬高 | 以自由生成整数 exact 为主 |
| 旧 checkpoint 或坐标模式误加载 | task/base/fixed2/ID/scale/split 等全契约拒绝 |
| 日志仍写 B×3、旧样本数或旧角色 | 从新 encoding/run metadata 生成报告 |
| 评估打乱 RNG、恢复重复/漏数据 | 连续/分段下一批及下一步更新对照 |
| 不断刷新 best 用尽磁盘 | 首个 checkpoint 实测、保存事件预算、磁盘检查；不自动删旧文件 |
| 把旧运行/计划当新结果 | PROGRESS 区分 written / executed / passed；报告绑定本次源码/数据哈希 |

完整实施后的交付应包括：隔离的新模块、明确的数据/编码契约、训练数组与审计、显式配置、通过的测试记录、可恢复的运行记录、full val 选择的 checkpoint，以及获准后独立的 test 报告。**本文完成不等于上述实施已经完成。**

## 13. 审阅与实施前最终确认清单

- [x] 首轮模型 4 层／256 宽度／4 heads，KV heads=4。
- [x] C4=32c4，当前源文件已存在，原始分式保留。
- [x] base-100 固定两块、符号在前、EOS 在后。
- [x] 用户已授权并准备本地冒烟候选代码；不执行本地测试/训练。
- [x] 主 split 确认 random-row，seed=42。
- [ ] 确认建议 batch=128、三组学习率和 500-pass 固定 horizon／初段 5,000 步。
- [x] 工程快照与新模块冒烟闭环已写；静态核对不等于运行通过。
- [x] 取消独立 CPU 测试/核验阶段；实际模型检查仅 CUDA。
- [ ] 待用户提供服务器后确认路径/环境/数据发布/GPU smoke 与训练的各阶段执行授权。
- [ ] 明确最终 test 的评估时机、是否追加多 seed 或对照实验。

## 14. 依据与参考

- [w6 工程入口与历史结果边界](../heptagon_w6_MHV/README.md)、[配置](../heptagon_w6_MHV/nanoinfra-main_symbol/projects/heptagon_symbol/configs/base.yaml)、[训练循环](../heptagon_w6_MHV/nanoinfra-main_symbol/projects/heptagon_symbol/train.py)、[checkpoint](../heptagon_w6_MHV/nanoinfra-main_symbol/projects/heptagon_symbol/checkpoint.py)。
- [w8 项目状态](../heptagon_w8_MHV/README.md)、[数据方案](../heptagon_w8_MHV/W8_SERVER_DATA_PLAN.md)、[八槽 tokenizer](../heptagon_w8_MHV/nanoinfra-main_symbol/projects/heptagon_symbol_w8/tokenizer.py)。w8 代码未运行验证的状态不能省略。
- [共享 GPT](../heptagon_w6_MHV/nanoinfra-main_symbol/core/model/gpt.py)、[优化器实际 LR 缩放](../heptagon_w6_MHV/nanoinfra-main_symbol/core/training/optim.py)、[监督 shift](../heptagon_w6_MHV/nanoinfra-main_symbol/core/data/supervision.py)、[word mask 工具](../heptagon_w6_MHV/nanoinfra-main_symbol/projects/amplitude_symbol/blocks/attention.py)。
- [四圈原始 symbol 说明](../Symbol_Data/hexagon_mhv_4loop_native/README.md)、[x32 整数版说明](../Symbol_Data/hexagon_mhv_4loop_native_x32/README.md)、[缩放清单](../Symbol_Data/hexagon_mhv_4loop_native_x32/scaling_manifest.json)。0y/轨道/进制统计来自对该版本的全量只读扫描，后续转换必须复核并写入正式 audit。
- [Cai 等，Transforming the Bootstrap，arXiv:2405.06107v2](https://arxiv.org/html/2405.06107v2)：借鉴符号序列与整数序列的监督任务表达。本文的固定两块 base100、学习率、切分、预算均为本项目设计，不冒充论文验证结论。
