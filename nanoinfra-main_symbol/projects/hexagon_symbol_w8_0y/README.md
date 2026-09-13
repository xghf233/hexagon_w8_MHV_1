# Hexagon w8 0y：数据与训练协议

当前状态：已写代码，人工静态审阅，GPU 未执行。入口与边界见根目录
[SERVER_HANDOFF.md](../../../SERVER_HANDOFF.md)。历史 Heptagon 模块不是本项目入口。

## 数据与 token

直接读取原生 x32 JSONL gzip 的整数 `numerator`，要求 `denominator="1"`，不再缩放。
原子字母 `a,b,c,mu,mv,mw,yu,yv,yw` 对应 ID 0…8 和
`hat_a,hat_b,hat_c,hat_d,hat_e,hat_f,y_U,y_V,y_W`；本阶段 word 长8、仅允许0…5。

转换格式：`words:uint8[11208,8]`，`coefficients:int16[11208]`，
`source_row_ids:int32[11208]`，`orbit_ids:int32[11208]`。
文件显式使用小端整数，`splits.npz` 保存小端 int32 的 train/val/test 索引。
索引指向过滤后的数组，`source_row_ids` 才指向完整243,000行源文件。
PCG64(seed42)排列后依次取8966/1120/1122行，各 split 内按数组行 ID 排序，禁止改动成员。

| token | ID |
| --- | --- |
| 9种原子字母 | 0…8 |
| BOS / COEFF / EOS / PAD | 9 / 10 / 11 / 12 |
| PLUS / MINUS | 13 / 14 |
| NUM_0 … NUM_99 | 15…114 |

序列固定16槽：

```text
位置      0    1 … 8    9       10    11    12   13   14  15
内容      BOS  word    COEFF   sign  high  low  EOS  PAD PAD
raw loss   0   0 … 0    0       1     1     1    1    0   0
```

解码 `C4 = sign * (100*high + low)`；小数值也保留 high=0。
词表大小115，类型：word=0、control=1、coefficient=2。
模型每次输入15个槽，shift后 `targets[:,9:13]` 是唯一监督区间，EOS也受监督，PAD不受监督。
word 双向注意力只覆盖 `[1,9)`；COEFF 和答案保持因果，不能用旧 `[1,11)` 默认值。

实际源样本 `a,a,mv,mu,mu,b,b,mu` 的 C4=1：

```text
word IDs: [0,0,4,3,3,1,1,3]
tokens:   [9,0,0,4,3,3,1,1,3,10,13,15,16,11,12,12]
answer:   PLUS NUM_0 NUM_1 EOS       物理系数 c4=1/32
1920:     PLUS NUM_19 NUM_20 EOS     仍然恰好两个数值块
```

系数非零，范围−156…1920，共59个值；1920六条保留。编码器支持|C|<10000（含正零语法，
但数据集拒绝零标签）；负零、越界、变长答案、提前/缺失 EOS 都拒绝。生成器仅输入十token prompt，
无约束 greedy 最多四步；不将真实符号/高位作为生成输入。

## 配置核对

| 配置 | samples/step | total_batch_size名义token | max_steps | warmup | val子集 |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke（默认） | 8 | 128 | 20 | 2 | 8 |
| tiny_overfit | 32 | 512 | 1000 | 20 | 32 |
| w8_0y_random（正式候选） | 128 | 2048 | 35024 | 200 | 512 |

全部单卡、累积1、sequence_len16、compile=false、4/256/4/4、head_softcap15。
参数量公式 `4*(12*256²) + (2*115+3)*256 = 3,205,376`（GPU入口将实测）。
Fused AdamW，betas(0.9,0.95)，weight_decay0.01，clip_grad_norm1.0，core eps1e-10。
原始 LR：matrix0.0002 / embedding0.14 / unembedding0.0028；
core 按宽度再乘 sqrt(3)，实际峰值约0.000346410 / 0.242487113 / 0.004849742。
日志角色顺序是 unembedding、embedding、matrix，不能按位置误标。

实际输入token计数为15×batch，有效监督为4×batch；`total_batch_size`不是样本数。
正式候选每500步全val，save_every2500且best改善也保存；只按full-val exact选best，
同分保留更早模型。smoke/tiny不更新full-val best。学习率与正式预算尚未经训练验证。

所有训练调用必须显式指定 `data_dir`、`metadata_sha256`、`output_dir`。
数据/模型/编码/mask/代码哈希/实际环境绑定 checkpoint；不支持加载旧Heptagon/base1000权重。
评估提供 exact/sign/magnitude/high/low/invalid、逐系数/逐绝对值和三个稀有组指标；空组返回null。
宏平均、orbit lookup对照、报告绘图等扩展尚未实施，本版定位为GPU冒烟闭环。
