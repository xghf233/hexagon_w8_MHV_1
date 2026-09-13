# Hexagon C₂ → A₃：相邻 y 的 M2/38 实验

当前计划：[四圈相邻 y、38 项输入的 M2 小实验](PLAN_M2_ADJACENT_38.md)。
模型仅接收 36 个有序精确系数和两个 y 类型；位置保留在数据中作审计/分层评估，不进入 prompt 或查表键。
状态（2026-09-14）：**首版本地候选代码已写，已人工静态核对；尚未运行验证。**
没有运行 Python 导入/语法编译、CPU/MPS 测试、训练数组转换或 GPU 训练。历史输入审计不是新实现通过的证据。

本模块与旧 0y 任务并列，复用公共 GPT/LMSystem、优化器、调度器、DCP 和通用 mask factory。
不导入旧任务的 tokenizer、数据、runner、模型适配器或 checkpoint；不修改公共 core。
每个任务从头初始化，使用各自的数据与运行目录，旧 0y 权重不用于本实验。

- [服务器交接与分阶段命令](SERVER_HANDOFF_M2.md)
- [静态审阅与 GPU 待验收项](STATIC_REVIEW_M2.md)
- [完整实验计划](PLAN_M2_ADJACENT_38.md)

## 当前实现协议

| 项目 | 锁定值 |
| --- | --- |
| 真值 | 同一四圈普通 BDS-like symbol 的 C4=32*c4；不再次缩放 |
| 目标池 | 40,776 个非零相邻 2y word；预期 13,338 个 input38，无冲突 |
| 条件 | 精确 0y 查表，36 个槽全部保留；不存在的合法 0y 项为零 |
| 模型特征 | conditions[36] + y_types[2]；不含背景、位置、来源 ID 或分组 ID |
| 切分 | PCG64 seed42 random-row；32,620 / 4,077 / 4,079，不做分组切分 |
| 数字 | sign + base100 高块 + 低块；目标最后加 EOS |
| token | vocab117、types3、prefix114、总长128；shift 后仅 [113,117) 监督 |
| 模型 | 4/256/4/4，预计 3,206,400 参数，prefix-LM [0,114) 双向 |
| smoke | batch8，同一20步 horizon，连续20对照10+10；共40次更新 |
| tiny | 32个不同 train 输入，最多1000步，要求自由生成全部正确 |
| pilot | batch64、5000步、warmup200、末20%退火；平均呈现9.81次 |
| 执行 | Linux 单张可见 bf16 CUDA GPU，compile=false；无CPU/MPS模型回退 |

序列为：

```text
[BOS][QUERY][y_alpha][y_beta][CONTEXT] (sign high low)×36
[ANSWER][target_sign][target_high][target_low][EOS] [PAD]×10
```

物理 y 位置仅用于构造条件和位置分层评估；RoPE 仍用于识别有序 token 槽。
数据发布器核对固定源哈希/全源直方图，独立逐目标查验36项，并逐数组读回后才发布。
每次消费数据必须提供可信 metadata SHA-256，不能因报错而重新计算一个摘要绕过检查。

评估同时给出行加权、input-group 宏平均、seen/unseen-input38、位置/y类型/稀有系数等指标。
众数和精确输入查表基线只从完整 train 拟合；查表键也不含位置。
random-row 的高总分不能直接解释为新输入泛化；空分层报告 count=0/null。
最佳 checkpoint 只由 full-val exact 选择，并列保留较早步数。test 需显式 opt-in。

默认入口是 `projects.hexagon_c2_a3.gpu_smoke`，不是 pilot。
本次没有提交/推送代码、上传数据、连接服务器或创建环境。

## 历史：已完成的纯数据表示审计

以下描述已有 `audit_representation.py`，不是当前训练入口。它保持原样，其已获准的本机 CPU
例外仅用于此前完成的审计，不延伸到新转换器或模型执行，也不表示现在应重跑。
该历史脚本只依赖标准库；**新训练任务**需要现有项目的 PyTorch、NumPy、Hydra/OmegaConf。

- 现有四圈 x32 symbol 中的 110,082 个非零 2y 目标；零目标不在本次范围。
- 36 个条件来自同一完整 symbol 的精确 0y 查表，包括零和重复数，不再次乘 32。
- 行优先排列：第一个 y 的替换决定行，第二个决定列；普通字母顺序固定为带帽 a…f。
- 审计 38 项（36 系数 + 两个有序 y 类型）和 40 项（再加两个 y 位置）。
- 精确元组分组，区分重复与冲突；输出行加权可辨识性上限、全部输入组及冲突见证。
- 不生成训练切分，不使用多数标签修改真值，不启动任何模型计算。
- 输出必须是代码仓库与 Symbol_Data 之外的新目录；现有输出不覆盖。

调用 `audit_representation.py --source <x32-jsonl.gz> --output <new-directory>`。
脚本启动和结束都核对固定源文件哈希，并保存脚本哈希与执行环境。
上述历史产物未被新实现修改；新数据发布仍从原始 x32 源独立构造。
