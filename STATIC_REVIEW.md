# 本地静态核对记录

日期：2026-09-13。结论：已形成可交接的 **GPU 冒烟候选版本**，不是已验收训练版本。
本轮采用阅读、文本检索、文件差异与哈希比对；没有运行 Python、语法编译、CPU 单测、
数据转换、模型前向/反向或 GPU 程序。因此没有新增测试成绩或训练指标。

## 已对照代码的事项

- 源：锁定已有 x32 gzip 与 scaling manifest 的 SHA-256；本轮读取文件摘要与锁定值相符。
  新转换器只筛0y并直接读取整数分子，要求denominator="1"，不二次乘32。
- 数据：转换器、Dataset 与配置共同锁定 loop4/weight8、11,208行、8槽、0…5 ID、非零int16、
  范围−156…1920；转换器还检查59种值、66条高位非零、六条1920。
- 切分：random-row PCG64 seed42，8966/1120/1122；持久化并校验完整分区。
  S3轨道只审计，不分组或增强。当前没有运行划分，稀有数字在各split的数量不作猜测。
- 编码：fixed2 base100、vocab115、type3、BOS/COEFF位置0/9，答案10…13；
  shift后监督9…12。PAD和NUM_0不同；物理c4与训练C4分开。
- 模型：配置与构建器强制4/256/4/4，head_dim64；参数量静态公式3,205,376，GPU入口将断言。
  显式word mask `[1,9)`，恢复后重新安装并在CUDA比对，避免旧十字母默认值。
- 配置：default smoke20步、batch8、val8；tiny32样本；正式候选batch128、val512、35024步。
  每步输入15B、监督4B；nominal total_batch_size=16B，累积1。
- 优化器：三组配置LR与core的sqrt(3)缩放一致，角色按真实顺序记录；full-val才更新best。
- checkpoint：新版本/数据/编码/模型/代码与环境严格绑定，保留优化器、RNG、下一行游标，
  输出采用新目录、staging发布。连续20与10+10对照维持同一horizon。
- 评估：十token prompt、四步无约束生成，无真实答案输入；strict decode、invalid/zero区分；
  test需显式opt-in，默认smoke不会使用test做模型评估。
- 运行边界：Linux单张bf16 CUDA GPU，无CPU/MPS模型回退；主机I/O/格式处理不可避免但不设CPU核验阶段。
- 工程隔离：core与共享attention副本未变；原w6仍仅有既存未跟踪training_curves.png，未写旧项目。

## 写入但等待 GPU 验收

`gpu_smoke.py` 包含真实数据发布、CUDA编码/监督检查、无泄漏正负对照、KV cache生成对照、
有限梯度检查、loader恢复和40次总更新的连续/分段恢复核对。检查代码本身也尚未运行，
不得据此宣称“验证通过”。依赖兼容、DCP实际恢复、bf16容差及显存/吞吐只有服务器执行后才能确认。
旧文档的PyTorch2.12.1+cu130/NumPy2.4.6只是历史环境线索；本次未探测服务器或安装依赖。

## 尚未做的工作

本地/服务器训练数组均未由新转换器生成；未执行smoke、tiny-overfit、compiled smoke、
正式训练和最终test；未完成全量可积性验证。orbit-grouped协议、宏平均/轨道查表基线、
报告绘图和完整坏输入测试矩阵属于后续扩展，不冒充当前冒烟闭环已覆盖的功能。
GPU失败时应保留输出并分析原因，不自动调整数据划分、模型规模、容差或预算。
