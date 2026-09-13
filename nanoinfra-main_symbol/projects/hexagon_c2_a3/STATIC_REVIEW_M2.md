# M2/38 本地静态核对记录

2026-09-14。**候选实现已写，未执行验证。**
本轮只阅读代码/配置、人工检查接口和下标、检索残留常量、检查文件差异与哈希。
未运行 Python 导入、语法编译、CPU/MPS测试、数据转换、模型前后向或服务器计算。

## 已人工核对

- source封装未修改的历史审计读取器，核对固定哈希、全源243,000行、y分层与完整直方图；
  只有全源通过后，缺失合法0y项才解释为零。C4不再次乘32。
- 40,776相邻目标、13,338输入、零冲突及已知分布已写成发布断言，断言尚未运行。
- C36行优先、含零/重复；第二遍源读取后逐目标独立重建，不把同一缓存自比较当证明。
- random-row seed42，32,620/4,077/4,079；保留全部目标行，输入键仅(C36,y_types)。
- 只读mmap/无pickle/固定schema/可信metadata摘要；旧0y或40项协议不得混用。
- 模型输入API不接收位置、背景word或标签；RoPE保留，审计metadata不进入token。
- vocab117/types3；108个条件token占5…112，ANSWER在113，目标在114…117，PAD在118…127；
  shift后仅113…116监督，CE只对4个有效token取平均。
- prefix [0,114)双向、答案因果；有效查询不看PAD，PAD查询有可见键；
  cache首次完整prefill114，后续单token，不用分块prefill/static-cache。
- 4/256/4/4；静态公式 `12*4*256**2+(2*117+3)*256=3,206,400`，GPU构建时再断言。
- smoke20/batch8、tiny1000/batch32、pilot5000/batch64分开配置，不自动启动后两阶段。
- 实际输入127B（含PAD）/监督4B/nominal128B；pilot完整退火，不继承旧35024步horizon。
- AdamW角色unembedding/embedding/matrix，峰值LR乘sqrt(3)、eps=1e-10；运行时复核实际值。
- sampler跨epoch拼接、不drop-last；恢复保存下一个未读行，检查实际下一批数据。
- RNG用JSON规范化列表，避免tuple/list差异误报；恢复覆盖参数、优化器、RNG与sampler。
- 基线只从完整train拟合，无位置键；报告覆盖率/covered正确率，未覆盖回退train众数。
- 全词表自由生成、严格sign/high/low/EOS；负零invalid、合法0单独计错；
  seen/unseen、位置、y类型、宏平均、稀有/空分层已接入；仅full-val选best，test显式门禁。
- checkpoint绑定独立版本、数据、编码、mask、模型/头、实际代码依赖、配方与环境；
  不复用旧权重/runner。输出独立新目录、写一次、staging发布，失败保留。
- 共享core、amplitude attention、旧0y任务、源Symbol_Data与历史audit脚本未修改；
  公共打包文件只增加M2包/YAML条目。

文件差异核对：受保护的已跟踪代码没有Git差异；新增文件的diff空白检查没有告警。
两个源文件摘要与计划中的固定值一致；历史audit脚本仍为
`17169844b3f5e7c89bcee20cbdddeaf968478cebcb180209b1bfafa3b3041eaa`。

## 必须等 GPU 的验证

Python导入与依赖API兼容、数据发布断言、CUDA编码/监督、无泄漏正负对照、有限梯度、
4-token平均loss、cache/full生成、DCP/AdamW恢复、20 vs 10+10闭环**均尚未运行**。
没有新增训练指标、实际切分覆盖率、GPU耗时或显存结论。

GPU检查代码不是“已通过”的依据。只有服务器命令退出0并产生完整passed_gpu_smoke报告，
才能更新状态；tiny-overfit、pilot与最终test还需各自验收。
旧0y成功或历史数据审计不能替代本新任务的执行证据。
