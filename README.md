# Hexagon 四圈 0y 系数预测

状态（2026-09-13）：**本地冒烟候选代码已写，已做人工静态核对；尚未运行验证。**
没有执行本地 Python/CPU 测试、数据转换、推理、训练或 GPU 服务器运行。
按用户确认，目录由 `hexagon_w6_MHV` 改名为 **`hexagon_w8_MHV`**；本任务从 **loop=4、weight=8** 开始。
代码仓库：[xghf233/hexagon_w8_MHV_1](https://github.com/xghf233/hexagon_w8_MHV_1)。

基于成熟的 Heptagon w6 快照，新增独立模块
`nanoinfra-main_symbol/projects/hexagon_symbol_w8_0y/`，保留公共 `core/` 和已有提取工具。

| 项目 | 本版锁定值 |
| --- | --- |
| 目标 | 普通 BDS-like E 四圈完整 symbol 的 0y 非零整数系数 C4=32c4 |
| 数据 | 已有 x32 gzip 的 243,000 行中筛选 11,208 行；不再次乘 32 |
| 输入 | 8 个原子字母，ID 0–5；全局保留 9 字母词表 |
| 切分 | random-row，PCG64 seed=42；train/val/test = 8,966/1,120/1,122 |
| 编码 | sign + base100 高块 + base100 低块 + EOS；固定两个数值块 |
| 模型 | 4 层 / 256 宽 / 4 heads / 4 KV heads；静态计算 3,205,376 参数 |
| 序列 | 16 槽、词表 115、3 种 token type；word 双向区间 [1,9) |
| 执行 | Linux 单张 bf16 CUDA GPU；无 CPU/MPS 模型回退；默认 smoke、compile=false |

当前源数据位置为同级 `Symbol_Data/hexagon_mhv_4loop_native_x32/`。训练数组尚未生成，
由之后服务器上的 GPU 冒烟入口统一准备。切分规则已锁定，具体稀有项落在哪个 split
要等转换后查看 `audit.json`；不为六条 1920 样本暗改 random-row。

- [服务器交接与命令模板](SERVER_HANDOFF.md)：待用户提供服务器后执行，本次不连接。
- [静态核对与未验收项](STATIC_REVIEW.md)：区分“已写”与“GPU 已通过”。
- [新模块协议](nanoinfra-main_symbol/projects/hexagon_symbol_w8_0y/README.md)：数组、token、loss、配置。
- [完整改造计划及当前范围](HEXAGON_W8_0Y_ADAPTATION_PLAN.md)。
- [快照来源与排除项](SNAPSHOT_PROVENANCE.md)。

新增独立任务（2026-09-14）：[相邻 y 的 M2/38 系数预测](nanoinfra-main_symbol/projects/hexagon_c2_a3/README.md)。
其本地候选实现已写、尚未执行 GPU 验收；与本页旧 0y 任务共享 GPT 类，不共享权重。
新任务请使用自己的交接文档；本页原有 0y 配置和历史状态保持不变。

下一阶段只先运行交接文档中的 GPU smoke。它不自动启动 tiny-overfit、正式训练或最终 test。
random-row 允许同一 S3 轨道跨 split，结果不能称为 orbit-grouped 泛化。

Symbol_Data 的历史提取说明/清单保留当时的旧目录名，不改动哈希锁定的原始数据与工具。
如需参考旧提取命令，把其中 `hexagon_w6_MHV/tools/` 解释为现有 `hexagon_w8_MHV/tools/`；
这不代表允许本地重跑提取或测试。
