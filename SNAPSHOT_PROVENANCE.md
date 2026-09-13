# 框架快照来源

日期：2026-09-13。

来源：同级 `heptagon_w6_MHV` 的已提交快照
`6f9aae9d5366b9a62e87c6b800360a8ead2926bc`，使用 `git archive` 按白名单复制，
不是复制整个含用户临时文件的工作区。旧项目的未跟踪 `training_curves.png` 原样保留，未带入。

复制白名单：根 `LICENSE`，`nanoinfra-main_symbol/{core,pyproject.toml,LICENSE,README.md}`，
`nanoinfra-main_symbol/projects/{amplitude_symbol,heptagon_symbol}`。
排除 `*/reports/*`、`*/results/*` 和 PNG；未带入 `.git`、`.venv`、数据、checkpoint、日志或缓存。
历史模块源码和测试仅用于追溯/依赖，其中的运行成绩、CPU测试指令不适用于当前任务。
排除报告后，历史文档的结果链接可能没有对应文件；请回到原 w6 仓库查历史证据。

本次新增适配：`projects/hexagon_symbol_w8_0y/`、根项目文档和 `.gitignore`；
仅在新副本修改 `pyproject.toml` 的任务描述/打包范围以及 edition README。
原 `heptagon_w6_MHV`、`heptagon_w8_MHV`、HZQ-git 未修改。
新目录原先已有的 `tools/` 与 Symbol_Data 源数据不改动，不重做提取或整数缩放。

静态文件比对：复制后的 `core/` 与旧 w6 逐文件相同；
`projects/amplitude_symbol/blocks/` 也逐文件相同。项目按显式 `[1,9)` 参数复用 mask 工厂。
复制源 commit 是文件身份锚点；实际 GPU 运行另将关键源码、配置和共享依赖的 SHA-256
写入 `environment.json` / checkpoint contract，不能把历史快照的运行结果归给新模块。

用户随后确认将本目录从 `hexagon_w6_MHV` 更名为 `hexagon_w8_MHV`，并授权建立独立的
初始提交，发布到 `https://github.com/xghf233/hexagon_w8_MHV_1`。不导入旧 Heptagon Git 历史，
来源 commit 仅作为快照追溯锚点。具体发布状态以本仓库 Git 日志与远端分支为准；
GitHub 代码发布不代表 GPU 服务器数据上传或运行验收。
