# ADR-0001：v2 进程拓扑与导入边界

- 状态：已接受
- 日期：2026-07-28
- 对应方案：§4、§14 阶段 0

## 决策

正式 v2 使用单一可执行入口和三个角色：

- Launcher：唯一解析 data root、持有单实例锁、执行迁移与恢复裁决、监护子进程。
- API：只注册 `/api/v2`，负责 REST/SSE/媒体、CPU render 和远程 operation。
- Worker：负责全局 job 队列、本地模型 operation、插件和 Chroma。

`saber_v2.py` 必须先执行 `multiprocessing.freeze_support()`、解析 `--role`，再导入角色模块。API 导入图禁止出现 `torch`、`torchvision`、`chromadb`、`src.plugins`、`src.interfaces`、legacy `app/src.app` 或高层模型/渲染模块。`src/backend_v2/__init__.py` 保持无副作用。

生产 API/Worker 只能接受 Launcher 签发的 epoch id/token；直接启动只允许显式 `--test-mode`。API 与 Worker 的业务协作只通过 SQLite 和文件资产，Launcher 的环境变量签发和进程监护仅属于控制面。

## 后果

- legacy `app.py` 在阶段 6 前继续作为独立行为对照入口，但不得被 v2 导入。
- 打包阶段收集 Worker 依赖不代表 API 运行时可以导入这些依赖；源码态与打包态都要运行 import-graph 探针。
- 子进程异常恢复只能由 Launcher 裁决，API/Worker 不能自行抢占新 epoch。
