# ADR-0003：任务、operation、租约与章节写入屏障

- 状态：已接受
- 日期：2026-07-28
- 对应方案：§4.4、§7

## 执行模型

- 所有长任务进入一个持久全局 job 队列，同一时刻最多一个 current job。
- 保存型即时工作进入 operations；内部文字渲染进入每页唯一的 render_requests。
- Worker job、Worker/API operation、render 和 transient 每次领取都创建 attempt id/lease token，并绑定执行进程 epoch。
- 检查点、事件、资产、current 指针和终态写入必须同时通过状态、attempt、lease、epoch 与业务 revision fencing。

## 续租失败

heartbeat 的条件 UPDATE 影响 0 行即代表执行权已经丢失：

- 进程 epoch 失租：立即关闭全部领取和后继步骤准入、毒化本进程 attempt、停止发布并受控退出，等待 Launcher。
- 单 attempt 失租：只毒化该 attempt，丢弃迟到计算结果，调度器重读数据库。
- 失租执行器无权自行裁决业务终态。

## 章节写入意图

队首章节写 job 不直接拿锁等待即时写入。它先为全部目标章节原子建立 `chapter_write_intents`，封闭新页面写入、写 operation/render 链和 import lease；意图建立前的 active 写链允许排空。排空后，在一个 `BEGIN IMMEDIATE` 中验证 intent set/generation/epoch/lease，原子创建所有章节写锁、删除意图、创建 job attempt 并执行 queued→running。

queued 取消释放意图；旧 Worker epoch 恢复清理旧意图但保留 queued job。paused/interrupted 的正式章节写锁继续保留，只有取消或终态释放。

## 后果

持续编辑不能造成队首写任务饥饿，也不存在“job 持锁等待 pending operation”的死锁。浏览器依据 intent/lock 事实切只读，但 SSE 事件不是准入事实源。
