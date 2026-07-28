# ADR-0002：SQLite 事实层、不可变资产与发布协议

- 状态：已接受
- 日期：2026-07-28
- 对应方案：§5、§11、§12

## 决策

- 结构化事实使用 SQLAlchemy 2 + SQLite，升级使用 Alembic。
- SQLite 固定开启 WAL、foreign keys 和 busy timeout；模型、网络、解码和压缩不得位于数据库事务中。
- 图片及大型文件以不可变 asset 保存到 `objects/{asset_id前两位}/{asset_id}.{ext}`；数据库仅保存规范化相对路径和明确 FK。
- 页面 current 资产通过 `page_assets(page_id, role)` 唯一指针发布，旧资产是否回收只由引用图和 GC 决定。
- 跨文件系统/SQLite 的创建使用 `object_commit_journal`：staging 写完并 fsync，原子改名到 objects，短事务建立 asset/业务引用，最后删除 journal。
- 所有发布同时验证 source/document revision 与 producer job/operation/render attempt fencing；过期执行器可以完成计算，但不能发布。

## 删除与保留

- 正式业务资产关系使用 `RESTRICT`，先由领域命令解除引用再异步 GC。
- 纯结构从属使用 `CASCADE`。
- producer 历史引用使用 `SET NULL`，清理历史不能删除当前资产。
- job/operation 对不可变输入和版本使用真实 FK，不以 JSON 中的 id 代替可达性。

## 后果

快速工作区转正、书籍改名或章节移动只改变数据库关系，不移动 objects。业务删除提交后旧媒体 URL 立即失效；物理文件允许在 GC 宽限期内暂留。
