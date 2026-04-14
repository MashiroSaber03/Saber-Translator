# 漫画分析角色卡兼容性回归清单

## 目标范围
- 角色卡标准：SillyTavern `chara_card_v2`（V2 稳定）
- 扩展目标：Tavern Helper（通过 `data.extensions.saber_tavern` 读取增强载荷）
- 导出载体：单角色 PNG（`chara` 文本块 + `ccv3` 镜像）

## 核心字段检查
- `spec == chara_card_v2`
- `spec_version` 以 `2` 开头
- `data.*` 必填字段完整（name/description/personality/scenario/first_mes/mes_example/creator_notes/system_prompt/post_history_instructions/alternate_greetings/tags/creator/character_version/character_book/extensions）
- `alternate_greetings` 为数组
- `tags` 为数组

## 世界书检查（`data.character_book`）
- 全局字段：`name/description/scan_depth/token_budget/recursive_scanning/extensions/entries`
- 每条 entry 至少包含：
  - 触发键：`key`（兼容镜像 `keys`）
  - 内容：`content`
  - 条目标识：`uid`（兼容镜像 `id`）
- 兼容镜像字段可选：`secondary_keys/priority/case_sensitive/name`

## 扩展区检查（`data.extensions.saber_tavern`）
- `regex_profiles` 为数组，规则项包含 `id/pattern/replacement/flags`
- `mvu.variables` 为数组，变量项包含 `name/type/scope/default/value/validator/description`
- `ui_manifest` 为对象，`panels/widgets/actions/events/bindings` 为数组
- `import_manifest` 为对象，包含 `version/requires/activate_steps/fallback_behavior`

## PNG 封装检查
- PNG 元数据存在 `chara` 字段
- `ccv3` 字段镜像存在（兼容读取器）
- 写入后执行 `PNG -> JSON` 回读一致性校验，失败则拒绝导出

## 批量导出检查
- 同名安全文件名冲突时自动避让（示例：`A_B.png`, `A_B_2.png`）
- ZIP 中每个 PNG 可独立回读并匹配原角色名
- 后端 compiled/png 存储采用防冲突文件名（安全名 + 哈希）

## 前端交互检查
- 角色卡工坊 Tab 可用
- 字段锁可阻止手动编辑/批量编辑/预设覆盖
- 批量编辑可作用于勾选角色集合
- 模板预设可作用于当前角色或勾选角色
- 编译面板可显示错误、警告、兼容性诊断

## 手工导入回归（建议每次发布前执行）
1. 导入 PNG 到 SillyTavern，验证核心字段可用。
2. 检查角色内嵌世界书是否可识别触发。
3. 未安装 Tavern Helper 时，聊天与角色基础能力不受影响。
4. 安装 Tavern Helper 后，确认 regex/mvu/ui_manifest 可被读取并激活。
