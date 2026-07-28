# Saber Translator Plugin v3

> 最后更新：2026-07-28
> v3 与旧插件不兼容。旧的 `PluginBase`、浏览器 Pipeline API、Base64
> payload 和直接放目录热加载方式均已废弃。

Plugin v3 是后端优先架构的一部分。浏览器只管理插件元数据和显示任务进度；
插件代码只由 Worker 加载。创建任务或 Worker 操作时，后端会冻结当前启用插件
的不可变版本 ID 和配置，因此浏览器关闭、插件开关变化、配置修改或新版本导入
都不会改变已经入队的任务。

## 包结构

插件以 ZIP 导入，根目录必须包含静态清单 `plugin.json` 和清单指定的 Python
入口文件：

```text
my_plugin.zip
├── plugin.json
├── plugin.py
└── helpers.py          # 可选；建议从入口使用相对导入
```

导入后，后端把包发布到受管理的不可变版本目录：

```text
data-v2/plugins/{plugin_id}/versions/{plugin_version_id}/
```

不要直接修改该目录。更新插件时重新导入 ZIP；即使
`package_version` 文本相同，也会生成新的内部 `plugin_version_id`。

## 最小清单

```json
{
  "schema_version": 3,
  "plugin_id": "translation_marker",
  "display_name": "翻译标记",
  "package_version": "1.0.0",
  "entrypoint": "plugin.py:Plugin",
  "hooks": ["after_translate"],
  "supported_steps": ["translate"],
  "supported_modes": ["standard", "hq"],
  "priority": 100,
  "failure_policy": "continue",
  "author": "Your Name",
  "description": "给普通译文追加可配置标记",
  "default_enabled": false,
  "config_schema": {
    "suffix": {
      "type": "text",
      "default": " [checked]"
    }
  }
}
```

字段约束：

- `schema_version` 必须为 `3`。
- `plugin_id` 只能包含字母、数字、下划线和连字符，且必须稳定唯一。
- `entrypoint` 使用 `相对路径.py:类名`。
- `priority` 越小越先执行；同优先级按 `plugin_id` 稳定排序。
- `failure_policy` 为 `continue` 或 `fail`。
- `config_schema` 支持 `text`、`number`、`boolean`、`select`；配置由后端
  校验、版本化，并在任务入队时冻结。
- `default_enabled` 是持久启动默认值；`runtime_enabled` 是本次后端进程
  的临时开关，重启时从默认值恢复。

## Hook

所有步骤都有 `before_*` 和 `after_*`：

| 粒度 | step |
|---|---|
| 每个章节 job 一次 | `job` |
| 每个完整流水线一次 | `pipeline` |
| 每页或单次操作 | `detect`, `ocr`, `color`, `translate`, `ai_translate`, `inpaint`, `render` |

完整 Hook 名例如 `before_job`、`after_pipeline`、`before_ocr`、
`after_ai_translate`。模式名固定为 `standard`、`hq`、`proofread`、
`remove_text`。

入口类不需要继承项目内部基类：

```python
class Plugin:
    def after_translate(self, context, data):
        result = dict(data)
        suffix = str(context.config.get("suffix", ""))
        translations = result.get("translations")
        if isinstance(translations, list):
            result["translations"] = [
                f"{value}{suffix}" for value in translations
            ]
        context.logger.info("translation marker applied")
        return result
```

每个 Hook 的签名是：

```python
hook(context, data) -> dict
```

必须返回 JSON 兼容对象。不要返回 `None`，也不要原地修改后不返回。

## Context

`context` 提供：

- `job_id`, `batch_id`, `book_id`, `chapter_id`, `page_id`
- `mode`, `step`, `scope`
- `config`：该任务/操作冻结的插件配置
- `repository`：只读领域查询，例如 `get_page()`、`get_bubbles()`
- `assets`：只读资产元数据与受大小限制的 `read_bytes(asset_id)`
- `logger`：`info()`、`warning()`、`error()`，输出进入持久任务/操作事件

插件不得依赖浏览器状态，不得访问用户在前端内存中的图片或翻译结果。图片和
其他二进制数据始终使用 `assetId` 等资产引用传递。

## 数据与失败语义

- Hook 输入和输出都经过结构校验；禁止 bytes、Base64、data URL 和伪装的
  Base64 大字符串。
- `failure_policy: continue`：记录事件警告，保留进入该插件前的数据并继续。
- `failure_policy: fail`：
  - `before_job` / `after_job` 失败整个 job；
  - pipeline 或原子 Hook 失败当前任务项，其他页面仍按任务状态机继续。
- 插件包加载或完整性失败时注册表标记为 `error` 并关闭运行时启用态；已冻结
  该版本的任务仍按其 failure policy 给出明确结果，不会静默跳过。

## 管理与发布

管理功能通过 `/api/v2/plugins` API 提供：列表、刷新校验、运行时开关、默认
开关、配置 CAS、导入、导出和删除。被任务或操作历史引用的版本不能删除，
后端返回 `423`，从而保证暂停、恢复和重试仍能加载完全相同的代码。

Plugin Agent 的规划会话是 30 分钟单活内存态；真正的编程执行会创建
`plugin_agent` 持久任务。Worker 只在
`temp/jobs/{job_id}/plugin-worktree/` 内使用六个文件工具，校验成功后才发布
新的不可变版本。浏览器断线不会终止执行，取消操作使用通用任务中心。

## 从旧版迁移

旧版插件需要人工或借助 Plugin Agent 改写：

1. 删除对 `src.plugins.base.PluginBase`、旧 Manager 和 HTTP 路由的依赖。
2. 把类元数据迁移到根目录 `plugin.json`。
3. Hook 改为始终返回字典，并改用 v3 `context`。
4. 删除 Base64 图片、绝对路径和浏览器 Pipeline 协调代码，改用资产 ID。
5. 打包 ZIP 后通过插件管理页面导入；不要把目录直接复制进受管理存储。
6. 用后端任务验证模式/步骤过滤、失败策略、配置冻结和重启恢复。
