# Saber Translator 插件开发文档

这份文档的目标很明确：

- 你**不需要熟悉整个项目**
- 你**不需要先看别人的插件**
- 你只要按本文档操作，就能写出一个**适配当前项目**、能被系统加载、能参与翻译流程的插件

本文档描述的是**当前项目正在使用的插件系统**，也就是围绕原子步骤翻译流程工作的插件系统。

---

## 1. 先理解插件能做什么

Saber Translator 的翻译流程已经拆成多个原子步骤。插件不是“自己发起一次翻译”，而是在系统执行这些步骤时插入逻辑。

当前插件可以接入的步骤有：

- `detect`
- `ocr`
- `color`
- `translate`
- `ai_translate`
- `inpaint`
- `render`
- `pipeline` ← **全局生命周期钩子**，每次完整翻译任务只触发一次

每个步骤都支持两类 hook：

- `before_*`
- `after_*`

你可以把插件理解成“步骤级中间件”：

- `before_*`：在某个步骤执行前，修改这个步骤的输入参数
- `after_*`：在某个步骤执行后，修改这个步骤的输出结果

例如：

- 在 `before_translate` 里给待翻译文本加前缀
- 在 `after_ocr` 里清洗 OCR 结果
- 在 `before_render` 里统一修改文字颜色
- 在 `after_detect` 里过滤掉你不想要的检测框

### `pipeline` 步骤说明

`pipeline` 是一个特殊步骤，区别于上面 7 个原子步骤：

- **触发时机**：用户点击「翻译当前图片 / 翻译全部 / 翻译范围 / 重试失败 / 校对 / 消除文字」等任何一次完整翻译任务时，`before_pipeline` 在所有图片所有原子步骤之前触发**一次**，`after_pipeline` 在所有原子步骤完成后触发**一次**
- **不会触发**：单气泡 OCR、单气泡修复、重渲染、应用样式到全部等「步骤级小操作」不算一次 pipeline，不会触发
- **payload 字段**：
  - `before_pipeline`：`pipeline_id` / `mode` / `scope` / `page_indexes` / `total_images`
  - `after_pipeline`：`pipeline_id` / `mode` / `scope` / `completed` / `failed` / `errors` / `warnings_count` / `duration_ms`
- **取消任务**：在 `before_pipeline` 里 `raise PluginException("...")`（且 `failure_policy="fail"`）会立即取消本次翻译，前端会收到 toast 错误并不会执行任何原子步骤
- **配对上下文**：每次任务有唯一的 `pipeline_id`，before 和 after 共享同一个 ID，便于做缓存、统计、合规审计

最小示例：

```python
from src.plugins.base import PluginBase


class MyLifecyclePlugin(PluginBase):
    plugin_id = "my_lifecycle_plugin"
    display_name = "我的生命周期插件"
    plugin_version = "1.0.0"
    plugin_author = "Your Name"
    plugin_description = "演示 pipeline 钩子"
    default_enabled = False
    supported_steps = ("pipeline",)
    supported_modes = ("standard", "hq", "proofread", "remove_text")
    priority = 10

    def before_pipeline(self, context, payload):
        self.logger.info("一次任务开始: pipeline_id=%s total_images=%s",
                         payload.get("pipeline_id"), payload.get("total_images"))
        return None

    def after_pipeline(self, context, result):
        self.logger.info("一次任务结束: pipeline_id=%s 成功=%s 失败=%s 耗时=%sms",
                         result.get("pipeline_id"),
                         result.get("completed"),
                         result.get("failed"),
                         result.get("duration_ms"))
        return None
```

更完整的示例可以参考 [plugins/pipeline_lifecycle_plugin/plugin.py](C:/Users/33252/Desktop/Saber-Translator/plugins/pipeline_lifecycle_plugin/plugin.py)。

---

## 2. 插件作用于哪些翻译模式

当前系统支持的翻译模式有：

- `standard`
- `hq`
- `proofread`
- `remove_text`

插件可以声明自己只在某些模式下生效。

例如：

- 只在普通翻译和高质量翻译时工作
- 不在消除文字模式下工作

你在插件里必须使用规范模式名：

```python
supported_modes = ("standard", "hq", "proofread", "remove_text")
```

---

## 3. 插件放在哪里

用户插件目录是项目根下的：

- [plugins](C:/Users/33252/Desktop/Saber-Translator/plugins)

每个插件都是一个子目录。

最小目录结构如下：

```text
plugins/
  my_plugin/
    __init__.py
    plugin.py
```

如果你要写一个名为 `my_first_plugin` 的插件，目录应该是：

```text
plugins/
  my_first_plugin/
    __init__.py
    plugin.py
```

如果你的插件逻辑比较多，也可以把辅助代码拆出去，例如：

```text
plugins/
  my_first_plugin/
    __init__.py
    plugin.py
    helpers.py
    prompts.py
```

在这种情况下，推荐使用**相对导入**：

```python
from .helpers import build_prompt
```

---

## 4. 从零创建一个插件

下面是最推荐的从零创建方法。

### 第一步：创建目录

创建：

```text
plugins/
  my_first_plugin/
```

### 第二步：创建 `__init__.py`

内容如下：

```python
from .plugin import MyFirstPlugin
```

这里的 `MyFirstPlugin` 是你的插件类名。

### 第三步：创建 `plugin.py`

最小可运行版本如下：

```python
from src.plugins.base import PluginBase


class MyFirstPlugin(PluginBase):
    plugin_id = "my_first_plugin"
    display_name = "我的第一个插件"
    plugin_version = "1.0.0"
    plugin_author = "Your Name"
    plugin_description = "一个最小可运行插件"
    default_enabled = False
    supported_steps = ("ocr",)
    supported_modes = ("standard", "hq", "proofread")
    priority = 100
    failure_policy = "continue"

    def after_ocr(self, context, result):
        self.logger.info(
            "after_ocr 被触发: step=%s mode=%s route=%s",
            context.step,
            context.mode,
            context.route,
        )
        return None
```

这个插件做的事情很简单：

- 它声明自己只接管 `ocr`
- 它只在 `standard / hq / proofread` 模式下运行
- 当 OCR 完成后，它只打日志，不修改结果

这是最适合第一次验证插件系统是否正常工作的起点。

---

## 5. 先记住一条最重要的实践规则

写插件时，尽量遵循下面这条原则：

- **优先读写当前插件系统定义的规范字段，不要自己发明别名，也不要依赖非规范写法**

例如：

- 优先使用 `original_texts`，不要优先假设一定会有 `original_text`
- 优先使用 `translated_texts`，不要自己创造别的字段名
- 优先使用 `remove_text`
- 优先使用 `ai_translate`

你的插件源码里应该始终按规范字段来写。

---

## 6. 插件类必须继承什么

所有插件都必须继承：

- [src/plugins/base.py](C:/Users/33252/Desktop/Saber-Translator/src/plugins/base.py) 里的 `PluginBase`

导入方式：

```python
from src.plugins.base import PluginBase
```

不要自己复制基类，不要写一个名字相似的类替代它。

---

## 7. 插件类上必须定义的字段

下面这些字段建议你都明确写出来。

### `plugin_id`

- 类型：`str`
- 必填
- 作用：插件的唯一标识符

这个字段非常重要，它会用于：

- 插件列表 API
- 插件启用/禁用
- 插件删除
- 插件配置文件命名
- 默认启用状态持久化

要求：

- 不能为空
- 必须唯一
- 推荐使用英文、数字、下划线

推荐写法：

```python
plugin_id = "my_first_plugin"
```

### `display_name`

- 类型：`str`
- 推荐必填
- 作用：显示在前端插件管理界面中的名字

例如：

```python
display_name = "我的第一个插件"
```

### `plugin_version`

- 类型：`str`
- 推荐必填

例如：

```python
plugin_version = "1.0.0"
```

### `plugin_author`

- 类型：`str`
- 推荐必填

### `plugin_description`

- 类型：`str`
- 推荐必填

这个字段会显示在插件管理界面中，建议写清楚插件用途。

### `default_enabled`

- 类型：`bool`
- 作用：插件第一次被系统发现时，默认是否启用

推荐：

```python
default_enabled = False
```

也就是说，开发中的插件默认先不要自动启用。

### `supported_steps`

- 类型：`tuple[str, ...]`
- 必填
- 作用：声明插件接管哪些步骤

可选值只有：

- `detect`
- `ocr`
- `color`
- `translate`
- `ai_translate`
- `inpaint`
- `render`
- `pipeline` ← 全局生命周期钩子

例如：

```python
supported_steps = ("ocr", "translate")
```

如果你写的是只关心一次任务起止的「生命周期插件」（统计、配额、缓存、审计等），就只需要：

```python
supported_steps = ("pipeline",)
```

### `supported_modes`

- 类型：`tuple[str, ...]`
- 必填
- 作用：声明插件在哪些翻译模式下生效

可选值只有：

- `standard`
- `hq`
- `proofread`
- `remove_text`

例如：

```python
supported_modes = ("standard", "hq")
```

### `priority`

- 类型：`int`
- 默认值：`100`
- 作用：多个插件命中同一步骤时的执行顺序

规则：

- 数字越小，执行越早
- 如果两个插件优先级一样，系统再按插件 ID 排序

例如：

```python
priority = 20
```

### `failure_policy`

- 类型：`str`
- 可选值：
  - `continue`
  - `fail`

含义：

- `continue`：插件报错时只记日志，不中断主流程
- `fail`：插件报错时直接让当前步骤失败

推荐写法：

```python
failure_policy = "continue"
```

大多数插件都应该用 `continue`。

---

## 8. 生命周期方法

除了步骤 hook 之外，插件还可以定义几个生命周期方法。

### `setup(self) -> bool`

插件被加载时调用。

适合在这里做：

- 初始化只读资源
- 做依赖检查
- 检查某些外部文件是否存在

重要说明：

- `setup()` 执行时，插件配置**还没有**加载进 `self.config`
- 如果你的逻辑依赖配置值，不要在 `setup()` 里读取它
- 依赖配置的逻辑应该放到：
  - `on_enable()`
  - 某个 hook
  - 你自己的懒加载逻辑里

返回值：

- `True`：插件加载成功
- `False`：插件加载失败

示例：

```python
def setup(self) -> bool:
    self.logger.info("插件初始化完成")
    return True
```

### `on_enable(self) -> None`

插件启用时调用。

适合在这里做：

- 清空调试日志
- 初始化运行时缓存
- 重置状态文件

### `on_disable(self) -> None`

插件禁用时调用。

适合在这里做：

- 清理临时状态
- 关闭外部资源句柄

---

## 9. 步骤 / 路由 / hook 对照表

下面这张表是开发插件时最值得反复查的一张表。

它回答了 3 个最关键的问题：

1. 你的功能最终会打到哪个后端路由
2. 这个路由属于哪个插件步骤
3. 这个路由前后会触发哪些 hook

| 场景 | 路由 | 步骤名 | 触发的 hook |
|---|---|---|---|
| 并行检测 | `/api/parallel/detect` | `detect` | `before_detect` → `after_detect` |
| 并行 OCR | `/api/parallel/ocr` | `ocr` | `before_ocr` → `after_ocr` |
| 并行颜色提取 | `/api/parallel/color` | `color` | `before_color` → `after_color` |
| 并行普通翻译 | `/api/parallel/translate` | `translate` | `before_translate` → `after_translate` |
| 并行修复 | `/api/parallel/inpaint` | `inpaint` | `before_inpaint` → `after_inpaint` |
| 并行渲染 | `/api/parallel/render` | `render` | `before_render` → `after_render` |
| 单条文本翻译 | `/api/translate_single_text` | `translate` | `before_translate` → `after_translate` |
| 高质量翻译批处理 | `/api/hq_translate_batch` | `ai_translate` | `before_ai_translate` → `after_ai_translate` |
| 单气泡 OCR | `/api/ocr_single_bubble` | `ocr` | `before_ocr` → `after_ocr` |
| 单气泡修复 | `/api/inpaint_single_bubble` | `inpaint` | `before_inpaint` → `after_inpaint` |
| 重新渲染整张图片 | `/api/re_render_image` | `render` | `before_render` → `after_render` |
| 重新渲染单个气泡 | `/api/re_render_single_bubble` | `render` | `before_render` → `after_render` |
| 将样式应用到所有图片 | `/api/apply_settings_to_all_images` | `render` | `before_render` → `after_render` |

如果你想开发插件，应该先确定：

- 你要改的是哪个场景
- 那个场景最终走哪个路由
- 那个路由对应哪个步骤
- 然后只实现那一步骤的 hook

例如：

- 你要改 OCR 文本，就优先写 `after_ocr`
- 你要改普通翻译文本，就优先写 `after_translate`
- 你要改高质量翻译输出，就优先写 `after_ai_translate`
- 你要改最终样式，就优先写 `before_render`

---

## 10. 不同翻译模式会触发哪些 hook

这是文档里最容易被忽略、但开发时最有用的一部分。

如果你不知道某个 hook 为什么没有触发，通常不是插件没加载，而是你选的翻译模式压根不会经过那个步骤。

### `standard`

执行顺序是：

1. `detect`
2. `ocr`
3. `color`
4. `translate`
5. `inpaint`
6. `render`

如果你在这些步骤上都实现了 `before_*` 和 `after_*`，它们会按这个顺序触发。

### `hq`

执行顺序是：

1. `detect`
2. `ocr`
3. `color`
4. `ai_translate`
5. `inpaint`
6. `render`

注意：

- `hq` 模式**不会走普通 `translate`**
- 它走的是 `ai_translate`

### `proofread`

执行顺序是：

1. `ai_translate`
2. `render`

注意：

- `proofread` 不会重新做检测、OCR、颜色提取、修复
- 所以这些步骤上的 hook 在 `proofread` 下不会触发

### `remove_text`

默认执行顺序是：

1. `detect`
2. `inpaint`
3. `render`

如果用户打开了“消除文字时也做 OCR”，则顺序会变成：

1. `detect`
2. `ocr`
3. `inpaint`
4. `render`

注意：

- `remove_text` 模式默认不会走 `translate`
- 也不会走 `ai_translate`

如果你的插件声明支持了 `translate`，但你在 `remove_text` 模式里测试它，它不会被触发，这是正常现象。

---

## 11. 所有可用 hook 名称

你只需要按名字覆盖对应方法即可。

### 检测步骤

- `before_detect`
- `after_detect`

### OCR 步骤

- `before_ocr`
- `after_ocr`

### 颜色提取步骤

- `before_color`
- `after_color`

### 普通翻译步骤

- `before_translate`
- `after_translate`

### AI 翻译 / AI 校对步骤

- `before_ai_translate`
- `after_ai_translate`

### 修复步骤

- `before_inpaint`
- `after_inpaint`

### 渲染步骤

- `before_render`
- `after_render`

方法签名统一如下：

```python
def before_xxx(self, context, payload):
    ...

def after_xxx(self, context, result):
    ...
```

---

## 12. hook 应该返回什么

这是开发插件时最重要的规则之一。

### 对于所有 `before_*`

你可以：

- 返回 `None`
- 返回一个 `dict`

含义：

- `None`：不修改输入
- `dict`：用你返回的字典替换当前步骤输入

### 对于所有 `after_*`

你也只能：

- 返回 `None`
- 返回一个 `dict`

含义：

- `None`：不修改输出
- `dict`：用你返回的字典替换当前步骤输出

### 不允许返回的类型

不要返回：

- `list`
- `str`
- `tuple`
- 自定义对象实例

如果 hook 返回错误类型，当前请求会报错。

### 推荐写法

如果你要修改内容，推荐先复制一份结果再改：

```python
import copy

def after_translate(self, context, result):
    updated = copy.deepcopy(result)
    updated["translated_texts"] = [
        f"{text}【插件】"
        for text in updated.get("translated_texts", [])
    ]
    return updated
```

---

## 13. `context` 参数里有什么

每个 hook 都会收到一个 `context` 参数。它的定义在：

- [src/plugins/hooks.py](C:/Users/33252/Desktop/Saber-Translator/src/plugins/hooks.py)

它至少包含这几个字段：

### `context.step`

当前步骤名，例如：

- `detect`
- `ocr`
- `translate`
- `render`

### `context.mode`

当前翻译模式，例如：

- `standard`
- `hq`
- `proofread`
- `remove_text`

### `context.route`

当前命中的后端路由，例如：

- `/api/parallel/ocr`
- `/api/parallel/translate`
- `/api/hq_translate_batch`
- `/api/re_render_single_bubble`

### `context.scope`

当前请求范围，例如：

- `bubble`
- `image`
- `batch`

### `context.metadata`

一个额外信息字典，不同步骤里内容可能不同。

常见键可能包括：

- `bubble_count`
- `text_count`
- `target_language`
- `source_language`
- `provider`
- `image_count`
- `is_proofreading`

使用建议：

```python
text_count = context.metadata.get("text_count", 0)
```

不要假设某个键一定存在。

---

## 14. 每个 hook 里通常会收到什么

这里不是项目所有路由字段的完全列表，而是为了让你开发插件时知道“最常见能改的东西是什么”。

### `before_detect`

常见输入字段：

- `image`
- `detector_type`
- `box_expand_ratio`
- `box_expand_top`
- `box_expand_bottom`
- `box_expand_left`
- `box_expand_right`
- `translation_mode`
- `translation_scope`

最常见用途：

- 强制改用某个检测器
- 修改检测参数

示例：

```python
def before_detect(self, context, payload):
    updated = dict(payload)
    updated["detector_type"] = "default"
    return updated
```

### `after_detect`

常见输出字段：

- `success`
- `bubble_coords`
- `bubble_angles`
- `bubble_polygons`
- `auto_directions`
- `raw_mask`
- `textlines_per_bubble`

最常见用途：

- 删除异常框
- 插入测试框
- 修正检测方向

### `before_ocr`

常见输入字段：

- `image`
- `bubble_coords`
- `ocr_engine`
- `source_language`
- `textlines_per_bubble`

最常见用途：

- 强制 OCR 引擎
- 修改 OCR 参数

### `after_ocr`

常见输出字段：

- `success`
- `original_texts`
- `ocr_results`
- `textlines_per_bubble`

最常见用途：

- 清洗 OCR 文本
- 替换特定词
- 给识别结果打标

如果你修改了 `original_texts`，**最好同步修改** `ocr_results[i]["text"]`。

示例：

```python
import copy

def after_ocr(self, context, result):
    updated = copy.deepcopy(result)
    updated["original_texts"] = [
        text.strip()
        for text in updated.get("original_texts", [])
    ]
    for index, item in enumerate(updated.get("ocr_results", [])):
        if isinstance(item, dict) and index < len(updated["original_texts"]):
            item["text"] = updated["original_texts"][index]
    return updated
```

### `before_color`

常见输入字段：

- `image`
- `bubble_coords`
- `textlines_per_bubble`

最常见用途：

- 修改颜色提取输入

### `after_color`

常见输出字段：

- `success`
- `colors`

`colors` 通常是数组，每项可能包含：

- `textColor`
- `bgColor`
- `autoFgColor`
- `autoBgColor`

最常见用途：

- 强制覆盖文字色和背景色

### `before_translate`

常见输入字段：

- `original_texts`
- `target_language`
- `source_language`
- `model_provider`
- `model_name`
- `prompt_content`
- `textbox_prompt_content`
- `glossary_settings`
- `non_translate_settings`

最常见用途：

- 修改待翻译文本
- 动态追加 prompt
- 切换模型参数

最常改的是：

- `original_texts`
- `prompt_content`

### `after_translate`

常见输出字段：

- `success`
- `translated_texts`
- `textbox_texts`
- `warnings`

最常见用途：

- 后处理普通翻译结果
- 给译文加标记
- 替换敏感词

### `before_ai_translate`

对应高质量翻译 / AI 校对步骤。

常见输入字段：

- `provider`
- `api_key`
- `model_name`
- `jsonData`
- `imageBase64Array`
- `target_language`
- `prompt`
- `systemPrompt`
- `isProofreading`

最常见用途：

- 修改发送给 AI 的 prompt
- 改 provider / model
- 改结构化输入数据

### `after_ai_translate`

常见输出字段：

- `success`
- `results`
- `warnings`

`results` 通常是数组，每项类似：

```json
{
  "imageIndex": 0,
  "bubbles": [
    {
      "bubbleIndex": 0,
      "translated": "..."
    }
  ]
}
```

最常见用途：

- 后处理高质量翻译结果
- 后处理 AI 校对结果

### `before_inpaint`

常见输入字段：

- `image`
- `bubble_coords`
- `bubble_polygons`
- `raw_mask`
- `user_mask`
- `method`
- `lama_model`
- `fill_color`

最常见用途：

- 改修复方法
- 改填充色
- 改 mask 参数

### `after_inpaint`

常见输出字段：

- `success`
- `clean_image`

最常见用途：

- 记录修复结果
- 替换修复后的背景图

### `before_render`

常见输入字段：

- `clean_image`
- `bubble_states`
- `fontSize`
- `fontFamily`
- `textDirection`
- `textColor`
- `strokeEnabled`
- `strokeColor`
- `strokeWidth`
- `lineSpacing`
- `textAlign`
- `autoFontSize`

最常见用途：

- 统一覆盖文字颜色
- 统一开启描边
- 批量改字号或布局
- 修改 `bubble_states` 中单个气泡的局部样式

### `after_render`

常见输出字段：

- `success`
- `final_image`
- `bubble_states`

在一些编辑相关路由里也可能出现：

- `rendered_image`
- `rendered_images`

最常见用途：

- 后处理渲染结果
- 调试输出

---

## 15. 插件里应该优先使用哪些字段

为了让你的插件尽可能稳定，建议优先依赖下面这些字段。

### 文本相关

优先使用：

- `original_texts`
- `translated_texts`
- `textbox_texts`
- `ocr_results`
- `warnings`

### 检测相关

优先使用：

- `bubble_coords`
- `bubble_angles`
- `bubble_polygons`
- `auto_directions`
- `textlines_per_bubble`

### 渲染相关

优先使用：

- `bubble_states`
- `textColor`
- `strokeColor`
- `strokeEnabled`
- `strokeWidth`
- `fontSize`
- `fontFamily`
- `textDirection`

### AI 翻译相关

优先使用：

- `jsonData`
- `imageBase64Array`
- `prompt`
- `systemPrompt`
- `isProofreading`
- `results`

### 一个重要建议

如果同一个概念同时出现在“单值字段”和“数组字段”里，插件开发时通常优先依赖数组/结构化字段。

例如：

- 普通翻译场景优先看 `original_texts`
- AI 翻译场景优先看 `results`
- 渲染场景优先看 `bubble_states`

---

## 16. 如何让插件带配置

如果你的插件需要让用户在前端配置参数，就实现：

```python
def get_config_schema(self) -> Dict[str, Dict[str, Any]]:
    ...
```

返回值必须是一个**字典对象**，不是数组。

示例：

```python
def get_config_schema(self):
    return {
        "suffix": {
            "type": "text",
            "label": "文本后缀",
            "default": "【插件】",
            "description": "追加到译文尾部"
        },
        "debug_enabled": {
            "type": "boolean",
            "label": "启用调试",
            "default": False,
            "description": "是否记录调试信息"
        },
        "max_items": {
            "type": "number",
            "label": "最大数量",
            "default": 10,
            "min": 1,
            "max": 100,
            "description": "示例数值配置"
        },
        "strategy": {
            "type": "select",
            "label": "策略",
            "default": "soft",
            "options": [
                {"value": "soft", "label": "Soft"},
                {"value": "hard", "label": "Hard"}
            ],
            "description": "示例下拉框"
        }
    }
```

支持的 `type` 有：

- `text`
- `number`
- `boolean`
- `select`

前端保存配置后，基类会自动把配置加载到：

- `self.config`

所以你在 hook 中直接读取即可：

```python
suffix = self.config.get("suffix", "")
debug_enabled = self.config.get("debug_enabled", False)
```

配置文件会自动保存到：

- `config/plugin_configs/<plugin_id>.json`

---

## 17. 插件能导入什么

在 `plugin.py` 里，你可以正常导入：

- Python 标准库
- 项目源码里的模块
- 你自己插件目录下的辅助模块
- 已经安装在项目虚拟环境里的第三方库

例如：

```python
import copy
import json
from pathlib import Path

from src.plugins.base import PluginBase
from .helpers import build_prompt
```

注意：

- 如果你依赖第三方库，它必须已经安装在项目运行时使用的虚拟环境里
- 不要假设用户机器全局 Python 装了你的依赖

---

## 18. 插件是怎么被发现和加载的

启动后端时，插件管理器会扫描两个目录：

- [src/plugins](C:/Users/33252/Desktop/Saber-Translator/src/plugins)（内置插件）
- [plugins](C:/Users/33252/Desktop/Saber-Translator/plugins)（用户插件）

扫描规则：

- 目录必须存在
- 插件目录里必须有 `__init__.py`
- 插件目录里必须有 `plugin.py`

一个插件被加载的大致顺序是：

1. 导入模块
2. 找到继承 `PluginBase` 的类
3. 校验元数据
4. 调用 `setup()`
5. 读取配置文件
6. 注册 hook
7. 根据默认启用状态决定是否启用

重要说明：

- **新建插件目录后，通常需要重启后端，插件才会被重新扫描到**
- **修改插件源码后，通常也需要重启后端，新的 Python 代码才会生效**
- 前端里的“启用/禁用插件”不需要重启，它只是在**已经加载成功的插件**之间切换运行状态

简单记忆：

- 新增插件：重启后端
- 改代码：重启后端
- 改配置：不用重启
- 启用/禁用：不用重启

---

## 19. 插件配置和启用状态保存在哪里

### 插件配置文件

保存在：

- `config/plugin_configs/<plugin_id>.json`

### 插件默认启用状态

保存在：

- [config/plugin_default_states.json](C:/Users/33252/Desktop/Saber-Translator/config/plugin_default_states.json)

例如：

```json
{
  "my_first_plugin": false
}
```

---

## 20. 如何在前端启用插件

启动项目后，打开：

- 设置
- 插件管理

你会看到插件列表、当前启用状态、默认启用状态、配置按钮。

启用插件后：

- 插件会立即参与后续请求

禁用插件后：

- 插件不会再执行 hook

---

## 21. 你真正应该按什么顺序开发插件

这是最推荐的顺序，尤其适合第一次写插件的人。

### 第一步：先写一个只打日志的插件

只实现一个 `after_ocr` 或 `after_translate`，先确认插件能被加载。

### 第二步：确认插件能出现在前端插件管理里

如果前端看不到插件，先不要继续写业务逻辑。

### 第三步：启用插件，跑一遍翻译流程

确认你的日志真的出现了。

### 第四步：让插件返回一个明显变化

例如给文本加 `【测试】` 后缀。

### 第五步：确认界面或接口结果真的变化了

如果结果没变化，说明 hook 没生效，或者你改错字段了。

### 第六步：最后再加配置项

等最小逻辑跑通后，再做前端可配置化。

---

## 22. 开发时最常见的错误

### 错误 1：hook 返回了错误类型

错误示例：

```python
def after_translate(self, context, result):
    return ["bad", "result"]
```

正确写法：

```python
def after_translate(self, context, result):
    updated = dict(result)
    return updated
```

### 错误 2：`supported_steps` 写了不存在的值

错误示例：

```python
supported_steps = ("translation",)
```

正确写法：

```python
supported_steps = ("translate",)
```

### 错误 3：`supported_modes` 写了不规范的值

错误示例：

```python
supported_modes = ("remove_text_mode",)
```

正确写法：

```python
supported_modes = ("remove_text",)
```

### 错误 4：忘了写 `__init__.py`

没有 `__init__.py` 的目录不会被当成插件包扫描。

### 错误 5：`plugin_id` 和别的插件重复

重复的 `plugin_id` 会导致加载失败。

### 错误 6：你改了 `original_texts`，但忘了同步 `ocr_results`

这样前端看到的原文和 OCR 结构化结果可能不一致。

---

## 23. 一个完整的、可以直接改出来用的示例插件

下面这个示例插件足够完整，开发者只靠这个示例和前面的字段说明，就能自己写出插件。

```python
import copy

from src.plugins.base import PluginBase


class DemoPlugin(PluginBase):
    plugin_id = "demo_plugin"
    display_name = "Demo Plugin"
    plugin_version = "1.0.0"
    plugin_author = "Demo Author"
    plugin_description = "演示 OCR 和翻译 hook"
    default_enabled = False
    supported_steps = ("ocr", "translate")
    supported_modes = ("standard", "hq", "proofread")
    priority = 50
    failure_policy = "continue"

    def get_config_schema(self):
        return {
            "ocr_suffix": {
                "type": "text",
                "label": "OCR 后缀",
                "default": "【OCR】",
                "description": "追加到 OCR 结果尾部"
            },
            "translate_suffix": {
                "type": "text",
                "label": "翻译后缀",
                "default": "【TR】",
                "description": "追加到翻译结果尾部"
            }
        }

    def after_ocr(self, context, result):
        updated = copy.deepcopy(result)
        suffix = self.config.get("ocr_suffix", "【OCR】")
        updated["original_texts"] = [
            f"{text}{suffix}"
            for text in updated.get("original_texts", [])
        ]
        for index, item in enumerate(updated.get("ocr_results", [])):
            if isinstance(item, dict) and index < len(updated["original_texts"]):
                item["text"] = updated["original_texts"][index]
        return updated

    def after_translate(self, context, result):
        updated = copy.deepcopy(result)
        suffix = self.config.get("translate_suffix", "【TR】")
        updated["translated_texts"] = [
            f"{text}{suffix}"
            for text in updated.get("translated_texts", [])
        ]
        return updated
```

---

## 24. 按需求选 hook 的开发配方

如果你已经知道自己“想做什么”，但不想先完整理解整个流程，可以直接看这一节。

下面这些配方的目标是：

- 让你先按需求选对步骤和 hook
- 再回头结合前面的字段说明写代码

### 场景 1：我只想改 OCR 文本

适合的 hook：

- `after_ocr`

原因：

- OCR 识别结果已经出来了
- 你能同时拿到：
  - `original_texts`
  - `ocr_results`

最适合做的事：

- 清洗 OCR 结果
- 去掉多余空格
- 替换固定词
- 给识别结果打标记

最小示例：

```python
import copy

def after_ocr(self, context, result):
    updated = copy.deepcopy(result)
    updated["original_texts"] = [
        text.replace("…", "...").strip()
        for text in updated.get("original_texts", [])
    ]
    for index, item in enumerate(updated.get("ocr_results", [])):
        if isinstance(item, dict) and index < len(updated["original_texts"]):
            item["text"] = updated["original_texts"][index]
    return updated
```

推荐模式：

- `standard`
- `hq`
- `proofread` 以外的模式都要注意是否真的经过 OCR

注意：

- `proofread` 默认不会重新做 OCR，所以 `after_ocr` 在 `proofread` 下不会触发

---

### 场景 2：我只想改普通翻译文本

适合的 hook：

- `before_translate`
- `after_translate`

什么时候用 `before_translate`：

- 你想改待翻译的源文
- 你想动态改 prompt
- 你想按文本内容决定某些翻译参数

什么时候用 `after_translate`：

- 你想后处理译文
- 你想统一补后缀
- 你想替换敏感词

最小示例：

```python
import copy

def before_translate(self, context, payload):
    updated = copy.deepcopy(payload)
    updated["original_texts"] = [
        f"[PRE]{text}"
        for text in updated.get("original_texts", [])
    ]
    return updated

def after_translate(self, context, result):
    updated = copy.deepcopy(result)
    updated["translated_texts"] = [
        f"{text}【后处理】"
        for text in updated.get("translated_texts", [])
    ]
    return updated
```

推荐模式：

- `standard`

注意：

- `hq` 模式不会走普通 `translate`
- `hq` 用的是 `ai_translate`

---

### 场景 3：我只想改 HQ 翻译或 AI 校对的 prompt / 输出

适合的 hook：

- `before_ai_translate`
- `after_ai_translate`

什么时候用 `before_ai_translate`：

- 你想改高质量翻译 prompt
- 你想改 AI 校对 prompt
- 你想改传给模型的结构化数据

什么时候用 `after_ai_translate`：

- 你想后处理高质量翻译结果
- 你想后处理 AI 校对结果

最小示例：

```python
import copy

def before_ai_translate(self, context, payload):
    updated = copy.deepcopy(payload)
    prompt = str(updated.get("prompt", "") or "")
    updated["prompt"] = prompt + "\n\n请统一使用更自然的中文口语。"
    return updated

def after_ai_translate(self, context, result):
    updated = copy.deepcopy(result)
    for image_item in updated.get("results", []):
        if not isinstance(image_item, dict):
            continue
        for bubble in image_item.get("bubbles", []):
            if isinstance(bubble, dict) and bubble.get("translated"):
                bubble["translated"] = bubble["translated"].strip()
    return updated
```

推荐模式：

- `hq`
- `proofread`

注意：

- 这一步用的是 `results`，不是 `translated_texts`

---

### 场景 4：我只想改渲染样式

适合的 hook：

- `before_render`

原因：

- 渲染前你能同时拿到全局样式参数和 `bubble_states`
- 这是改文字颜色、描边、字号、方向最直接的位置

最小示例：

```python
import copy

def before_render(self, context, payload):
    updated = copy.deepcopy(payload)
    updated["textColor"] = "#ff0000"
    updated["strokeEnabled"] = True
    updated["strokeColor"] = "#0000ff"

    for bubble_state in updated.get("bubble_states", []):
        if not isinstance(bubble_state, dict):
            continue
        bubble_state["textColor"] = "#ff0000"
        bubble_state["strokeEnabled"] = True
        bubble_state["strokeColor"] = "#0000ff"
        bubble_state["strokeWidth"] = 2

    return updated
```

推荐模式：

- `standard`
- `hq`
- `proofread`
- `remove_text`

注意：

- 如果你只改全局字段，不改 `bubble_states`，有些局部样式可能不会被覆盖
- 所以最稳妥的方式是全局字段和 `bubble_states` 一起改

---

### 场景 5：我只想改检测结果

适合的 hook：

- `before_detect`
- `after_detect`

什么时候用 `before_detect`：

- 你想强制检测器
- 你想改检测参数

什么时候用 `after_detect`：

- 你想过滤检测框
- 你想插入测试框
- 你想修正检测结果

最小示例：

```python
import copy

def after_detect(self, context, result):
    updated = copy.deepcopy(result)
    coords = updated.get("bubble_coords", [])
    updated["bubble_coords"] = [coord for coord in coords if len(coord) >= 4]
    return updated
```

推荐模式：

- `standard`
- `hq`
- `remove_text`

注意：

- `proofread` 默认不会做检测，所以 `after_detect` 在 `proofread` 下不会触发

---

### 场景 6：我只想改颜色提取结果

适合的 hook：

- `after_color`

最适合做的事：

- 强制文字颜色
- 强制背景颜色
- 覆盖自动提取出的颜色

最小示例：

```python
import copy

def after_color(self, context, result):
    updated = copy.deepcopy(result)
    for color_info in updated.get("colors", []):
        if not isinstance(color_info, dict):
            continue
        color_info["textColor"] = "#111111"
        color_info["bgColor"] = "#fff4b8"
    return updated
```

推荐模式：

- `standard`
- `hq`

注意：

- `proofread` 默认不会做颜色提取

---

### 场景 7：我只想改修复结果

适合的 hook：

- `before_inpaint`
- `after_inpaint`

什么时候用 `before_inpaint`：

- 你想改修复方法
- 你想改 `fill_color`
- 你想改 mask 相关参数

什么时候用 `after_inpaint`：

- 你想替换修复后的背景图
- 你想记录修复结果

最小示例：

```python
import copy

def before_inpaint(self, context, payload):
    updated = copy.deepcopy(payload)
    updated["fill_color"] = "#ccffcc"
    return updated
```

推荐模式：

- `standard`
- `hq`
- `remove_text`

注意：

- `proofread` 默认不会走修复步骤

---

### 场景 8：我只想做日志追踪 / 观测，不改任何结果

适合的 hook：

- 任意你关心的 hook

推荐做法：

- 先只实现一个 hook
- 先只打日志
- 先确认触发，再逐步增加改写逻辑

最小示例：

```python
def after_translate(self, context, result):
    self.logger.info(
        "after_translate: mode=%s route=%s text_count=%s",
        context.mode,
        context.route,
        context.metadata.get("text_count"),
    )
    return None
```

什么时候最适合这样做：

- 你第一次写插件
- 你不确定流程到底走了哪个步骤
- 你想先定位问题，再决定改哪一层

---

### 场景 9：我不确定到底该写哪个 hook

按下面顺序判断：

1. 我是想改“检测框”吗？
   - 是：看 `detect`
2. 我是想改“OCR 原文”吗？
   - 是：看 `ocr`
3. 我是想改“普通翻译结果”吗？
   - 是：看 `translate`
4. 我是想改“高质量翻译 / 校对结果”吗？
   - 是：看 `ai_translate`
5. 我是想改“修复背景图”吗？
   - 是：看 `inpaint`
6. 我是想改“最终显示效果”吗？
   - 是：看 `render`

如果你还是不确定，最稳妥的办法是：

- 先写一个只打日志的插件
- 在你怀疑的几个步骤上分别打日志
- 看哪个 hook 真的被触发

---

## 25. 如何验证你写的插件真的适配当前项目

按下面顺序测试最稳妥。

### 第一步：启动后端

使用虚拟环境：

```powershell
.\venv\Scripts\python.exe app.py
```

### 第二步：打开前端

进入：

- 设置
- 插件管理

### 第三步：确认插件出现在列表中

正常现象：

- 能看到你的 `display_name`
- 能看到支持的步骤和模式
- 如果你实现了 `get_config_schema()`，会有配置按钮

### 第四步：启用插件

### 第五步：跑一遍你声明支持的模式

例如：

- 如果你支持 `standard`，就跑一次普通翻译
- 如果你支持 `hq`，就跑一次高质量翻译
- 如果你支持 `proofread`，就跑一次 AI 校对
- 如果你支持 `remove_text`，就跑一次消除文字

### 第六步：看你设计的可见结果

推荐至少做一个“肉眼可见”的变化，例如：

- OCR 结果后面追加 `【测试】`
- 译文后面追加 `【插件】`
- 文字颜色改成红色
- 描边颜色改成蓝色

不要只依赖日志。

---

## 26. 如何最快定位“为什么我的 hook 没触发”

按下面顺序排查最有效：

1. 插件是否出现在前端“插件管理”里
2. 插件是否已经启用
3. `plugin_id` 是否唯一
4. `supported_steps` 是否写对
5. `supported_modes` 是否写对
6. 你当前测试的翻译模式是否真的会经过这个步骤
7. hook 是否返回了非法类型
8. 插件是否在 `setup()` 阶段就返回了 `False`

最常见的真实原因通常是：

- 你在 `proofread` 模式下测试 `after_ocr`
- 你在 `remove_text` 模式下测试 `after_translate`
- 你把 `supported_steps` 写成了 `translation`
- 你返回了 `list` 而不是 `dict`

---

## 27. 推荐的调试手法

最简单有效的调试方式有三种。

### 方式 1：写日志

```python
self.logger.info("当前 step=%s mode=%s", context.step, context.mode)
```

### 方式 2：做一个可见的字符串改写

```python
def after_translate(self, context, result):
    updated = copy.deepcopy(result)
    updated["translated_texts"] = [
        f"{text}【测试】"
        for text in updated.get("translated_texts", [])
    ]
    return updated
```

### 方式 3：做一个可见的样式改写

```python
def before_render(self, context, payload):
    updated = copy.deepcopy(payload)
    updated["textColor"] = "#ff0000"
    updated["strokeEnabled"] = True
    updated["strokeColor"] = "#0000ff"
    return updated
```

---

## 28. 什么时候不适合用插件

插件适合做：

- 文本预处理
- 文本后处理
- 动态改 prompt
- 样式修正
- 调试观测
- 步骤输入输出微调

插件不适合做：

- 重写整个前端页面结构
- 增加一个完全新的翻译调度器
- 替换整个会话系统
- 重构书架、阅读器、分析系统的数据模型

如果你的需求是“在现有步骤前后改一点东西”，插件就很合适。

---

## 29. 开发完成前的自查清单

在你认为插件写完之前，逐条检查：

- 插件目录在 `plugins/` 下
- 目录里有 `__init__.py`
- 目录里有 `plugin.py`
- 插件类继承 `PluginBase`
- `plugin_id` 唯一
- `supported_steps` 只用了合法值
- `supported_modes` 只用了合法值
- hook 返回的是 `dict` 或 `None`
- 配置 schema 返回的是字典对象，不是数组
- 插件能在前端插件管理里出现
- 启用后至少有一个可见效果验证 hook 生效

如果这些都满足，通常这个插件就已经适配当前项目了。
