# 存储格式开发手册

> 最后更新：2026-01-19  
> 版本：适用于当前双格式兼容架构

本文档详细说明 Saber-Translator 的会话存储系统，包括新旧两种格式的结构、相关模块职责、以及如何进行修改。

---

## 目录

1. [架构概述](#架构概述)
2. [两种存储格式](#两种存储格式)
3. [相关模块清单](#相关模块清单)
4. [格式判断逻辑](#格式判断逻辑)
5. [数据流向](#数据流向)
6. [修改指南](#修改指南)
7. [常见问题](#常见问题)

---

## 架构概述

系统当前支持两种存储格式，用于兼容历史存档：

| 格式 | 状态 | 使用场景 |
|------|------|----------|
| **新格式** | ✅ 当前使用 | 所有新保存的书架会话 |
| **旧格式** | ⚠️ 仅读取 | 兼容历史存档，不再创建新的 |

### 存储位置

```
project_root/
└── data/
    └── sessions/
        ├── bookshelf/           # 书架模式会话
        │   └── {book_id}/
        │       └── {chapter_id}/
        │           └── ... (会话文件)
        └── {session_name}/      # 非书架模式会话（普通会话）
            └── ... (会话文件)
```

---

## 两种存储格式

### 新格式（推荐）

**特征**：嵌套目录结构，每页图片独立存储

```
data/sessions/bookshelf/{book_id}/{chapter_id}/
├── session_meta.json           # 会话元数据（含 total_pages 字段）
└── images/                     # 图片目录
    ├── 0/                      # 第1页
    │   ├── original.png        # 原图
    │   ├── translated.png      # 翻译后的图
    │   ├── clean.png           # 干净背景
    │   └── meta.json           # 页面元数据（气泡坐标等）
    ├── 1/                      # 第2页
    │   └── ...
    └── N/                      # 第N页
        └── ...
```

#### session_meta.json 结构（新格式）

```json
{
  "metadata": {
    "name": "bookshelf/book_id/chapter_id",
    "saved_at": "2026-01-19 15:00:00",
    "version": "3.0.0"
  },
  "total_pages": 10,                    // ⭐ 新格式特有字段
  "currentImageIndex": 0,
  "ui_settings": {
    "fontSize": 16,
    "fontFamily": "Source Han Sans",
    "textColor": "#000000",
    // ... 其他 UI 设置
  }
}
```

#### 页面 meta.json 结构

```json
{
  "hasOriginalData": true,
  "hasTranslatedData": true,
  "hasCleanData": false,
  "translationStatus": "translated",
  "bubbleStates": [
    {
      "originalText": "こんにちは",
      "translatedText": "你好",
      "rect": [100, 200, 150, 80],
      "polygon": [[100, 200], [250, 200], [250, 280], [100, 280]],
      // ... 其他气泡数据
    }
  ],
  "imageLevelFontSize": 16,
  "imageLevelFontFamily": "Source Han Sans"
}
```

### 旧格式（兼容）

**特征**：平铺文件结构，所有图片在同一目录

```
data/sessions/bookshelf/{book_id}/{chapter_id}/
├── session_meta.json           # 会话元数据（含 images_meta 数组）
├── image_0_original.png        # 第1页原图
├── image_0_translated.png      # 第1页翻译图
├── image_0_clean.png           # 第1页干净背景
├── image_1_original.png        # 第2页原图
└── ...
```

#### session_meta.json 结构（旧格式）

```json
{
  "metadata": {
    "name": "会话名称",
    "saved_at": "2024-01-01T00:00:00.000Z",
    "translator_version": "0.9.0"
  },
  "images_meta": [                      // ⭐ 旧格式特有字段
    {
      "hasOriginalData": true,
      "hasTranslatedData": true,
      "translationStatus": "translated",
      "bubbleStates": [...],
      // ... 每页的元数据嵌入在这里
    },
    // ... 更多页面
  ],
  "currentImageIndex": 0,
  "ui_settings": { ... }
}
```

### 格式对比

| 特性 | 新格式 | 旧格式 |
|------|--------|--------|
| 页数标识字段 | `total_pages` | `images_meta.length` |
| 图片目录 | `images/{idx}/` 子目录 | 根目录平铺 |
| 图片命名 | `original.png` / `translated.png` | `image_{idx}_original.png` |
| 页面元数据 | 独立 `meta.json` | 嵌入 `images_meta` 数组 |
| 支持增量保存 | ✅ 是 | ❌ 否 |
| 还在使用 | ✅ 保存 + 加载 | ⚠️ 仅加载 |

---

## 相关模块清单

### 核心存储模块

| 模块路径 | 职责 | 格式支持 |
|---------|------|----------|
| `src/core/page_storage.py` | 新格式保存/加载 | 新格式 |
| `src/core/session_manager.py` | 会话加载（含格式判断） | 新旧两种 |

### API 层

| 模块路径 | 职责 |
|---------|------|
| `src/app/api/page_storage_api.py` | 新格式保存/加载 REST API |
| `src/app/api/session_api.py` | 旧格式图片服务 + 通用会话 API |
| `src/app/api/bookshelf_api.py` | 阅读模式图片 URL 生成 |

### 使用存储数据的模块

| 模块路径 | 用途 | 需要格式兼容 |
|---------|------|-------------|
| `src/core/bookshelf_manager.py` | 统计章节页数 | ✅ |
| `src/core/manga_insight/analyzer.py` | 漫画分析获取图片 | ✅ |
| `src/app/api/manga_insight/data_routes.py` | 漫画分析图片服务 | ✅ |

---

## 格式判断逻辑

### 核心判断代码

所有需要兼容两种格式的地方，使用以下模式：

```python
# 读取 session_meta.json
with open(session_meta_path, "r", encoding="utf-8") as f:
    session_data = json.load(f)

# 判断格式（核心逻辑在 session_manager.load_session_by_path）
# 完整判断：有 total_pages 字段 且 存在 images/ 目录
images_dir = os.path.join(session_dir, "images")
is_new_format = "total_pages" in session_data and os.path.isdir(images_dir)

if is_new_format:
    # 新格式
    image_count = session_data.get("total_pages", 0)
    # 图片路径: images/{idx}/original.png
else:
    # 旧格式
    images_meta = session_data.get("images_meta", [])
    image_count = len(images_meta)
    # 图片路径: image_{idx}_original.png
```

> **注意**：在简化场景（如统计页数）中，可以只检查 `"total_pages" in session_data`，
> 但在需要加载图片的场景中，应同时检查 `images/` 目录是否存在。

### 判断逻辑所在位置

需要修改格式时，以下位置都有格式判断代码：

| 位置 | 文件 | 行号范围 | 用途 |
|------|-----|---------|------|
| 1 | `session_manager.py` | `load_session_by_path()` | 分发到新旧加载函数 |
| 2 | `session_manager.py` | `list_sessions()` | 列表显示图片数量 |
| 3 | `bookshelf_manager.py` | `get_book()` | 统计章节页数 |
| 4 | `analyzer.py` | `get_book_info()` | 漫画分析图片列表 |
| 5 | `data_routes.py` | `get_page_thumbnail()` | 获取缩略图 |
| 6 | `data_routes.py` | `get_page_image()` | 获取原图 |

---

## 数据流向

### 保存流程（新格式）

```
前端 SessionStore
    │
    ▼ POST /api/sessions/presave/{session_path}
page_storage_api.py
    │ 调用
    ▼
page_storage.presave_all_pages()
    │ 循环调用
    ├─▶ save_page_image(original)
    ├─▶ save_page_image(clean)
    ├─▶ save_page_image(translated)
    └─▶ save_page_meta()
    │
    ▼
page_storage.save_session_meta()
    │ 写入
    ▼
磁盘文件
```

### 加载流程（书架模式）

```
前端 SessionStore
    │
    ▼ GET /api/sessions/load_by_path?session_path=...
session_api.load_session_by_path_api()
    │ 调用
    ▼
session_manager.load_session_by_path()
    │ 判断格式
    ├─▶ _load_new_format_session()  →  返回 URL 格式
    └─▶ _load_old_format_session()  →  返回 URL 格式
    │
    ▼ 返回给前端
前端获取 URL，请求图片
    │
    ├─▶ GET /api/sessions/page/{path}/{idx}/{type}      (新格式)
    │       └─▶ page_storage_api.api_load_page_image()
    │
    └─▶ GET /api/sessions/image_by_path/{path}/{file}   (旧格式)
            └─▶ session_api.get_session_image_by_path()
```

### 阅读模式流程

```
前端 ReaderView
    │
    ▼ GET /api/bookshelf/books/{book}/chapters/{chapter}/images
bookshelf_api.get_chapter_images()
    │ 调用
    ▼
session_manager.load_session_by_path()
    │ 返回带 URL 的数据
    ▼
提取 originalDataURL / translatedDataURL
    │
    ▼ 前端直接使用 URL 显示图片
```

---

## 修改指南

### 场景 1：修改保存格式

如果需要修改新格式的保存结构：

1. **修改 `page_storage.py`**
   - 调整 `save_page_image()` 的路径生成
   - 调整 `save_session_meta()` 的元数据结构

2. **修改 `session_manager.py`**
   - 更新 `_load_new_format_session()` 的加载逻辑
   - 确保 URL 生成正确

3. **修改 API 路由** (如果路径变化)
   - 更新 `page_storage_api.py` 的路由模式

### 场景 2：添加新的存储字段

1. **保存时**：修改 `page_storage.py` 的保存函数
2. **加载时**：修改 `session_manager.py` 的加载函数
3. **前端**：更新 `sessionStore.ts` 和 `types/image.ts`

### 场景 3：弃用旧格式

1. **步骤 1**：创建迁移脚本
   ```python
   # 将 image_N_type.png 移动到 images/N/type.png
   # 重写 session_meta.json 使用 total_pages
   ```

2. **步骤 2**：运行迁移

3. **步骤 3**：删除以下代码
   - `session_manager._load_old_format_session()`
   - `session_api.get_session_image_by_path()` 路由
   - 所有 `if "total_pages" in session_data` 分支的 else 部分

### 场景 4：添加新的存储功能

例如添加图片压缩：

1. 在 `page_storage.py` 添加压缩逻辑
2. 在 `session_meta.json` 添加 `compression` 字段
3. 加载时检测并解压

---

## 常见问题

### Q: 阅读模式图片 404

**原因**：`bookshelf_api.get_chapter_images()` 返回的 URL 格式不正确

**检查点**：
1. 确认 `session_manager.load_session_by_path()` 返回正确格式的 URL
2. 新格式应返回 `/api/sessions/page/{path}/{idx}/{type}`
3. 旧格式应返回 `/api/sessions/image_by_path/{path}/image_{idx}_{type}.png`

### Q: 漫画分析显示 0 页

**原因**：`analyzer.get_book_info()` 没有正确识别存档格式

**检查点**：
1. 检查是否有 `if "total_pages" in session_data` 的判断
2. 新格式需要在 `images/{idx}/` 查找图片

### Q: 保存后再加载数据丢失

**检查点**：
1. 确认 `page_storage.save_page_meta()` 保存了所有字段
2. 确认 `session_manager._load_new_format_session()` 正确读取了元数据
3. 检查前端是否正确序列化了 `bubbleStates`

---

## 附录：关键函数速查

### page_storage.py

| 函数 | 用途 |
|------|------|
| `save_session_meta()` | 保存会话元数据 |
| `load_session_meta()` | 加载会话元数据 |
| `save_page_image()` | 保存单页图片 |
| `load_page_image()` | 加载单页图片 |
| `save_page_meta()` | 保存单页元数据 |
| `load_page_meta()` | 加载单页元数据 |
| `presave_all_pages()` | 批量预保存 |
| `save_translated_page()` | 保存翻译后的页面 |

### session_manager.py

| 函数 | 用途 |
|------|------|
| `load_session()` | 非书架模式会话加载 |
| `load_session_by_path()` | 书架模式会话加载（入口） |
| `_load_new_format_session()` | 新格式加载实现 |
| `_load_old_format_session()` | 旧格式加载实现 |
| `list_sessions()` | 列出所有会话 |

### data_routes.py（漫画分析）

| 函数 | 用途 |
|------|------|
| `_find_image_path()` | 查找图片路径（兼容两种格式） |
| `get_page_thumbnail()` | 获取缩略图 |
| `get_page_image()` | 获取原图 |

---

## 变更历史

| 日期 | 变更内容 |
|------|---------|
| 2026-01-19 | 创建文档，记录双格式兼容架构 |
