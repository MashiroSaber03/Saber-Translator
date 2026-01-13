# 翻译、保存、加载逻辑开发手册

本文档描述了 Saber-Translator 中翻译、保存、加载功能的架构设计和实现细节。

## 目录

1. [架构概述](#架构概述)
2. [存储结构](#存储结构)
3. [保存流程](#保存流程)
4. [加载流程](#加载流程)
5. [API 接口](#api-接口)
6. [前端函数](#前端函数)
7. [使用示例](#使用示例)
8. [注意事项](#注意事项)

---

## 架构概述

### 设计原则

1. **单页独立存储**：每张图片独立保存为文件，避免大请求和并发冲突
2. **增量更新**：只更新变化的内容，不重写整个会话
3. **进度可视**：保存和预保存过程都显示详细进度
4. **向后兼容**：支持加载旧格式存档

### 模块分布

```
后端:
├── src/core/page_storage.py        # 单页存储管理模块
├── src/core/session_manager.py     # 会话管理（加载 + 旧格式兼容）
├── src/app/api/page_storage_api.py # 单页存储 API
└── src/app/api/session_api.py      # 会话管理 API

前端:
├── vue-frontend/src/api/pageStorage.ts           # 单页存储 API
├── vue-frontend/src/api/session.ts               # 会话管理 API
├── vue-frontend/src/composables/translation/core/saveStep.ts  # 自动保存逻辑
└── vue-frontend/src/stores/sessionStore.ts       # 会话状态管理
```

---

## 存储结构

### 新版存储结构（推荐）

```
data/sessions/{session_path}/
├── session_meta.json              # 会话元数据
└── images/
    ├── 0/                         # 第1页 (0-indexed)
    │   ├── original.png           # 原图
    │   ├── translated.png         # 译图
    │   ├── clean.png              # 干净背景
    │   └── meta.json              # 页面元数据
    ├── 1/                         # 第2页
    │   ├── original.png
    │   ├── translated.png
    │   ├── clean.png
    │   └── meta.json
    └── ...
```

### session_meta.json 结构

```json
{
  "metadata": {
    "name": "bookshelf/book123/chapter1",
    "saved_at": "2026-01-14 00:00:00",
    "version": "3.0.0"
  },
  "ui_settings": {
    "fontSize": 24,
    "autoFontSize": true,
    "fontFamily": "思源黑体SourceHanSansK-Bold.TTF",
    "layoutDirection": "auto",
    "textColor": "#000000",
    "fillColor": "#FFFFFF",
    "strokeEnabled": false,
    "strokeColor": "#FFFFFF",
    "strokeWidth": 2,
    "useAutoTextColor": false,
    "useInpaintingMethod": "solid"
  },
  "total_pages": 52,
  "currentImageIndex": 0
}
```

### 页面 meta.json 结构

```json
{
  "fileName": "page_001.png",
  "translationStatus": "completed",
  "translationFailed": false,
  "bubbleStates": [...],
  "isManuallyAnnotated": false,
  "relativePath": "chapter1/page_001.png",
  "folderPath": "chapter1",
  "fontSize": 24,
  "autoFontSize": true,
  "fontFamily": "思源黑体SourceHanSansK-Bold.TTF",
  "layoutDirection": "auto",
  "useAutoTextColor": false,
  "textColor": "#000000",
  "fillColor": "#FFFFFF",
  "inpaintMethod": "solid",
  "strokeEnabled": false,
  "strokeColor": "#FFFFFF",
  "strokeWidth": 2
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `useAutoTextColor` | boolean | 是否启用自动文字颜色（根据背景自动选择） |
| `useInpaintingMethod` / `inpaintMethod` | string | 气泡填充方式：<br>• `solid` = 纯色填充<br>• `lama_mpe` = LAMA修复 (速度优化)<br>• `litelama` = LAMA修复 (通用) |
| `layoutDirection` | string | 文字排版方向：`auto` / `vertical` / `horizontal` |

> **注意**：语言设置 (`targetLanguage`, `sourceLanguage`) 不保存在会话中，由设置菜单统一管理。

### 旧版存储结构（兼容）

```
data/sessions/{session_path}/
├── session_meta.json
├── image_0_original.png
├── image_0_translated.png
├── image_0_clean.png
├── image_1_original.png
├── ...
└── images_meta (包含所有页面的元数据)
```

系统会自动检测存储格式并使用相应的加载逻辑。

---

## 保存流程

### 翻译时的自动保存（书架模式）

```
┌─────────────────────────────────────────────────────────────┐
│                     翻译开始                                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 预保存阶段 (preSaveOriginalImages)                       │
│    - 逐页保存原图、已有译图、元数据                          │
│    - 更新书架章节图片数量                                    │
│    - 显示进度: "预保存原始图片 39/198..."                    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 翻译阶段                                                 │
│    顺序模式: 每页翻译完成后 → saveTranslatedImage()          │
│    并行模式: RenderPool 完成后 → saveTranslatedImage()       │
│    只保存译图和干净背景                                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 完成阶段 (finalizeSave)                                  │
│    - 更新会话元数据                                          │
│    - 更新书架章节图片数量（双保险）                          │
└─────────────────────────────────────────────────────────────┘
```

### 手动保存（点击保存按钮）

```
┌─────────────────────────────────────────────────────────────┐
│ saveChapterSession() 或 handleSave()                        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 转换 URL 格式图片为 Base64                                │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 逐页保存 (saveAllPagesSequentially)                      │
│    - 显示进度: "保存图片 39/198..."                          │
│    - 保存原图、译图、干净背景、元数据                        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 保存会话元数据                                            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 更新书架章节图片数量                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 加载流程

```
┌─────────────────────────────────────────────────────────────┐
│ loadSessionByPath(sessionPath)                              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 后端: load_session_by_path()                                │
│ - 检测存储格式                                               │
│   - 有 total_pages + images/ 目录 → 新格式                   │
│   - 否则 → 旧格式                                            │
└─────────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│ 新格式加载               │  │ 旧格式加载               │
│ _load_new_format_session │  │ _load_old_format_session │
│ - 读取 session_meta.json │  │ - 读取旧版元数据         │
│ - 遍历 images/{index}/   │  │ - 拼接图片 URL           │
│ - 生成图片 URL           │  │                         │
└─────────────────────────┘  └─────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 返回数据结构:                                                │
│ {                                                           │
│   "ui_settings": {...},                                     │
│   "images": [                                               │
│     {                                                       │
│       "originalDataURL": "/api/sessions/page/.../0/original"│
│       "translatedDataURL": "/api/sessions/page/.../0/trans" │
│       "cleanImageData": "/api/sessions/page/.../0/clean"    │
│       ...页面元数据                                          │
│     },                                                      │
│     ...                                                     │
│   ],                                                        │
│   "currentImageIndex": 0                                    │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## API 接口

### 后端 API（page_storage_api.py）

| 方法 | 路由 | 说明 |
|------|------|------|
| POST | `/api/sessions/meta/{path}` | 保存会话元数据 |
| GET | `/api/sessions/meta/{path}` | 加载会话元数据 |
| POST | `/api/sessions/page/{path}/{index}/{type}` | 保存单页图片 |
| GET | `/api/sessions/page/{path}/{index}/{type}` | 加载单页图片 |
| POST | `/api/sessions/page/{path}/{index}/meta` | 保存页面元数据 |
| GET | `/api/sessions/page/{path}/{index}/meta` | 加载页面元数据 |
| POST | `/api/sessions/presave/{path}` | 批量预保存（备用） |
| POST | `/api/sessions/save_translated/{path}/{index}` | 保存翻译完成的页面 |
| GET | `/api/sessions/load/{path}` | 加载完整会话 |

### 参数说明

- `path`: 会话路径，如 `bookshelf/book123/chapter1`
- `index`: 页面索引，0-based
- `type`: 图片类型，`original` | `translated` | `clean`

### 请求体示例

**保存单页图片：**
```json
{
  "data": "base64编码的图片数据..."
}
```

**保存翻译完成的页面：**
```json
{
  "translated": "base64编码的译图...",
  "clean": "base64编码的干净背景...",
  "meta": {
    "translationStatus": "completed",
    "bubbleStates": [...]
  }
}
```

---

## 前端函数

### pageStorage.ts

```typescript
// 保存会话元数据
saveSessionMeta(sessionPath: string, metadata: object): Promise<{success: boolean}>

// 保存单页图片
savePageImage(sessionPath: string, pageIndex: number, imageType: string, base64Data: string): Promise<{success: boolean}>

// 保存页面元数据
savePageMeta(sessionPath: string, pageIndex: number, meta: object): Promise<{success: boolean}>

// 保存翻译完成的页面（译图 + 干净背景 + 元数据）
saveTranslatedPage(sessionPath: string, pageIndex: number, data: {translated?, clean?, meta?}): Promise<{success: boolean}>

// 逐页保存所有图片（公共函数，带进度回调）
saveAllPagesSequentially(sessionPath: string, images: ImageDataForSave[], callback?: SaveProgressCallback): Promise<number>

// 提取纯 Base64 数据（去掉 data:image/...;base64, 前缀）
extractBase64(dataUrl: string | null | undefined): string | null
```

### saveStep.ts

```typescript
// 检查是否启用自动保存
shouldEnableAutoSave(): boolean

// 获取当前会话路径
getSessionPath(): string | null

// 预保存所有页面（翻译前调用）
preSaveOriginalImages(progressCallback?: PreSaveProgressCallback): Promise<boolean>

// 保存翻译完成的页面（每页翻译完成后调用）
saveTranslatedImage(pageIndex: number): Promise<void>

// 完成保存（翻译结束后调用）
finalizeSave(): Promise<void>

// 重置保存状态（取消翻译时调用）
resetSaveState(): void
```

### sessionStore.ts

```typescript
// 按路径加载会话
loadSessionByPath(sessionPath: string): Promise<boolean>

// 保存章节会话（书架模式手动保存）
saveChapterSession(): Promise<boolean>
```

---

## 使用示例

### 翻译时自动保存（顺序管线）

```typescript
// SequentialPipeline.ts
import { shouldEnableAutoSave, preSaveOriginalImages, saveTranslatedImage, finalizeSave } from './saveStep'

async function execute() {
    const enableAutoSave = shouldEnableAutoSave()
    
    // 1. 预保存
    if (enableAutoSave) {
        await preSaveOriginalImages({
            onProgress: (current, total) => {
                reporter.setPercentage(current / total * 10, `预保存原始图片 ${current}/${total}...`)
            }
        })
    }
    
    // 2. 翻译过程
    for (const image of images) {
        await translateImage(image)
        
        // 每页翻译完成后保存
        if (enableAutoSave) {
            await saveTranslatedImage(image.index)
        }
    }
    
    // 3. 完成保存
    if (enableAutoSave) {
        await finalizeSave()
    }
}
```

### 手动保存会话

```typescript
// SessionModal.vue
import { saveAllPagesSequentially, saveSessionMeta } from '@/api/pageStorage'

async function handleSave() {
    const savedCount = await saveAllPagesSequentially(
        sessionName,
        imageStore.images,
        {
            onProgress: (current, total) => {
                saveProgress.value = { current, total }
            }
        }
    )
    
    await saveSessionMeta(sessionName, {
        ui_settings: uiSettings,
        total_pages: totalImages,
        currentImageIndex: imageStore.currentImageIndex
    })
}
```

### 加载会话

```typescript
// sessionStore.ts
async function loadSessionByPath(sessionPath: string) {
    const { loadSessionByPathApi } = await import('@/api/session')
    const response = await loadSessionByPathApi(sessionPath)
    
    if (response.success && response.session) {
        // 转换图片数据到 ImageData 格式
        const images = response.session.images.map(img => ({
            originalDataURL: img.originalDataURL,       // URL 格式
            translatedDataURL: img.translatedDataURL,   // URL 格式
            cleanImageData: img.cleanImageData,         // URL 格式
            ...其他元数据
        }))
        
        imageStore.setImages(images)
    }
}
```

---

## 注意事项

### 1. Base64 vs URL

- **保存时**：需要提取纯 Base64 数据，使用 `extractBase64()` 去掉 `data:image/...;base64,` 前缀
- **加载时**：返回的是 URL 格式 `/api/sessions/page/...`，可直接用于 `<img>` 标签

### 2. 图片格式转换

如果图片是 URL 格式（如 `/api/sessions/page/...`），需要先转换为 Base64 才能保存：

```typescript
async function convertImagesToBase64(images) {
    for (const img of images) {
        if (img.originalDataURL?.startsWith('/api/')) {
            const response = await fetch(img.originalDataURL)
            const blob = await response.blob()
            img.originalDataURL = await blobToBase64(blob)
        }
    }
}
```

### 3. 并发安全

后端使用线程锁保护写入操作：

```python
# page_storage.py
_locks: Dict[str, threading.Lock] = {}

def _get_lock(key: str) -> threading.Lock:
    with _locks_lock:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]
```

### 4. 原子写入

使用临时文件 + rename 实现原子写入，防止写入中断导致文件损坏：

```python
def _safe_write_json(filepath: str, data: dict):
    temp_path = filepath + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, filepath)  # 原子操作
```

### 5. 图片数量更新

在以下时机更新书架章节的图片数量：
- 预保存完成后 (`preSaveOriginalImages`)
- 完成保存时 (`finalizeSave`)
- 手动保存时 (`saveChapterSession`)

```typescript
await apiClient.put(`/api/bookshelf/books/${bookId}/chapters/${chapterId}/image-count`, {
    count: totalImages
})
```

### 6. 进度显示

保存和预保存过程都应该显示进度：

```typescript
// 预保存进度
progressCallback?.onProgress?.(current, total)

// 手动保存进度
loadingProgress.value = {
    current,
    total,
    message: `保存图片 ${current}/${total}...`
}
```

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `src/core/page_storage.py` | 后端单页存储核心实现 |
| `src/core/session_manager.py` | 后端会话加载（新旧格式兼容） |
| `src/app/api/page_storage_api.py` | 后端单页存储 API 路由 |
| `src/app/api/session_api.py` | 后端会话管理 API 路由 |
| `vue-frontend/src/api/pageStorage.ts` | 前端单页存储 API |
| `vue-frontend/src/api/session.ts` | 前端会话管理 API |
| `vue-frontend/src/composables/translation/core/saveStep.ts` | 自动保存逻辑 |
| `vue-frontend/src/composables/translation/core/pipeline.ts` | 并行模式入口 |
| `vue-frontend/src/composables/translation/core/SequentialPipeline.ts` | 顺序模式管线 |
| `vue-frontend/src/stores/sessionStore.ts` | 会话状态管理 |

---

## 更新日志

| 日期 | 版本 | 内容 |
|------|------|------|
| 2026-01-14 | v1.0 | 初始版本，描述新的单页独立存储架构 |
