# 漫画翻译文字设置开发手册

本文档描述了 Saber-Translator 中文字设置（字体、字号、颜色、描边等）的架构设计和实现细节。

## 目录

1. [架构概述](#架构概述)
2. [数据结构](#数据结构)
3. [三层数据模型](#三层数据模型)
4. [翻译流程中的文字设置](#翻译流程中的文字设置)
5. [侧边栏与图片的同步](#侧边栏与图片的同步)
6. [编辑模式中的气泡修改](#编辑模式中的气泡修改)
7. [应用到全部功能](#应用到全部功能)
8. [会话保存与加载](#会话保存与加载)
9. [关键文件](#关键文件)
10. [注意事项](#注意事项)

---

## 架构概述

### 设计原则

1. **三层数据模型**：全局设置 → 图片级别设置 → 气泡级别设置
2. **渲染以气泡为准**：实际渲染始终使用每个气泡的独立样式
3. **批量翻译锁定设置**：翻译开始时保存当前设置，确保所有图片使用一致的设置
4. **编辑模式独立**：编辑模式中修改单个气泡不影响其他气泡

### 核心流程

```
┌─────────────────────────────────────────────────────────────┐
│                 用户在侧边栏修改文字设置                       │
│                    (settingsStore.textStyle)                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 批量翻译时：                                              │
│    saveCurrentStyles() → 保存到 savedTextStyles              │
│    → 翻译过程中使用 savedTextStyles                          │
│    → 保存到每张图片的 ImageData 和每个气泡的 BubbleState       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 渲染时：                                                  │
│    始终使用 BubbleState 中的独立样式 (use_individual_styles)  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 切换图片时：                                              │
│    syncImageToSidebar() → 从 ImageData 恢复到侧边栏           │
└─────────────────────────────────────────────────────────────┘
```

---

## 数据结构

### 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fontSize` | `number` | `25` | 字号大小 |
| `autoFontSize` | `boolean` | `false` | 是否自动计算字号 |
| `fontFamily` | `string` | `fonts/思源黑体...` | 字体文件路径 |
| `layoutDirection` | `'auto' \| 'vertical' \| 'horizontal'` | `'auto'` | 排版方向 |
| `textColor` | `string` | `'#000000'` | 文字颜色 |
| `fillColor` | `string` | `'#FFFFFF'` | 填充颜色（背景） |
| `strokeEnabled` | `boolean` | `true` | 是否启用描边 |
| `strokeColor` | `string` | `'#FFFFFF'` | 描边颜色 |
| `strokeWidth` | `number` | `3` | 描边宽度 |
| `inpaintMethod` | `'solid' \| 'lama_mpe' \| 'litelama'` | `'solid'` | 修复方式 |
| `useAutoTextColor` | `boolean` | `false` | 是否使用自动检测的颜色 |

### TextStyleSettings 类型 (settingsStore)

```typescript
// stores/settings/types.ts
interface TextStyleSettings {
  fontSize: number
  autoFontSize: boolean
  fontFamily: string
  layoutDirection: 'auto' | 'vertical' | 'horizontal'
  textColor: string
  fillColor: string
  strokeEnabled: boolean
  strokeColor: string
  strokeWidth: number
  inpaintMethod: 'solid' | 'lama_mpe' | 'litelama'
  useAutoTextColor: boolean
}
```

### ImageData 类型 (图片级别)

```typescript
// types/image.ts
interface ImageData {
  // ... 其他字段

  // 图片级别设置
  fontSize: number
  autoFontSize: boolean
  fontFamily: string
  layoutDirection: TextDirection
  textColor: string
  fillColor: string
  inpaintMethod: InpaintMethod
  strokeEnabled: boolean
  strokeColor: string
  strokeWidth: number
  useAutoTextColor?: boolean
}
```

### BubbleState 类型 (气泡级别)

```typescript
// types/bubble.ts
interface BubbleState {
  // 文本内容
  originalText: string
  translatedText: string
  textboxText: string

  // 坐标信息
  coords: BubbleCoords
  polygon: PolygonCoords

  // 渲染参数
  fontSize: number
  fontFamily: string
  textDirection: TextDirection       // 实际渲染方向
  autoTextDirection: TextDirection  // 自动检测的方向（备份）
  textColor: string
  fillColor: string
  rotationAngle: number
  position: BubblePosition

  // 描边参数
  strokeEnabled: boolean
  strokeColor: string
  strokeWidth: number

  // 修复参数
  inpaintMethod: InpaintMethod

  // 自动颜色提取
  autoFgColor?: [number, number, number] | null
  autoBgColor?: [number, number, number] | null
}
```

---

## 三层数据模型

```
┌─────────────────────────────────────────────────────────────┐
│ 第1层: settingsStore.settings.textStyle                     │
│                                                             │
│ 用途: 左侧边栏 UI 显示/编辑                                  │
│ 持久化: 不持久化到 localStorage，每次刷新重置为默认值          │
│ 同步: 通过 watcher 与当前图片的 ImageData 双向同步            │
└─────────────────────────────────────────────────────────────┘
                           │
                    (syncSidebarToImage)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 第2层: ImageData (imageStore)                               │
│                                                             │
│ 用途: 每张图片保存自己的设置副本                             │
│ 持久化: 会话保存时持久化到 page meta.json                    │
│ 同步: 翻译时从 sidebar 同步，切换图片时恢复到 sidebar         │
└─────────────────────────────────────────────────────────────┘
                           │
                    (翻译时构建)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 第3层: BubbleState (bubbleStore / ImageData.bubbleStates)   │
│                                                             │
│ 用途: 每个气泡的实际渲染样式                                 │
│ 持久化: 随 ImageData.bubbleStates 一起保存                  │
│ 渲染: 后端使用 use_individual_styles=true 逐个气泡渲染       │
└─────────────────────────────────────────────────────────────┘
```

---

## 翻译流程中的文字设置

### 顺序翻译 (SequentialPipeline)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 翻译开始: saveCurrentStyles()                            │
│    - 从 settingsStore.settings.textStyle 读取所有设置        │
│    - 保存到 savedTextStyles 变量                            │
│    - 后续所有图片使用这个锁定的设置                          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 检测/OCR/翻译步骤                                        │
│    - 使用 settingsStore 中的 OCR、翻译服务设置               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 渲染步骤: executeRender()                                │
│    - 为每个气泡构建 BubbleState，使用 savedTextStyles        │
│    - 处理 layoutDirection='auto' → 使用检测结果              │
│    - 处理 useAutoTextColor=true → 使用自动提取的颜色          │
│    - 调用后端 parallelRender() 渲染                          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 保存结果: updateImageStore()                             │
│    - 保存 bubbleStates 到 ImageData                         │
│    - 保存所有文字设置字段到 ImageData                         │
└─────────────────────────────────────────────────────────────┘
```

### SavedTextStyles 接口

```typescript
// composables/translation/core/types.ts
interface SavedTextStyles {
  fontFamily: string
  fontSize: number
  autoFontSize: boolean
  textDirection: string
  autoTextDirection: boolean
  layoutDirection: 'auto' | 'vertical' | 'horizontal'
  fillColor: string
  textColor: string
  rotationAngle: number
  strokeEnabled: boolean
  strokeColor: string
  strokeWidth: number
  useAutoTextColor: boolean
  inpaintMethod: 'solid' | 'lama_mpe' | 'litelama'
}
```

### 关键代码: saveCurrentStyles()

```typescript
// SequentialPipeline.ts
function saveCurrentStyles(): void {
  const { textStyle } = settingsStore.settings
  const layoutDirectionValue = textStyle.layoutDirection
  savedTextStyles = {
    fontFamily: textStyle.fontFamily,
    fontSize: textStyle.fontSize,
    autoFontSize: textStyle.autoFontSize,
    autoTextDirection: layoutDirectionValue === 'auto',
    textDirection: layoutDirectionValue === 'auto' ? 'vertical' : layoutDirectionValue,
    layoutDirection: layoutDirectionValue,
    fillColor: textStyle.fillColor,
    textColor: textStyle.textColor,
    rotationAngle: 0,
    strokeEnabled: textStyle.strokeEnabled,
    strokeColor: textStyle.strokeColor,
    strokeWidth: textStyle.strokeWidth,
    useAutoTextColor: textStyle.useAutoTextColor,
    inpaintMethod: textStyle.inpaintMethod
  }
}
```

---

## 侧边栏与图片的同步

### 同步机制

```typescript
// TranslateView.vue

// 防止循环触发的标志
const isSyncingTextStyle = ref(false)

// 从图片同步到侧边栏
function syncImageToSidebar(image: ImageData) {
  if (!image) return
  const currentStyle = settingsStore.settings.textStyle
  
  settingsStore.updateTextStyle({
    fontSize: image.fontSize ?? currentStyle.fontSize,
    autoFontSize: image.autoFontSize ?? currentStyle.autoFontSize,
    fontFamily: image.fontFamily ?? currentStyle.fontFamily,
    layoutDirection: image.layoutDirection ?? currentStyle.layoutDirection,
    textColor: image.textColor ?? currentStyle.textColor,
    fillColor: image.fillColor ?? currentStyle.fillColor,
    strokeEnabled: image.strokeEnabled ?? currentStyle.strokeEnabled,
    strokeColor: image.strokeColor ?? currentStyle.strokeColor,
    strokeWidth: image.strokeWidth ?? currentStyle.strokeWidth,
    inpaintMethod: image.inpaintMethod ?? currentStyle.inpaintMethod,
    useAutoTextColor: image.useAutoTextColor ?? currentStyle.useAutoTextColor
  })
}

// 从侧边栏同步到图片
function syncSidebarToImage(style: TextStyleSettings) {
  const currentImg = imageStore.currentImage
  if (!currentImg) return
  
  imageStore.updateCurrentImage({
    fontSize: style.fontSize,
    autoFontSize: style.autoFontSize,
    fontFamily: style.fontFamily,
    layoutDirection: style.layoutDirection,
    textColor: style.textColor,
    fillColor: style.fillColor,
    strokeEnabled: style.strokeEnabled,
    strokeColor: style.strokeColor,
    strokeWidth: style.strokeWidth,
    inpaintMethod: style.inpaintMethod,
    useAutoTextColor: style.useAutoTextColor
  })
}

// 监听图片切换
watch(
  () => imageStore.currentImage,
  (newImage) => {
    if (newImage && !isSyncingTextStyle.value) {
      isSyncingTextStyle.value = true
      try {
        syncImageToSidebar(newImage)
      } finally {
        isSyncingTextStyle.value = false
      }
    }
  }
)

// 监听侧边栏设置变化
watch(
  () => settingsStore.settings.textStyle,
  (newStyle) => {
    if (imageStore.currentImage && !isSyncingTextStyle.value) {
      isSyncingTextStyle.value = true
      try {
        syncSidebarToImage(newStyle)
      } finally {
        isSyncingTextStyle.value = false
      }
    }
  },
  { deep: true }
)
```

---

## 编辑模式中的气泡修改

### 核心原则

1. 编辑模式中**修改单个气泡**只影响该气泡的 BubbleState，不影响图片级别设置或其他气泡
2. 编辑模式中**添加新气泡**时，会从当前侧边栏读取所有样式设置

### 添加新气泡

新添加的气泡会自动继承当前侧边栏的所有样式设置：

```typescript
// bubbleStore.ts
function addBubble(coords: BubbleCoords, overrides?: Partial<BubbleState>): BubbleState {
  // 自动计算排版方向
  const autoDirection = detectTextDirection(coords)

  // 【复刻原版】从 settingsStore 读取当前 UI 设置
  const settingsStore = useSettingsStore()
  const textStyle = settingsStore.settings.textStyle

  // 处理排版方向：如果是 'auto' 则使用检测结果
  const layoutDirection = textStyle.layoutDirection
  const bubbleTextDirection =
    (layoutDirection === 'vertical' || layoutDirection === 'horizontal')
      ? layoutDirection
      : autoDirection

  const newBubble = createBubbleState({
    coords,
    translatedText: '',
    autoTextDirection: autoDirection,
    // 从当前侧边栏读取所有样式设置
    fontSize: textStyle.fontSize,
    fontFamily: textStyle.fontFamily,
    textDirection: bubbleTextDirection,
    textColor: textStyle.textColor,
    fillColor: textStyle.fillColor,
    inpaintMethod: textStyle.inpaintMethod,
    strokeEnabled: textStyle.strokeEnabled,
    strokeColor: textStyle.strokeColor,
    strokeWidth: textStyle.strokeWidth,
    rotationAngle: 0,
    position: { x: 0, y: 0 },
    // 允许 overrides 覆盖
    ...overrides
  })

  bubbles.value.push(newBubble)
  syncToCurrentImage()  // 同步到 imageStore
  return newBubble
}
```

**继承的设置项：**

| 设置项 | 来源 |
|--------|------|
| `fontSize` | `textStyle.fontSize` |
| `fontFamily` | `textStyle.fontFamily` |
| `textDirection` | 根据 `textStyle.layoutDirection` 和气泡宽高比自动计算 |
| `textColor` | `textStyle.textColor` |
| `fillColor` | `textStyle.fillColor` |
| `strokeEnabled` | `textStyle.strokeEnabled` |
| `strokeColor` | `textStyle.strokeColor` |
| `strokeWidth` | `textStyle.strokeWidth` |
| `inpaintMethod` | `textStyle.inpaintMethod` |

### 气泡更新流程

```typescript
// bubbleStore.ts
function updateBubble(index: number, updates: BubbleStateUpdates): boolean {
  const bubble = bubbles.value[index]
  if (bubble) {
    Object.assign(bubble, updates)  // 只更新指定气泡
    syncToCurrentImage()            // 同步到 imageStore
    return true
  }
  return false
}

// 同步到 imageStore
function syncToCurrentImage(): void {
  const imageStore = useImageStore()
  const currentImage = imageStore.currentImage
  if (currentImage) {
    currentImage.bubbleStates = cloneBubbleStates(bubbles.value)
    currentImage.hasUnsavedChanges = true
  }
}
```


---

## 应用到全部功能

### 功能说明

"应用到全部"功能允许用户将当前侧边栏的设置应用到所有已翻译的图片，并触发重渲染。

### 支持的设置项

| 设置项 | 说明 |
|--------|------|
| `fontSize` | 字号（支持自动字号） |
| `fontFamily` | 字体 |
| `layoutDirection` | 排版方向（支持自动） |
| `textColor` | 文字颜色（支持自动） |
| `fillColor` | 填充颜色（支持自动） |
| `strokeEnabled` | 描边开关 |
| `strokeColor` | 描边颜色 |
| `strokeWidth` | 描边宽度 |

> **注意**：`inpaintMethod` 不支持"应用到全部"，因为修复方式只在翻译时生效，已翻译的图片使用的是处理好的 `cleanImage`。

### 处理流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 遍历所有已翻译的图片                                      │
│    - 更新每个气泡的 BubbleState                              │
│    - 更新图片级别的 ImageData 设置                           │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 处理自动模式                                              │
│    - autoFontSize=true: 后端重新计算字号                     │
│    - layoutDirection='auto': 使用每个气泡的 autoTextDirection │
│    - useAutoTextColor=true: 使用每个气泡的 autoFgColor/autoBgColor │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 调用 parallelRender 重新渲染每张图片                       │
│    - 使用 use_individual_styles: true                        │
│    - 显示进度条                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 会话保存与加载

### 保存时

```typescript
// sessionStore.ts saveChapterSession()

const { textStyle } = settingsStore.settings
const uiSettings = {
  fontSize: textStyle.fontSize,
  autoFontSize: textStyle.autoFontSize,
  fontFamily: textStyle.fontFamily,
  layoutDirection: textStyle.layoutDirection,
  textColor: textStyle.textColor,
  fillColor: textStyle.fillColor,
  strokeEnabled: textStyle.strokeEnabled,
  strokeColor: textStyle.strokeColor,
  strokeWidth: textStyle.strokeWidth,
  useInpaintingMethod: textStyle.inpaintMethod,  // 注意：保存时键名为 useInpaintingMethod
  useAutoTextColor: textStyle.useAutoTextColor,
}
```

### 加载时

```typescript
// sessionStore.ts loadSessionByPath()

// 1. 恢复每张图片的设置 (ImageData 级别)
const images = sessionData.images.map((img, index) => ({
  fontSize: img.fontSize || 24,
  autoFontSize: img.autoFontSize ?? false,
  fontFamily: img.fontFamily || 'Microsoft YaHei',
  layoutDirection: img.layoutDirection || 'auto',
  textColor: img.textColor || '#000000',
  fillColor: img.fillColor || '#FFFFFF',
  inpaintMethod: img.inpaintMethod || 'litelama',  // 加载时默认 litelama
  strokeEnabled: img.strokeEnabled ?? false,
  strokeColor: img.strokeColor || '#FFFFFF',
  strokeWidth: img.strokeWidth || 2,
  useAutoTextColor: img.useAutoTextColor ?? false,
  // ...
}))

// 2. 恢复 UI 设置到侧边栏 (textStyle 级别)
settingsStore.updateTextStyle({
  fontSize: uiSettings.fontSize || defaults.fontSize,
  autoFontSize: uiSettings.autoFontSize ?? defaults.autoFontSize,
  // ...所有设置项
})
```

---

## 关键文件

| 文件 | 说明 |
|------|------|
| `stores/settings/types.ts` | TextStyleSettings 类型定义 |
| `stores/settings/defaults.ts` | 默认值定义 |
| `stores/settings/modules/misc.ts` | updateTextStyle 方法 |
| `stores/imageStore.ts` | ImageData 管理 |
| `stores/bubbleStore.ts` | BubbleState 管理、addBubble 方法 |
| `stores/sessionStore.ts` | 会话保存/加载逻辑 |
| `types/image.ts` | ImageData 类型定义 |
| `types/bubble.ts` | BubbleState 类型定义 |
| `utils/bubbleFactory.ts` | 气泡状态工厂函数、createBubbleState |
| `views/TranslateView.vue` | 同步逻辑、handleTextStyleChanged、handleApplyToAll |
| `composables/translation/core/types.ts` | SavedTextStyles 类型定义 |
| `composables/translation/core/SequentialPipeline.ts` | 翻译时的设置保存/应用逻辑 |
| `composables/translation/parallel/pools/RenderPool.ts` | 并行翻译的渲染逻辑 |

---

## 注意事项

### 1. textStyle 不持久化到 localStorage

```typescript
// stores/settings/index.ts loadFromStorage()

// 【复刻原版】左侧边栏文字设置始终使用默认值，不从 localStorage 恢复
settings.value.textStyle = { ...defaults.textStyle }
```

这是设计决策：防止意外设置影响下次翻译。

### 2. 批量翻译使用锁定的设置

```typescript
// SequentialPipeline.ts 中的使用模式

// ✅ 正确：使用 savedTextStyles
fontSize: savedTextStyles?.fontSize ?? textStyle.fontSize

// ❌ 错误：直接使用 textStyle（用户可能在翻译过程中修改）
fontSize: textStyle.fontSize
```

### 3. 渲染始终使用气泡级别样式

```typescript
// parallelRender API 调用
const response = await parallelRender({
  bubble_states: bubbleStates,       // 每个气泡的独立样式
  use_individual_styles: true,       // 必须设为 true
  // 以下参数作为默认值/fallback
  fontSize: textStyle.fontSize,
  fontFamily: textStyle.fontFamily,
  // ...
})
```

### 4. 自动设置的处理

| 自动设置 | 存储位置 | 使用方式 |
|----------|----------|----------|
| `autoFontSize` | ImageData, TextStyle | 传递给后端，由后端计算实际字号 |
| `layoutDirection='auto'` | TextStyle | 翻译时使用检测结果填充 `textDirection` |
| `useAutoTextColor` | ImageData, TextStyle | 翻译时使用 `autoFgColor`/`autoBgColor` |
| `autoTextDirection` | BubbleState | 自动检测结果的备份，切换回"自动"时使用 |
| `autoFgColor`/`autoBgColor` | BubbleState | 后端颜色提取结果 |

### 5. inpaintMethod 的特殊性

`inpaintMethod` 只在**首次翻译**时生效，因为它控制的是背景修复方式。已翻译的图片使用的是处理好的 `cleanImage`，修改 `inpaintMethod` 不会影响已有的 `cleanImage`。

如果用户希望改变某张图片的修复方式，需要重新翻译该图片。

---

## 更新日志

| 日期 | 版本 | 内容 |
|------|------|------|
| 2026-01-18 | v1.0 | 初始版本，描述文字设置的完整架构 |
| 2026-01-18 | v1.1 | 添加"添加新气泡"章节；修正会话保存/加载代码示例；补充关键文件列表 |
