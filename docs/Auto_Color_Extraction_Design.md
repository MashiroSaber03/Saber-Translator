# 自动色号提取功能 - 实现方案

## 📋 概述

基于 48px OCR 模型的颜色预测能力，实现**智能颜色识别功能**。

### 核心特性

1. **强制提取** - 翻译时自动提取所有气泡的文字和背景颜色
2. **灵活使用** - 用户可选择使用自动颜色、默认颜色或自定义颜色
3. **完整数据** - 始终保留完整的颜色信息，编辑时可随时切换

---

## 🎯 核心设计理念 ⭐

**强制提取 + 灵活使用**：颜色提取是自动进行的，开关只控制是否默认使用。

### 关键原则

```typescript
// ✅ 新方案
翻译时：总是提取颜色（除非失败）
数据字段：autoFgColor 定义为可选（兼容旧数据和手动气泡）
开关控制：是否默认使用自动颜色
编辑时：用户可随时切换（自动/默认/自定义）
```

### 设计逻辑

| 阶段 | 行为 | 说明 |
|------|------|------|
| **翻译** | 强制提取颜色 | 48px 模型分析所有气泡，提取文字色和背景色 |
| **初始化** | 根据设置填充 | `useAutoColor` 开启 → 用自动颜色<br>`useAutoColor` 关闭 → 用全局默认 |
| **编辑** | 自由切换 | 用户可在自动/默认/自定义之间随意切换 |

### 对比原方案

| 特性 | 原方案（可选提取） | ✅ 新方案（强制提取） |
|------|------------------|---------------------|
| **提取时机** | 用户勾选才提取 | **总是提取** |
| **开关含义** | "是否提取颜色" | **"是否默认使用自动颜色"** |
| **autoFgColor** | 可能为 null | **翻译时总是提取** （但字段定义为可选） |
| **默认值问题** | 需复杂处理 | **无需处理** |
| **编辑灵活性** | 未提取则无法使用 | **随时可切换** |
| **数据完整性** | 有缺失 | **总是完整** |

---

## 🗂️ 数据结构设计

### BubbleState 扩展

**现状**：当前 `BubbleState` **没有**颜色相关字段，这些是**全新添加**的字段。

**需要添加的字段**：

```typescript
// vue-frontend/src/types/bubble.ts

export interface BubbleState {
  // ========== 现有字段 ==========
  originalText: string
  translatedText: string
  textboxText: string
  coords: BubbleCoords
  polygon: PolygonCoords
  fontSize: number
  fontFamily: string
  textDirection: TextDirection
  autoTextDirection: TextDirection
  textColor: string                    // ← 已存在
  fillColor: string                    // ← 已存在
  rotationAngle: number
  position: BubblePosition
  strokeEnabled: boolean
  strokeColor: string
  strokeWidth: number
  inpaintMethod: InpaintMethod
  
  // ========== ✨ 新增字段 ==========
  /** 自动提取的文字颜色（RGB数组） */
  autoFgColor?: [number, number, number]    // ← 新增！可选
  /** 自动提取的背景颜色（RGB数组） */
  autoBgColor?: [number, number, number]    // ← 新增！可选
  /** 颜色提取置信度 0-1 */
  colorConfidence?: number                  // ← 新增！可选
  /** 颜色提取时间戳 */
  colorExtractedAt?: string                 // ← 新增！可选
}
```

**字段可选性说明**：

| 字段 | 可选性 | 原因 |
|------|--------|------|
| `textColor` | 必需 | 已存在，始终需要有效颜色 |
| `fillColor` | 必需 | 已存在，始终需要有效颜色 |
| `autoFgColor` | **可选** ✅ | 新增字段，旧数据没有，手动气泡也没有 |
| `autoBgColor` | **可选** ✅ | 新增字段，旧数据没有，手动气泡也没有 |
| `colorConfidence` | **可选** ✅ | 新增字段，仅在颜色提取成功时有值 |
| `colorExtractedAt` | **可选** ✅ | 新增字段，仅在颜色提取成功时有值 |

**重要**：虽然设计原则是"强制提取"，但字段定义必须可选，以支持：
1. ✅ 向后兼容旧数据
2. ✅ 手动创建的气泡
3. ✅ 颜色提取失败的情况
4. ✅ TypeScript 类型安全* - 允许用户一键恢复到自动颜色
3. **调试和审计** - 记录原始的自动提取值

---

### autoFgColor 的作用变化

**原方案（可选提取）**：
```typescript
autoFgColor?: [number, number, number] | null  // 可能不存在
```

**✅ 新方案（强制提取）**：
```typescript
autoFgColor?: [number, number, number]  // 总是存在！但定义为可选以兼容旧数据和手动气泡
```

`autoFgColor` **不参与渲染**，仅作为元数据，用于：

1. **显示来源信息** - 告诉用户"这个颜色是自动识别的"
2. **重新应用功能** - 允许用户一键恢复到自动颜色
3. **调试和审计** - 记录原始的自动提取值

---

### 🎨 颜色格式统一

**统一使用 Hex 格式**

**前端**：统一使用 Hex 格式（`#rrggbb`）
- ✅ 与 `<input type="color">` 完美兼容
- ✅ 简单易读，方便用户修改

**后端**：**直接支持 Hex 格式** ✅
```python
# src/shared/constants.py
DEFAULT_TEXT_COLOR = '#231816'  # ← Hex 格式

# PIL ImageDraw 直接支持 Hex
draw.text((x, y), text, fill='#ff0000')  # ✅ 支持
```

**结论**：前后端统一使用 Hex，无需格式转换！

---

### 🛠️ 颜色格式工具函数

为了处理自动颜色（RGB 数组）到 Hex 的转换，提供以下工具函数：

```typescript
// src/utils/colorUtils.ts

/**
 * 将 RGB 数组转换为 Hex 字符串
 * @example rgbArrayToHex([15, 20, 25]) => '#0f1419'
 */
export function rgbArrayToHex(rgb: [number, number, number]): string {
  const toHex = (n: number) => {
    const clamped = Math.max(0, Math.min(255, Math.round(n)))
    return clamped.toString(16).padStart(2, '0')
  }
  return `#${toHex(rgb[0])}${toHex(rgb[1])}${toHex(rgb[2])}`
}

/**
 * 将 Hex 字符串转换为 RGB 数组（保留以备用）
 * @example hexToRgbArray('#0f1419') => [15, 20, 25]
 */
export function hexToRgbArray(hex: string): [number, number, number] {
  const cleaned = hex.replace('#', '')
  const r = parseInt(cleaned.slice(0, 2), 16)
  const g = parseInt(cleaned.slice(2, 4), 16)
  const b = parseInt(cleaned.slice(4, 6), 16)
  return [r, g, b]
}

/**
 * 验证 Hex 颜色格式
 * @example isValidHex('#ff0000') => true
 */
export function isValidHex(hex: string): boolean {
  return /^#?[0-9A-Fa-f]{6}$/.test(hex)
}
```
---

## 🔄 完整数据流

### 1. 翻译时（颜色填充）

```typescript
// 后端返回
{
  "bubbles": [
    {
      "text": "こんにちは",
      "coords": [100, 200, 300, 400],
      "autoFgColor": [15, 20, 25],      // 自动提取的前景色
      "autoBgColor": [248, 250, 252],   // 自动提取的背景色
      "colorConfidence": 0.92
    }
  ]
}

// 前端处理（关键逻辑）
import { rgbArrayToHex } from '@/utils/colorUtils'
import { useSettingsStore } from '@/stores/settingsStore'

function createBubbleState(apiData, userSettings) {
  const bubble: BubbleState = {
    x: apiData.coords[0],
    y: apiData.coords[1],
    width: apiData.coords[2] - apiData.coords[0],
    height: apiData.coords[3] - apiData.coords[1],
    originalText: apiData.text,
    translatedText: apiData.translated || '',
    // ... 其他字段
    
    // ✨ 颜色填充逻辑（强制提取）
    // 1. 保存自动提取的 RGB 数组（可能为 null）
    autoFgColor: apiData.autoFgColor || null,
    autoBgColor: apiData.autoBgColor || null,
    colorConfidence: apiData.colorConfidence || 0,
    colorExtractedAt: apiData.autoFgColor ? new Date().toISOString() : undefined,
    
    // 2. 根据用户设置决定初始使用什么颜色  
    textColor: (apiData.autoFgColor && userSettings.useAutoFgColorByDefault)
      ? rgbArrayToHex(apiData.autoFgColor)           // 使用自动颜色
      : (userSettings.defaultTextColor || '#000000'), // 使用全局默认
      
    fillColor: (apiData.autoBgColor && userSettings.useAutoBgColorByDefault)
      ? rgbArrayToHex(apiData.autoBgColor)           // 使用自动颜色
      : (userSettings.defaultFillColor || '#FFFFFF') // 使用全局默认
  }
  
  return bubble
}
```

**关键点**：
1. ✅ 翻译时总是提取颜色（字段定义为可选以兼容旧数据）
2. ✅ 开关控制 `useAutoFgColorByDefault`（是否默认使用）
3. ✅ 用户可在编辑时随时切换

### 2. 编辑时（用户修改）

```typescript
// 用户直接修改 textColor（无需特殊处理）
function handleTextColorChange(newColor: string) {
  if (isValidHex(newColor)) {
    bubble.textColor = newColor  // 直接更新
    // autoFgColor 保持不变（允许后续恢复）
  }
}

// 用户想恢复到自动颜色
function resetToAutoFgColor() {
  if (bubble.autoFgColor) {
    bubble.textColor = rgbArrayToHex(bubble.autoFgColor)
  }
}

// 用户想恢复到自动背景色
function resetToAutoBgColor() {
  if (bubble.autoBgColor) {
    bubble.fillColor = rgbArrayToHex(bubble.autoBgColor)
  }
}

// 检查当前是否使用自动颜色（用于 UI 显示）
function isUsingAutoFgColor() {
  if (!bubble.autoFgColor) return false
  const autoHex = rgbArrayToHex(bubble.autoFgColor)
  return bubble.textColor.toLowerCase() === autoHex.toLowerCase()
}
```

### 3. 渲染时（直接使用）

```typescript
// ✨ 极简渲染逻辑
function renderBubble(bubble: BubbleState) {
  renderText({
    text: bubble.translatedText,
    color: bubble.textColor,        // 直接用！
    backgroundColor: bubble.fillColor, // 直接用！
    fontSize: bubble.fontSize,
    // ...
  })
}

// 无需 getEffectiveColor() 等复杂判断！
```

### 4. 保存/加载（自动）

```typescript
// 切图时自动保存（无需修改）
function saveBubbleStatesToImage() {
  imageStore.updateCurrentBubbleStates([...bubbles.value])
  // textColor, fillColor, autoFgColor, autoBgColor 都一起保存
}

// 切回时自动加载（无需修改）
function loadBubbleStatesFromImage() {
  bubbleStore.setBubbles([...currentImage.value.bubbleStates])
  // 所有字段自动恢复
}
```

---

## 💻 后端实现

### 1. 颜色提取接口

```python
# src/core/color_extractor.py

class ColorExtractor:
    """48px 模型颜色提取器"""
    
    def extract_colors(
        self,
        image: Image.Image,
        bubble_coords: List[Tuple[int, int, int, int]],
        textlines_per_bubble: Optional[List[List[Dict]]] = None,
        extract_fg: bool = True,
        extract_bg: bool = True
    ) -> List[Dict]:
        """
        提取每个气泡的颜色
        
        Returns:
            [
                {
                    'fg_color': [r, g, b] or None,  # RGB 0-255
                    'bg_color': [r, g, b] or None,
                    'confidence': float
                },
                ...
            ]
        """
        # 复用 48px OCR 模型
        # 详细实现见核心算法章节
        pass
```

### 2. 后端集成 (src/core/processing.py)

```python
def process_image_translation(
    image_path: str,
    ocr_engine: str = 'manga_ocr',
    # ✨ 移除颜色提取开关（强制提取）
    **kwargs
):
    # ... 现有的检测和 OCR 逻辑 ...
    
    # OCR 识别
    bubble_results = recognize_bubbles(...)
    
    # ✨ 强制提取颜色（总是执行）
    from src.core.color_extractor import get_color_extractor
    
    extractor = get_color_extractor()
    if extractor.initialize(device):
        colors = extractor.extract_colors(
            image_pil, 
            bubble_coords, 
            textlines_per_bubble,
            extract_fg=True,  # 总是提取前景色
            extract_bg=True   # 总是提取背景色
        )
        
        # 直接将颜色附加到气泡数据中
        for i, color_info in enumerate(colors):
            if i < len(bubble_results):
                # ✨ 保证总是有颜色数据
                bubble_results[i]['autoFgColor'] = color_info['fg_color'] or [0, 0, 0]
                bubble_results[i]['autoBgColor'] = color_info['bg_color'] or [255, 255, 255]
                bubble_results[i]['colorConfidence'] = color_info['confidence']
    
    return {
        'bubbles': bubble_results,  # 每个 bubble 都包含颜色
        'translated_url': translated_url,
        ...
    }
```

**关键点**：
- ✅ 移除 `enable_auto_fg_color` 等参数
- ✅ 总是调用 `extract_colors()`
- ✅ 提供默认值兜底（黑色文字/白色背景）

---

## 🎨 前端实现

翻译和编辑时的颜色处理逻辑。

---

### 1. 翻译设置（全局配置）

**TranslateSettings.vue** - 控制是否默认使用自动颜色：

```vue
<template>
  <div class="translate-settings">
    <!-- OCR 设置 -->
    <div class="setting-section">
      <h3>OCR 识别</h3>
      <select v-model="settings.ocrEngine">
        <option value="manga_ocr">MangaOCR</option>
        <option value="paddle_ocr">PaddleOCR</option>
        <option value="48px_ocr">48px OCR</option>
      </select>
    </div>
    
    <!-- ✨ 颜色设置（改为"默认使用"） -->
    <div class="setting-section">
      <h3>智能颜色识别</h3>
      
      <div class="info-box">
        💡 翻译时会自动识别所有气泡的文字和背景颜色
      </div>
      
      <label>
        <input type="checkbox" v-model="settings.useAutoFgColorByDefault" />
        默认使用自动识别的文字颜色
      </label>
      
      <label>
        <input type="checkbox" v-model="settings.useAutoBgColorByDefault" />
        默认使用自动识别的背景颜色
      </label>
      
      <div class="hint">
        取消勾选时，翻译后的气泡会使用全局默认颜色<br>
        编辑时可随时切换为使用自动颜色
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { useSettingsStore } from '@/stores/settingsStore'

const settingsStore = useSettingsStore()
const { settings } = storeToRefs(settingsStore)
</script>

<style scoped>
.info-box {
  padding: 8px 12px;
  background: #e0f2fe;
  border-left: 3px solid #0ea5e9;
  border-radius: 4px;
  margin-bottom: 12px;
  font-size: 0.9em;
}
</style>
```

**类型定义 (types/settings.ts)**：

```typescript
export interface TranslateSettings {
  // 现有设置
  ocrEngine: string
  detector: string
  targetLanguage: string
  // ...
  
  // ✨ 新增：是否默认使用自动颜色
  useAutoFgColorByDefault: boolean  // 默认使用自动文字色
  useAutoBgColorByDefault: boolean  // 默认使用自动背景色
  
  // 全局默认颜色（用于未勾选"默认使用"时）
  defaultTextColor: string          // '#000000'
  defaultFillColor: string          // '#FFFFFF'
}
```

### 2. 气泡编辑器（快捷切换）

**BubbleEditor.vue** - 提供三种快捷选择：

```vue
<template>
  <div class="bubble-editor">
    <!-- 文字颜色 -->
    <div class="setting-item">
      <label>文字颜色:</label>
      
      <!-- 颜色选择器 -->
      <input 
        type="color" 
        v-model="bubble.textColor"
        @change="handleTextColorChange"
      />
      
      <!-- ✨ 快捷切换按钮组 -->
      <div class="color-quick-actions">
        <button 
          @click="useAutoTextColor"
          :class="{ active: isUsingAutoTextColor }"
          :disabled="!bubble?.autoFgColor"
          class="btn-quick"
        >
          💡 自动
        </button>
        
        <button 
          @click="useDefaultTextColor"
          :class="{ active: isUsingDefaultTextColor }"
          class="btn-quick"
        >
          🎨 默认
        </button>
        
        <span v-if="!isUsingAutoTextColor && !isUsingDefaultTextColor" class="badge-custom">
          ✏️ 自定义
        </span>
      </div>
      
      <!-- 显示详细信息 -->
      <div class="color-info">
        <div v-if="isUsingAutoTextColor">
          ✓ 使用自动识别颜色 RGB({{ bubble.autoFgColor.join(', ') }})
          <span class="confidence">置信度 {{ (bubble.colorConfidence * 100).toFixed(0) }}%</span>
        </div>
        <div v-else-if="isUsingDefaultTextColor">
          ✓ 使用全局默认颜色
        </div>
        <div v-else>
          ✓ 使用自定义颜色 {{ bubble.textColor }}
        </div>
      </div>
    </div>
    
    <!-- 填充颜色（纯色填充时显示） -->
    <div class="setting-item" v-if="bubble.inpaintMethod === 'solid'">
      <label>填充颜色:</label>
      
      <input 
        type="color" 
        v-model="bubble.fillColor"
        @change="handleFillColorChange"
      />
      
      <div class="color-quick-actions">
        <button 
          @click="useAutoFillColor"
          :class="{ active: isUsingAutoFillColor }"
          :disabled="!bubble?.autoBgColor"
          class="btn-quick"
        >
          💡 自动
        </button>
        
        <button 
          @click="useDefaultFillColor"
          :class="{ active: isUsingDefaultFillColor }"
          class="btn-quick"
        >
          🎨 默认
        </button>
        
        <span v-if="!isUsingAutoFillColor && !isUsingDefaultFillColor" class="badge-custom">
          ✏️ 自定义
        </span>
      </div>
      
      <div class="color-info">
        <div v-if="isUsingAutoFillColor">
          ✓ 使用自动识别颜色 RGB({{ bubble.autoBgColor.join(', ') }})
          <span class="confidence">置信度 {{ (bubble.colorConfidence * 100).toFixed(0) }}%</span>
        </div>
        <div v-else-if="isUsingDefaultFillColor">
          ✓ 使用全局默认颜色
        </div>
        <div v-else>
          ✓ 使用自定义颜色 {{ bubble.fillColor }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { BubbleState } from '@/types/bubble'
import { rgbArrayToHex, isValidHex } from '@/utils/colorUtils'
import { useSettingsStore } from '@/stores/settingsStore'

const props = defineProps<{
  bubble: BubbleState | null
}>()

const emit = defineEmits<{
  (e: 'update', updates: Partial<BubbleState>): void
}>()

const settingsStore = useSettingsStore()

// ============ 文字颜色相关 ============

// 判断是否正在使用自动文字色
const isUsingAutoTextColor = computed(() => {
  if (!props.bubble || !props.bubble.autoFgColor) return false
  const autoHex = rgbArrayToHex(props.bubble.autoFgColor)
  return props.bubble.textColor.toLowerCase() === autoHex.toLowerCase()
})

// 判断是否正在使用默认文字色
const isUsingDefaultTextColor = computed(() => {
  if (!props.bubble) return false
  return props.bubble.textColor === settingsStore.settings.defaultTextColor
})

// 使用自动文字色
function useAutoTextColor() {
  if (props.bubble && props.bubble.autoFgColor) {
    emit('update', { 
      textColor: rgbArrayToHex(props.bubble.autoFgColor)
    })
  }
}

// 使用默认文字色
function useDefaultTextColor() {
  emit('update', { 
    textColor: settingsStore.settings.defaultTextColor
  })
}

// ============ 填充颜色相关 ============

const isUsingAutoFillColor = computed(() => {
  if (!props.bubble || !props.bubble.autoBgColor) return false
  const autoHex = rgbArrayToHex(props.bubble.autoBgColor)
  return props.bubble.fillColor.toLowerCase() === autoHex.toLowerCase()
})

const isUsingDefaultFillColor = computed(() => {
  if (!props.bubble) return false
  return props.bubble.fillColor === settingsStore.settings.defaultFillColor
})

function useAutoFillColor() {
  if (props.bubble && props.bubble.autoBgColor) {
    emit('update', { 
      fillColor: rgbArrayToHex(props.bubble.autoBgColor)
    })
  }
}

function useDefaultFillColor() {
  emit('update', { 
    fillColor: settingsStore.settings.defaultFillColor
  })
}

// ============ 颜色change处理 ============

function handleTextColorChange(event: Event) {
  const newColor = (event.target as HTMLInputElement).value
  if (isValidHex(newColor)) {
    emit('update', { textColor: newColor })
  }
}

function handleFillColorChange(event: Event) {
  const newColor = (event.target as HTMLInputElement).value
  if (isValidHex(newColor)) {
    emit('update', { fillColor: newColor })
  }
}
</script>

<style scoped>
.color-quick-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  align-items: center;
}

.btn-quick {
  padding: 4px 12px;
  background: #f3f4f6;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9em;
  transition: all 0.2s;
}

.btn-quick:hover {
  background: #e5e7eb;
}

.btn-quick.active {
  background: #3b82f6;
  color: white;
  border-color: #3b82f6;
}

.badge-custom {
  padding: 4px 8px;
  background: #f59e0b;
  color: white;
  border-radius: 4px;
  font-size: 0.85em;
  font-weight: 500;
}

.color-info {
  margin-top: 8px;
  padding: 6px 10px;
  background: #f9fafb;
  border-left: 3px solid #10b981;
  border-radius: 4px;
  font-size: 0.9em;
  color: #374151;
}

.confidence {
  color: #6b7280;
  font-size: 0.85em;
  margin-left: 8px;
}
</style>
```

```vue
<template>
  <div class="bubble-editor">
    <!-- 文字颜色 -->
    <div class="setting-item">
      <label>文字颜色:</label>
      <div class="color-control">
        <input 
          type="color" 
          v-model="bubble.textColor"
          @change="handleTextColorChange"
        />
        
        <!-- ✨ 显示自动颜色来源信息（智能显示） -->
        <div v-if="bubble.autoFgColor" class="auto-color-info">
          <span class="badge">💡 自动识别</span>
          <span class="color-value">{{ formatRgb(bubble.autoFgColor) }}</span>
          <span class="confidence">
            置信度 {{ (bubble.colorConfidence * 100).toFixed(0) }}%
          </span>
          
          <!-- 只在"当前颜色 ≠ 自动颜色"时显示恢复按钮 -->
          <button 
            v-if="!isUsingAutoFgColor"
            @click="resetToAutoFgColor" 
            class="btn-reset"
            title="恢复到自动识别的颜色"
          >
            ↺ 恢复
          </button>
          <span v-else class="badge-active">✓ 使用中</span>
        </div>
      </div>
    </div>
    
    <!-- 填充颜色 -->
    <div class="setting-item" v-if="bubble.inpaintMethod === 'solid'">
      <label>填充颜色:</label>
      <div class="color-control">
        <input 
          type="color" 
          v-model="bubble.fillColor"
          @change="handleFillColorChange"
        />
        
        <div v-if="bubble.autoBgColor" class="auto-color-info">
          <span class="badge">💡 自动识别</span>
          <span class="color-value">{{ formatRgb(bubble.autoBgColor) }}</span>
          <span class="confidence">
            置信度 {{ (bubble.colorConfidence * 100).toFixed(0) }}%
          </span>
          
          <button 
            v-if="!isUsingAutoBgColor"
            @click="resetToAutoBgColor" 
            class="btn-reset"
          >
            ↺ 恢复
          </button>
          <span v-else class="badge-active">✓ 使用中</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { BubbleState } from '@/types/bubble'
import { rgbArrayToHex, isValidHex } from '@/utils/colorUtils'

const props = defineProps<{
  bubble: BubbleState | null
}>()

const emit = defineEmits<{
  (e: 'update', updates: Partial<BubbleState>): void
}>()

// 计算：是否正在使用自动前景色
const isUsingAutoFgColor = computed(() => {
  if (!props.bubble?.autoFgColor) return false
  const autoHex = rgbArrayToHex(props.bubble.autoFgColor)
  return props.bubble.textColor.toLowerCase() === autoHex.toLowerCase()
})

// 计算：是否正在使用自动背景色
const isUsingAutoBgColor = computed(() => {
  if (!props.bubble?.autoBgColor) return false
  const autoHex = rgbArrayToHex(props.bubble.autoBgColor)
  return props.bubble.fillColor.toLowerCase() === autoHex.toLowerCase()
})

// 格式化 RGB 显示
function formatRgb(rgb: [number, number, number]): string {
  return `RGB(${rgb.join(', ')})`
}

// 恢复到自动文字颜色
function resetToAutoFgColor() {
  if (props.bubble?.autoFgColor) {
    emit('update', { 
      textColor: rgbArrayToHex(props.bubble.autoFgColor) 
    })
  }
}

// 恢复到自动背景颜色
function resetToAutoBgColor() {
  if (props.bubble?.autoBgColor) {
    emit('update', { 
      fillColor: rgbArrayToHex(props.bubble.autoBgColor) 
    })
  }
}

// 处理文字颜色变化（验证格式）
function handleTextColorChange(event: Event) {
  const newColor = (event.target as HTMLInputElement).value
  if (isValidHex(newColor)) {
    emit('update', { textColor: newColor })
  }
}

// 处理填充颜色变化
function handleFillColorChange(event: Event) {
  const newColor = (event.target as HTMLInputElement).value
  if (isValidHex(newColor)) {
    emit('update', { fillColor: newColor })
  }
}
</script>

<style scoped>
.color-control {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.auto-color-info {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  background: #f0f9ff;
  border: 1px solid #bfdbfe;
  border-radius: 4px;
  font-size: 0.9em;
}

.badge {
  padding: 2px 6px;
  background: #3b82f6;
  color: white;
  border-radius: 3px;
  font-size: 0.85em;
  font-weight: 500;
}

.badge-active {
  padding: 2px 6px;
  background: #10b981;
  color: white;
  border-radius: 3px;
  font-size: 0.85em;
  font-weight: 500;
}

.color-value {
  font-family: monospace;
  font-size: 0.9em;
  color: #374151;
}

.confidence {
  color: #6b7280;
  font-size: 0.85em;
}

.btn-reset {
  margin-left: auto;
  padding: 3px 10px;
  background: #e5e7eb;
  border: 1px solid #d1d5db;
  border-radius: 3px;
  cursor: pointer;
  font-size: 0.9em;
  transition: all 0.2s;
}

.btn-reset:hover {
  background: #d1d5db;
  border-color: #9ca3af;
}

.btn-reset:active {
  transform: scale(0.95);
}
</style>
```

### 3. 类型定义

**types/bubble.ts**：

```typescript
export interface BubbleState {
  // 现有字段
  x: number
  y: number
  width: number
  height: number
  originalText: string
  translatedText: string
  coords: BubbleCoords
  polygon: PolygonCoords
  fontSize: number
  fontFamily: string
  textDirection: TextDirection
  autoTextDirection: TextDirection
  rotationAngle: number
  position: BubblePosition
  strokeEnabled: boolean
  strokeColor: string
  strokeWidth: number
  inpaintMethod: InpaintMethod
  
  // ✨ 颜色字段（新增）
  textColor: string                             // 文字颜色（始终是有效值）
  fillColor: string                             // 填充颜色（始终是有效值）
  autoFgColor?: [number, number, number] | null // 元数据：自动前景色
  autoBgColor?: [number, number, number] | null // 元数据：自动背景色
  colorConfidence?: number                      // 颜色置信度 0-1
  colorExtractedAt?: string                     // 提取时间戳
}
```

### 4. 渲染实现

**useEditRender.ts**：

```typescript
// 构建渲染参数（无需修改！）
const bubbleStatesForApi = bubbleStates.map((s) => ({
  translatedText: s.translatedText || '',
  coords: s.coords,
  fontSize: Number(s.fontSize) || 24,
  fontFamily: s.fontFamily || 'fonts/STSONG.TTF',
  textDirection: getEffectiveDirection(s),
  textColor: s.textColor || '#231816',        // ✅ 直接用！
  fillColor: s.fillColor || '#FFFFFF',        // ✅ 直接用！
  rotationAngle: Math.round(Number(s.rotationAngle) || 0),
  position: s.position || { x: 0, y: 0 },
  strokeEnabled: s.strokeEnabled !== undefined ? s.strokeEnabled : true,
  strokeColor: s.strokeColor || '#FFFFFF',
  strokeWidth: Number(s.strokeWidth) || 3,
}))

// 调用后端渲染
await reRenderImage({
  clean_image: cleanBase64,
  bubble_states: bubbleStatesForApi,
  // ...
})
```

**关键**：渲染逻辑**完全不需要修改**，因为 `textColor` 始终是有效的颜色值！

---

## 📐 核心算法

### 颜色预测（48px 模型）

```python
# 模型输出
(pred_chars, prob, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred) = model.infer(...)

# fg_pred: [seq_len, 3] 每个字符的前景色 RGB (0-1)
# bg_pred: [seq_len, 3] 每个字符的背景色 RGB (0-1)
# fg_ind_pred: [seq_len, 2] [无前景概率, 有前景概率]
# bg_ind_pred: [seq_len, 2] [无背景概率, 有背景概率]
```

### 颜色聚合

```python
def aggregate_colors(predictions):
    """对所有字符的颜色取平均"""
    fg_sum = [0, 0, 0]
    bg_sum = [0, 0, 0]
    fg_count = 0
    bg_count = 0
    
    for char_fg, char_bg, has_fg, has_bg in predictions:
        # 跳过特殊字符（<S>, </S>）
        
        if has_fg:  # fg_ind_pred[:, 1] > fg_ind_pred[:, 0]
            fg_sum += char_fg * 255
            fg_count += 1
        
        if has_bg:
            bg_sum += char_bg * 255
            bg_count += 1
        else:
            # 无背景时用前景色
            bg_sum += char_fg * 255
            bg_count += 1
    
    final_fg = [int(c / fg_count) for c in fg_sum] if fg_count > 0 else None
    final_bg = [int(c / bg_count) for c in bg_sum] if bg_count > 0 else None
    
    return final_fg, final_bg
```

### 对比度调整

```python
def adjust_colors(fg, bg):
    """确保前景和背景有足够对比度"""
    # 使用 CIE76 色差公式（LAB 色彩空间）
    diff = color_difference_lab(fg, bg)
    
    if diff < 30:  # 对比度不足
        fg_brightness = sum(fg) / 3
        if fg_brightness <= 127:
            bg = [255, 255, 255]  # 深色文字 → 白色背景
        else:
            bg = [0, 0, 0]        # 浅色文字 → 黑色背景
    
    return fg, bg

def color_difference_lab(rgb1, rgb2):
    """CIE76 色差"""
    lab1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2LAB)
    lab2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2LAB)
    diff = lab1 - lab2
    diff[..., 0] *= 0.392  # L 通道权重
    return np.linalg.norm(diff)
```

---

## ⚙️ 实现步骤

### Phase 1: 后端核心功能（2天）

1. ✅ 创建 `src/core/color_extractor.py`
2. ✅ 实现 `ColorExtractor` 类（复用 48px OCR 模型）
3. ✅ 实现颜色聚合和对比度调整算法
4. ✅ 单元测试

### Phase 2: 后端集成（1天）

1. ✅ 修改 `src/core/processing.py` 添加颜色提取调用
2. ✅ 更新 API 返回结构（bubbles 中添加 autoFgColor/autoBgColor）
3. ✅ 集成测试

### Phase 3: 前端类型和工具（0.5天）

1. ✅ 更新 `types/bubble.ts` 添加颜色字段
2. ✅ 更新 `types/settings.ts` 添加颜色提取开关
3. ✅ 实现 `rgbArrayToString()` 工具函数

### Phase 4: 前端UI（1天）

1. ✅ 修改 `TranslateSettings.vue` 添加颜色提取开关
2. ✅ 修改 `BubbleEditor.vue` 添加颜色来源信息和恢复按钮
3. ✅ 修改翻译响应处理逻辑，填充 textColor/fillColor

### Phase 5: 测试和文档（0.5天）

1. ✅ 端到端测试（各种 OCR 引擎组合）
2. ✅ 更新用户文档
3. ✅ 代码审查

**总计**：约 5 天

---

## 🧪 测试用例

### 场景测试

| 测试场景 | 预期结果 |
|---------|---------|
| **翻译时启用自动文字色** | bubble.textColor = 'rgb(...)' |
| **翻译时不启用** | bubble.textColor = 默认颜色, autoFgColor = null |
| **用户手动修改颜色** | textColor 改变，autoFgColor 保持 |
| **点击"恢复"按钮** | textColor 恢复为 rgb(autoFgColor) |
| **切图保存/加载** | 所有颜色字段正确保存和恢复 |
| **渲染** | 使用 textColor 渲染，无判断逻辑 |

### OCR 引擎组合测试

1. MangaOCR + 自动颜色
2. PaddleOCR + 自动颜色
3. 48px OCR + 自动颜色（模型复用）
4. 百度 OCR + 自动颜色
5. YOLOv5 检测器（降级处理）

---

## ⚠️ 兼容性

### 检测器兼容性

| 检测器 | 是否支持 | 说明 |
|--------|---------|------|
| CTD | ✅ | 输出原始文本行 |
| Default | ✅ | 输出原始文本行 |
| YOLO | ✅ | 输出原始文本行 |
| YOLOv5 | ⚠️ 降级 | 仅输出合并框，使用简单裁剪 |

---

## ⚠️ 注意事项和最佳实践

### 1. 颜色格式统一

**规则**：前后端统一使用 **Hex 格式**

```typescript
// ✅ 正确
bubble.textColor = '#0f1419'

// ❌ 错误（会导致 color input 无法显示）
bubble.textColor = 'rgb(15, 20, 25)'
```

### 2. 向后兼容性（重要！）⭐

**问题**：旧数据没有 `autoFgColor` 字段

**解决方案**：类型定义使用可选字段

```typescript
// ✅ 正确定义
interface BubbleState {
  textColor: string                              // 必需
  autoFgColor?: [number, number, number]        // ← 可选！向后兼容
  fillColor: string                              // 必需
  autoBgColor?: [number, number, number]         // ← 可选！向后兼容
  colorConfidence?: number                       // 可选
  colorExtractedAt?: string                      // 可选
}

// ✅ 使用时安全检查
if (bubble.autoFgColor) {
  const autoHex = rgbArrayToHex(bubble.autoFgColor)
  console.log(`自动颜色: ${autoHex}`)
}
```

### 3. 颜色提取失败的容错处理

**场景**：48px 模型加载失败或推理失败

**后端兜底策略**：

```python
# src/core/processing.py

try:
    colors = extractor.extract_colors(...)
    
    for i, color_info in enumerate(colors):
        if i < len(bubble_results):
            # ✅ 提供安全默认值
            bubble_results[i]['autoFgColor'] = color_info.get('fg_color') or [0, 0, 0]
            bubble_results[i]['autoBgColor'] = color_info.get('bg_color') or [255, 255, 255]
            bubble_results[i]['colorConfidence'] = color_info.get('confidence', 0.0)
            
except Exception as e:
    logger.error(f"颜色提取失败: {e}")
    # ✅ 失败时使用默认值
    for bubble in bubble_results:
        bubble['autoFgColor'] = [0, 0, 0]      # 黑色
        bubble['autoBgColor'] = [255, 255, 255]  # 白色
        bubble['colorConfidence'] = 0.0
```

**前端处理**：

```typescript
// 判断颜色是否为失败的默认值
function isColorExtractionFailed(rgb: [number, number, number]): boolean {
  // 全黑或全白可能是失败的标志（置信度为0时）
  const isBlack = rgb[0] === 0 && rgb[1] === 0 && rgb[2] === 0
  const isWhite = rgb[0] === 255 && rgb[1] === 255 && rgb[2] === 255
  return (isBlack || isWhite) && (bubble.colorConfidence === 0)
}

// ✅ UI 显示
<button 
  @click="useAutoTextColor"
  :disabled="!bubble.autoFgColor || isColorExtractionFailed(bubble.autoFgColor)"
>
  💡 自动
  <span v-if="isColorExtractionFailed(bubble.autoFgColor)">(提取失败)</span>
</button>
```

### 4. 手动创建气泡的处理

**场景**：用户在编辑模式手动画了一个新气泡

**问题**：没有图片无法提取颜色

**解决方案**：

```typescript
function createManualBubble(coords) {
  const settings = useSettingsStore()
  
  return {
    coords,
    textColor: settings.defaultTextColor || '#000000',  // ← 使用全局默认
    fillColor: settings.defaultFillColor || '#FFFFFF',
    autoFgColor: null,  // ← 手动创建没有自动颜色
    autoBgColor: null,
    colorConfidence: 0
  }
}

// UI 处理
<button 
  @click="useAutoTextColor"
  :disabled="!bubble.autoFgColor"  // ← 禁用按钮
>
  💡 自动
</button>

<span v-if="!bubble.autoFgColor" class="hint">
  手动创建的气泡无自动颜色
</span>
```

### 5. 恢复按钮显示逻辑

**最佳实践**：只在"当前颜色 ≠ 自动颜色"时显示恢复按钮

```vue
<button 
  v-if="bubble.autoFgColor && !isUsingAutoFgColor"
  @click="resetToAutoFgColor"
>
  ↺ 恢复
</button>
<span v-else-if="isUsingAutoFgColor" class="badge-active">
  ✓ 使用中
</span>
```

这样用户能清楚地知道：
- 有"恢复"按钮 → 颜色被改过
- 显示"✓ 使用中" → 正在使用自动颜色

### 6. 颜色值验证

**安全实践**：用户修改颜色时验证格式

```typescript
function handleTextColorChange(event: Event) {
  const newColor = (event.target as HTMLInputElement).value
  if (isValidHex(newColor)) {
    emit('update', { textColor: newColor })
  } else {
    console.warn('无效的颜色格式:', newColor)
    // 可选：弹出提示或恢复原值
  }
}
```

### 7. Settings 持久化

**✅ 确认**：Settings Store 已有 localStorage 持久化机制

```typescript
// vue-frontend/src/stores/settings/index.ts
const settingsStore = useSettingsStore()

// 设置会自动持久化到 localStorage
settingsStore.updateSettings({
  useAutoFgColorByDefault: true,
  useAutoBgColorByDefault: true
})
```

**刷新页面后会自动加载**，无需额外处理。

### 8. 性能优化

**避免频繁转换**：缓存 Hex 值

```typescript
// ❌ 每次渲染都转换（性能差且不安全）
computed(() => {
  if (!bubble.autoFgColor) return null
  return rgbArrayToHex(bubble.autoFgColor)
})

// ✅ 翻译时转换一次，存储 Hex（安全且高效）
bubble.textColor = apiData.autoFgColor 
  ? rgbArrayToHex(apiData.autoFgColor)
  : (settings.defaultTextColor || '#000000')
```

---

## 📊 优势总结

| 优势 | 说明 |
|------|------|
| ✅ **极简渲染** | `textColor` 始终有效，直接用 |
| ✅ **零判断逻辑** | 无需 `getEffectiveColor()` 等函数 |
| ✅ **数据完整** | autoFgColor **总是存在**，永不为 null |
| ✅ **灵活切换** | 用户可随时在自动/默认/自定义间切换 |
| ✅ **无默认值问题** | 强制提取消除了"没颜色时用什么"的困扰 |
| ✅ **直观UI** | 快捷按钮清晰展示当前使用的颜色来源 |
| ✅ **性能友好** | 颜色提取与OCR并行，成本低 |
| ✅ **易于维护** | 逻辑简单，不易出错 |

---

**文档版本**: v4.0（强制提取 + 灵活使用）  
**最后更新**: 2026-01-03 18:11  
**作者**: Saber-Translator Team

**变更记录**：
- v4.0: 采用强制提取方案，autoFgColor 永不为 null，简化默认值处理
- v3.0: 补充颜色格式转换、默认值处理、UI 优化等细节
- v2.0: 采用简化方案（textColor 始终存储有效值）
- v1.0: 初始设计（复刻 textDirection 模式，已废弃）

