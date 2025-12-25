<!--
  气泡编辑器组件
  编辑单个气泡的文本、字体、颜色等属性
  使用原版Office风格浅色主题
  - 支持原文和译文编辑
  - 支持日语软键盘输入
  - 支持单气泡重新OCR识别和翻译
  - 支持样式设置（字号、字体、颜色、描边等）
  - 支持修复方式选择
-->
<template>
  <div class="edit-panel-content">
    <!-- 【复刻原版】始终显示编辑面板，不显示"请选择气泡"提示 -->
    <!-- 原文编辑区 -->
    <div class="text-column original-text-column text-block">
      <div class="text-column-header">
        <span class="column-title">🇯🇵 日语原文 (OCR结果)</span>
        <button class="re-ocr-btn" @click="handleOcrRecognize" title="重新OCR此气泡">🔄</button>
      </div>
      <textarea
        ref="originalTextInput"
        v-model="localOriginalText"
        class="text-editor original-editor"
        placeholder="OCR识别的日语原文..."
        spellcheck="false"
        @input="handleOriginalTextChange"
      ></textarea>
      <div class="text-actions">
        <button class="copy-btn" @click="copyOriginalText">📋 复制</button>
        <button class="keyboard-toggle-btn" @click="toggleJpKeyboard" title="显示/隐藏50音键盘">
          ⌨️ 50音
        </button>
      </div>

      <!-- 50音软键盘 -->
      <JapaneseKeyboard
        :visible="showJpKeyboard"
        :default-target="jpKeyboardTarget"
        @close="showJpKeyboard = false"
        @insert="handleKanaInsert"
        @delete="handleKanaDelete"
      />
    </div>

    <!-- 译文编辑区 -->
    <div class="text-column translated-text-column text-block">
      <div class="text-column-header">
        <span class="column-title">🇨🇳 中文译文</span>
        <button class="re-translate-btn" @click="handleReTranslate" title="重新翻译此气泡">
          🔄
        </button>
      </div>
      <textarea
        ref="translatedTextInput"
        v-model="localTranslatedText"
        class="text-editor translated-editor"
        placeholder="翻译后的中文..."
        spellcheck="false"
        @input="handleTextChange"
      ></textarea>
      <div class="text-actions">
        <button class="copy-btn" @click="copyTranslatedText">📋 复制</button>
        <button class="apply-text-btn" @click="handleApplyBubble">✓ 应用文本</button>
      </div>
    </div>

    <!-- 样式设置区 -->
    <div class="style-settings-section text-block">
      <!-- Office风格文字设置工具栏 -->
      <div class="office-toolbar">
        <!-- 第一行：字体 + 字号 -->
        <div class="toolbar-row toolbar-row-top">
          <div class="combo-control font-control">
            <label>字体</label>
            <CustomSelect
              v-model="localFontFamily"
              :groups="fontSelectGroups"
              title="字体"
              @change="handleFontFamilyChange"
            />
          </div>
          <div class="combo-control size-control">
            <label>字号</label>
            <div class="size-input-wrap">
              <input
                type="number"
                v-model.number="localFontSize"
                class="toolbar-fontsize-input"
                :min="FONT_SIZE_MIN"
                :max="FONT_SIZE_MAX"
                :step="FONT_SIZE_STEP"
                title="字号"
                @change="handleFontSizeChange"
              />
              <div class="toolbar-fontsize-btns">
                <button class="toolbar-fontsize-btn" @click="increaseFontSize" title="增大字号">
                  A+
                </button>
                <button class="toolbar-fontsize-btn" @click="decreaseFontSize" title="减小字号">
                  A-
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- 第二行：样式工具按钮 -->
        <div class="toolbar-row toolbar-row-actions">
          <!-- 排版方向 -->
          <div class="toolbar-icon-group" aria-label="排版方向">
            <button
              class="toolbar-btn"
              :data-active="localTextDirection === 'vertical'"
              @click="setTextDirection('vertical')"
              title="竖向排版"
            >
              <svg viewBox="0 0 16 16" width="16" height="16">
                <path
                  d="M8 2v12M8 2L5 5M8 2l3 3"
                  stroke="currentColor"
                  stroke-width="1.5"
                  fill="none"
                />
              </svg>
            </button>
            <button
              class="toolbar-btn"
              :data-active="localTextDirection === 'horizontal'"
              @click="setTextDirection('horizontal')"
              title="横向排版"
            >
              <svg viewBox="0 0 16 16" width="16" height="16">
                <path
                  d="M2 8h12M14 8l-3-3M14 8l-3 3"
                  stroke="currentColor"
                  stroke-width="1.5"
                  fill="none"
                />
              </svg>
            </button>
          </div>

          <div class="toolbar-divider vertical"></div>

          <!-- 文字颜色 -->
          <div class="toolbar-color-group">
            <div class="toolbar-color-picker" title="文字颜色">
              <button class="toolbar-btn toolbar-color-btn" @click="triggerTextColorPicker">
                <svg viewBox="0 0 16 16" width="16" height="16">
                  <text x="3" y="11" font-size="10" font-weight="bold" fill="currentColor">A</text>
                </svg>
                <span class="color-indicator" :style="{ background: localTextColor }"></span>
              </button>
              <input
                ref="textColorInput"
                type="color"
                v-model="localTextColor"
                class="hidden-color-input"
                @change="handleTextColorChange"
              />
            </div>
          </div>

          <div class="toolbar-divider vertical"></div>

          <!-- 背景修复方式选择器 -->
          <div class="toolbar-inpaint-group" title="背景修复方式">
            <CustomSelect
              v-model="localInpaintMethod"
              :options="inpaintMethodOptions"
              @change="handleInpaintMethodChange"
            />

            <!-- 纯色填充时的颜色选择器 -->
            <div
              class="toolbar-color-picker toolbar-solid-color-options"
              :class="{ hidden: localInpaintMethod !== 'solid' }"
            >
              <button class="toolbar-btn toolbar-color-btn" @click="triggerFillColorPicker">
                <svg viewBox="0 0 16 16" width="16" height="16">
                  <rect
                    x="2"
                    y="2"
                    width="12"
                    height="12"
                    rx="2"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.2"
                  />
                  <rect x="4" y="4" width="8" height="8" rx="1" fill="currentColor" opacity="0.3" />
                </svg>
                <span class="color-indicator" :style="{ background: localFillColor }"></span>
              </button>
              <input
                ref="fillColorInput"
                type="color"
                v-model="localFillColor"
                class="hidden-color-input"
                @change="handleFillColorChange"
              />
            </div>
          </div>

          <div class="toolbar-divider vertical"></div>

          <!-- 描边设置 -->
          <div class="toolbar-stroke-cluster">
            <button
              class="toolbar-btn"
              :data-active="localStrokeEnabled"
              @click="toggleStroke"
              title="文字描边"
            >
              <svg viewBox="0 0 16 16" width="16" height="16">
                <text
                  x="3"
                  y="12"
                  font-size="11"
                  font-weight="bold"
                  stroke="currentColor"
                  stroke-width="2"
                  fill="none"
                >
                  A
                </text>
                <text x="3" y="12" font-size="11" font-weight="bold" fill="currentColor">A</text>
              </svg>
            </button>

            <div
              class="toolbar-color-picker toolbar-stroke-options"
              :class="{ hidden: !localStrokeEnabled }"
              title="描边颜色"
            >
              <button class="toolbar-btn toolbar-color-btn" @click="triggerStrokeColorPicker">
                <svg viewBox="0 0 16 16" width="16" height="16">
                  <circle cx="8" cy="8" r="5" fill="none" stroke="currentColor" stroke-width="2" />
                </svg>
                <span class="color-indicator" :style="{ background: localStrokeColor }"></span>
              </button>
              <input
                ref="strokeColorInput"
                type="color"
                v-model="localStrokeColor"
                class="hidden-color-input"
                @change="handleStrokeColorChange"
              />
            </div>

            <div
              class="toolbar-stroke-width toolbar-stroke-options"
              :class="{ hidden: !localStrokeEnabled }"
              title="描边宽度"
            >
              <input
                type="number"
                v-model.number="localStrokeWidth"
                class="toolbar-mini-input"
                min="0"
                max="10"
                @change="handleStrokeWidthChange"
              />
              <span class="toolbar-unit">px</span>
            </div>
          </div>
        </div>

        <!-- 第三行：旋转 + 位置 -->
        <div class="toolbar-row toolbar-row-bottom">
          <div class="toolbar-rotation-group" title="旋转角度">
            <button class="toolbar-btn" @click="rotateLeft" title="逆时针旋转">
              <svg viewBox="0 0 16 16" width="16" height="16">
                <path
                  d="M2 8a6 6 0 1 1 1.5 4"
                  stroke="currentColor"
                  stroke-width="1.5"
                  fill="none"
                />
                <path d="M2 5v3.5h3.5" stroke="currentColor" stroke-width="1.5" fill="none" />
              </svg>
            </button>
            <input
              type="number"
              v-model.number="localRotationAngle"
              class="toolbar-mini-input toolbar-rotation-input"
              min="-180"
              max="180"
              step="5"
              @change="handleRotationChange"
            />
            <span class="toolbar-unit">°</span>
            <button class="toolbar-btn" @click="rotateRight" title="顺时针旋转">
              <svg viewBox="0 0 16 16" width="16" height="16">
                <path
                  d="M14 8a6 6 0 1 0-1.5 4"
                  stroke="currentColor"
                  stroke-width="1.5"
                  fill="none"
                />
                <path d="M14 5v3.5h-3.5" stroke="currentColor" stroke-width="1.5" fill="none" />
              </svg>
            </button>
            <button class="toolbar-btn toolbar-small-btn" @click="resetRotation" title="重置旋转">
              0
            </button>
          </div>

          <div class="toolbar-divider vertical"></div>

          <div class="toolbar-position-group" title="位置调整">
            <button class="toolbar-btn" @click="moveLeft" title="左移">
              <svg viewBox="0 0 16 16" width="14" height="14">
                <path d="M10 3L5 8l5 5" stroke="currentColor" stroke-width="1.5" fill="none" />
              </svg>
            </button>
            <button class="toolbar-btn" @click="moveRight" title="右移">
              <svg viewBox="0 0 16 16" width="14" height="14">
                <path d="M6 3l5 5-5 5" stroke="currentColor" stroke-width="1.5" fill="none" />
              </svg>
            </button>
            <button class="toolbar-btn" @click="moveUp" title="上移">
              <svg viewBox="0 0 16 16" width="14" height="14">
                <path d="M3 10l5-5 5 5" stroke="currentColor" stroke-width="1.5" fill="none" />
              </svg>
            </button>
            <button class="toolbar-btn" @click="moveDown" title="下移">
              <svg viewBox="0 0 16 16" width="14" height="14">
                <path d="M3 6l5 5 5-5" stroke="currentColor" stroke-width="1.5" fill="none" />
              </svg>
            </button>
            <span class="toolbar-position-value">
              <span>{{ positionX }}</span
              >,<span>{{ positionY }}</span>
            </span>
            <button class="toolbar-btn toolbar-small-btn" @click="resetPosition" title="重置位置">
              ⌂
            </button>
          </div>
        </div>
      </div>

      <!-- 字号预设快捷按钮（可折叠） -->
      <details class="fontsize-presets-panel">
        <summary>字号预设</summary>
        <div class="font-size-presets">
          <button
            v-for="preset in FONT_SIZE_PRESETS"
            :key="preset"
            class="preset-btn"
            :class="{ active: localFontSize === preset }"
            @click="setFontSize(preset)"
          >
            {{ preset }}
          </button>
        </div>
      </details>

      <!-- 操作按钮 -->
      <div class="edit-action-buttons">
        <button class="btn-apply" @click="handleApplyBubble">应用</button>
        <button class="btn-apply-all" @click="applyToAll">应用全部</button>
        <button class="btn-reset" @click="resetBubbleEdit">重置</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 气泡编辑器组件
 * 编辑单个气泡的文本和样式属性
 * 使用原版Office风格浅色主题
 */
import { ref, watch, computed, onMounted, nextTick } from 'vue'
import { useBubbleStore } from '@/stores/bubbleStore'
import { FONT_SIZE_PRESETS, FONT_SIZE_MIN, FONT_SIZE_MAX, FONT_SIZE_STEP } from '@/constants'
import type { BubbleState, TextDirection, InpaintMethod } from '@/types/bubble'
import { getFontListApi } from '@/api/config'
import JapaneseKeyboard from './JapaneseKeyboard.vue'
import CustomSelect from '@/components/common/CustomSelect.vue'

// ============================================================
// Props 和 Emits
// ============================================================

const props = defineProps<{
  /** 气泡数据（可能为null） */
  bubble: BubbleState | null
  /** 气泡索引 */
  bubbleIndex: number
}>()

const emit = defineEmits<{
  /** 更新气泡属性 */
  (e: 'update', updates: Partial<BubbleState>): void
  /** 重新渲染 */
  (e: 'reRender'): void
  /** 重新OCR识别 */
  (e: 'ocrRecognize', index: number): void
  /** 重新翻译单个气泡 */
  (e: 'reTranslate', index: number): void
  /** 应用当前气泡更改 */
  (e: 'applyBubble', index: number): void
  /** 【复刻原版 4.3】重置当前气泡到初始状态 */
  (e: 'resetCurrent', index: number): void
}>()

// ============================================================
// Store
// ============================================================

const bubbleStore = useBubbleStore()

// ============================================================
// 默认值
// ============================================================

const defaultBubble: BubbleState = {
  coords: [0, 0, 0, 0],
  polygon: [],
  originalText: '',
  translatedText: '',
  textboxText: '',
  fontSize: 24,
  fontFamily: 'fonts/STSONG.TTF',
  textDirection: 'auto',
  autoTextDirection: 'vertical',
  textColor: '#231816',
  fillColor: '#FFFFFF',
  strokeEnabled: true,
  strokeColor: '#FFFFFF',
  strokeWidth: 3,
  rotationAngle: 0,
  inpaintMethod: 'solid',
  position: { x: 0, y: 0 },
}

// ============================================================
// 本地状态（用于双向绑定）
// ============================================================

const localOriginalText = ref('')
const localTranslatedText = ref('')
const localFontSize = ref(24)
const localFontFamily = ref('fonts/STSONG.TTF')
const localTextDirection = ref<TextDirection>('auto')
const localTextColor = ref('#231816')
const localFillColor = ref('#FFFFFF')
const localStrokeEnabled = ref(true)
const localStrokeColor = ref('#FFFFFF')
const localStrokeWidth = ref(3)
const localRotationAngle = ref(0)
const localInpaintMethod = ref<InpaintMethod>('solid')
const localPositionX = ref(0)
const localPositionY = ref(0)

// 文本输入框引用
const originalTextInput = ref<HTMLTextAreaElement | null>(null)
const translatedTextInput = ref<HTMLTextAreaElement | null>(null)

// 颜色选择器引用
const textColorInput = ref<HTMLInputElement | null>(null)
const fillColorInput = ref<HTMLInputElement | null>(null)
const strokeColorInput = ref<HTMLInputElement | null>(null)

// 日语软键盘状态
const showJpKeyboard = ref(false)
const jpKeyboardTarget = ref<'original' | 'translated'>('original')

// 字体相关
const systemFonts = ref<{ name: string; path: string }[]>([
  { name: '华文宋体', path: 'fonts/STSONG.TTF' },
  { name: '华文楷体', path: 'fonts/STKAITI.TTF' },
  { name: '华文细黑', path: 'fonts/STXIHEI.TTF' },
  { name: '黑体', path: 'fonts/SIMHEI.TTF' },
  { name: '宋体', path: 'fonts/SIMSUN.TTC' },
])
const customFonts = ref<{ name: string; path: string }[]>([])

// ============================================================
// 计算属性
// ============================================================

/** 位置X */
const positionX = computed(() => {
  if (!props.bubble) return 0
  return props.bubble.coords[0] + localPositionX.value
})

/** 位置Y */
const positionY = computed(() => {
  if (!props.bubble) return 0
  return props.bubble.coords[1] + localPositionY.value
})

/** 字体选择器分组选项（用于CustomSelect） */
const fontSelectGroups = computed(() => {
  const groups = [
    {
      label: '系统字体',
      options: systemFonts.value.map(f => ({ label: f.name, value: f.path })),
    },
  ]
  if (customFonts.value.length > 0) {
    groups.push({
      label: '自定义字体',
      options: customFonts.value.map(f => ({ label: f.name, value: f.path })),
    })
  }
  return groups
})

/** 背景修复方式选项（用于CustomSelect） */
const inpaintMethodOptions = [
  { label: '纯色填充', value: 'solid' },
  { label: 'LAMA修复(漫画)', value: 'lama_mpe' },
  { label: 'LAMA修复(通用)', value: 'litelama' },
]

// ============================================================
// 同步本地状态
// ============================================================

/** 从气泡数据同步到本地状态 */
function syncFromBubble(bubble: BubbleState | null): void {
  const b = bubble || defaultBubble
  localOriginalText.value = b.originalText
  localTranslatedText.value = b.translatedText
  localFontSize.value = b.fontSize
  localFontFamily.value = b.fontFamily
  localTextDirection.value = b.textDirection
  localTextColor.value = b.textColor
  localFillColor.value = b.fillColor
  localStrokeEnabled.value = b.strokeEnabled
  localStrokeColor.value = b.strokeColor
  localStrokeWidth.value = b.strokeWidth
  localRotationAngle.value = b.rotationAngle
  localInpaintMethod.value = b.inpaintMethod
  localPositionX.value = b.position?.x || 0
  localPositionY.value = b.position?.y || 0
}

// 监听 props 变化，同步本地状态
watch(
  () => props.bubble,
  newBubble => {
    syncFromBubble(newBubble)
  },
  { deep: true, immediate: true }
)

// ============================================================
// 事件处理 - 文本
// ============================================================

/** 处理原文变化 */
function handleOriginalTextChange(): void {
  emit('update', { originalText: localOriginalText.value })
}

/** 处理译文变化 */
function handleTextChange(): void {
  emit('update', { translatedText: localTranslatedText.value })
}

/** 复制原文 */
function copyOriginalText(): void {
  navigator.clipboard.writeText(localOriginalText.value)
}

/** 复制译文 */
function copyTranslatedText(): void {
  navigator.clipboard.writeText(localTranslatedText.value)
}

// ============================================================
// 事件处理 - 字体和字号
// ============================================================

/** 处理字号变化 */
function handleFontSizeChange(): void {
  emit('update', { fontSize: localFontSize.value })
}

/** 设置字号 */
function setFontSize(size: number): void {
  localFontSize.value = size
  emit('update', { fontSize: size })
}

/** 增大字号 */
function increaseFontSize(): void {
  localFontSize.value = Math.min(FONT_SIZE_MAX, localFontSize.value + FONT_SIZE_STEP)
  emit('update', { fontSize: localFontSize.value })
}

/** 减小字号 */
function decreaseFontSize(): void {
  localFontSize.value = Math.max(FONT_SIZE_MIN, localFontSize.value - FONT_SIZE_STEP)
  emit('update', { fontSize: localFontSize.value })
}

/** 处理字体变化 */
function handleFontFamilyChange(): void {
  emit('update', { fontFamily: localFontFamily.value })
}

// ============================================================
// 事件处理 - 排版方向
// ============================================================

/** 设置排版方向 */
function setTextDirection(direction: TextDirection): void {
  localTextDirection.value = direction
  emit('update', { textDirection: direction })
}

// ============================================================
// 事件处理 - 颜色
// ============================================================

/** 触发文字颜色选择器 */
function triggerTextColorPicker(): void {
  textColorInput.value?.click()
}

/** 处理文字颜色变化 */
function handleTextColorChange(): void {
  emit('update', { textColor: localTextColor.value })
}

/** 触发填充颜色选择器 */
function triggerFillColorPicker(): void {
  fillColorInput.value?.click()
}

/** 处理填充颜色变化 */
function handleFillColorChange(): void {
  emit('update', { fillColor: localFillColor.value })
}

/** 触发描边颜色选择器 */
function triggerStrokeColorPicker(): void {
  strokeColorInput.value?.click()
}

/** 处理描边颜色变化 */
function handleStrokeColorChange(): void {
  emit('update', { strokeColor: localStrokeColor.value })
}

// ============================================================
// 事件处理 - 描边
// ============================================================

/** 切换描边 */
function toggleStroke(): void {
  localStrokeEnabled.value = !localStrokeEnabled.value
  emit('update', { strokeEnabled: localStrokeEnabled.value })
}

/** 处理描边宽度变化 */
function handleStrokeWidthChange(): void {
  emit('update', { strokeWidth: localStrokeWidth.value })
}

// ============================================================
// 事件处理 - 修复方式
// ============================================================

/** 处理修复方式变化 */
function handleInpaintMethodChange(): void {
  emit('update', { inpaintMethod: localInpaintMethod.value })
}

// ============================================================
// 事件处理 - 旋转
// ============================================================

/** 处理旋转角度变化 */
function handleRotationChange(): void {
  emit('update', { rotationAngle: localRotationAngle.value })
}

/** 逆时针旋转 */
function rotateLeft(): void {
  localRotationAngle.value = Math.max(-180, localRotationAngle.value - 5)
  emit('update', { rotationAngle: localRotationAngle.value })
}

/** 顺时针旋转 */
function rotateRight(): void {
  localRotationAngle.value = Math.min(180, localRotationAngle.value + 5)
  emit('update', { rotationAngle: localRotationAngle.value })
}

/** 重置旋转 */
function resetRotation(): void {
  localRotationAngle.value = 0
  emit('update', { rotationAngle: 0 })
}

// ============================================================
// 事件处理 - 位置
// ============================================================

const MOVE_STEP = 2

/** 左移 */
function moveLeft(): void {
  localPositionX.value -= MOVE_STEP
  emit('update', { position: { x: localPositionX.value, y: localPositionY.value } })
}

/** 右移 */
function moveRight(): void {
  localPositionX.value += MOVE_STEP
  emit('update', { position: { x: localPositionX.value, y: localPositionY.value } })
}

/** 上移 */
function moveUp(): void {
  localPositionY.value -= MOVE_STEP
  emit('update', { position: { x: localPositionX.value, y: localPositionY.value } })
}

/** 下移 */
function moveDown(): void {
  localPositionY.value += MOVE_STEP
  emit('update', { position: { x: localPositionX.value, y: localPositionY.value } })
}

/** 重置位置 */
function resetPosition(): void {
  localPositionX.value = 0
  localPositionY.value = 0
  emit('update', { position: { x: 0, y: 0 } })
}

// ============================================================
// 事件处理 - 操作按钮
// ============================================================

/** 应用当前气泡更改 */
function handleApplyBubble(): void {
  emit('applyBubble', props.bubbleIndex)
}

/** 应用到全部气泡 */
function applyToAll(): void {
  bubbleStore.updateAllBubbles({
    fontSize: localFontSize.value,
    fontFamily: localFontFamily.value,
    textDirection: localTextDirection.value,
    textColor: localTextColor.value,
    fillColor: localFillColor.value,
    strokeEnabled: localStrokeEnabled.value,
    strokeColor: localStrokeColor.value,
    strokeWidth: localStrokeWidth.value,
    inpaintMethod: localInpaintMethod.value,
  })
  console.log('样式已应用到所有气泡')
  // 触发重新渲染
  emit('reRender')
}

/** 重置气泡编辑 */
function resetBubbleEdit(): void {
  // 【复刻原版 4.3】通知父组件重置当前气泡到初始状态
  // 旧版使用 state.initialBubbleStates 保存进入编辑模式时的快照
  emit('resetCurrent', props.bubbleIndex)
}

/** 重新OCR识别 */
function handleOcrRecognize(): void {
  emit('ocrRecognize', props.bubbleIndex)
}

/** 重新翻译单个气泡 */
function handleReTranslate(): void {
  emit('reTranslate', props.bubbleIndex)
}

// ============================================================
// 日语软键盘相关
// ============================================================

/** 切换日语软键盘显示 */
function toggleJpKeyboard(): void {
  showJpKeyboard.value = !showJpKeyboard.value
}

/** 处理假名插入 */
function handleKanaInsert(char: string, target: 'original' | 'translated'): void {
  if (target === 'original') {
    const input = originalTextInput.value
    if (input) {
      const start = input.selectionStart || localOriginalText.value.length
      const end = input.selectionEnd || localOriginalText.value.length
      const text = localOriginalText.value
      localOriginalText.value = text.slice(0, start) + char + text.slice(end)
      nextTick(() => {
        input.selectionStart = input.selectionEnd = start + char.length
        input.focus()
      })
      emit('update', { originalText: localOriginalText.value })
    }
  } else {
    const input = translatedTextInput.value
    if (input) {
      const start = input.selectionStart || localTranslatedText.value.length
      const end = input.selectionEnd || localTranslatedText.value.length
      const text = localTranslatedText.value
      localTranslatedText.value = text.slice(0, start) + char + text.slice(end)
      nextTick(() => {
        input.selectionStart = input.selectionEnd = start + char.length
        input.focus()
      })
      emit('update', { translatedText: localTranslatedText.value })
    }
  }
}

/** 处理假名删除 */
function handleKanaDelete(target: 'original' | 'translated'): void {
  if (target === 'original') {
    const input = originalTextInput.value
    if (input && localOriginalText.value.length > 0) {
      const start = input.selectionStart || localOriginalText.value.length
      const end = input.selectionEnd || localOriginalText.value.length
      const text = localOriginalText.value
      if (start === end && start > 0) {
        localOriginalText.value = text.slice(0, start - 1) + text.slice(end)
        nextTick(() => {
          input.selectionStart = input.selectionEnd = start - 1
          input.focus()
        })
      } else if (start !== end) {
        localOriginalText.value = text.slice(0, start) + text.slice(end)
        nextTick(() => {
          input.selectionStart = input.selectionEnd = start
          input.focus()
        })
      }
      emit('update', { originalText: localOriginalText.value })
    }
  } else {
    const input = translatedTextInput.value
    if (input && localTranslatedText.value.length > 0) {
      const start = input.selectionStart || localTranslatedText.value.length
      const end = input.selectionEnd || localTranslatedText.value.length
      const text = localTranslatedText.value
      if (start === end && start > 0) {
        localTranslatedText.value = text.slice(0, start - 1) + text.slice(end)
        nextTick(() => {
          input.selectionStart = input.selectionEnd = start - 1
          input.focus()
        })
      } else if (start !== end) {
        localTranslatedText.value = text.slice(0, start) + text.slice(end)
        nextTick(() => {
          input.selectionStart = input.selectionEnd = start
          input.focus()
        })
      }
      emit('update', { translatedText: localTranslatedText.value })
    }
  }
}

// ============================================================
// 字体管理
// ============================================================

/** 加载字体列表 */
async function loadFontList(): Promise<void> {
  try {
    const response = await getFontListApi()
    if (response.fonts) {
      const system: { name: string; path: string }[] = []
      const custom: { name: string; path: string }[] = []

      for (const font of response.fonts) {
        // API返回的字段是display_name，需要转换为name
        const fontItem = {
          name: typeof font === 'string' ? font : font.display_name || font.file_name || '',
          path: typeof font === 'string' ? font : font.path,
        }
        if (fontItem.path.startsWith('fonts/')) {
          system.push(fontItem)
        } else {
          custom.push(fontItem)
        }
      }

      if (system.length > 0) {
        systemFonts.value = system
      }
      customFonts.value = custom
    }
  } catch (error) {
    console.error('加载字体列表失败:', error)
  }
}

// ============================================================
// 生命周期
// ============================================================

onMounted(() => {
  loadFontList()
})
</script>

<style scoped>
/* ============ 编辑面板内容 - 使用原版浅色主题 ============ */

.edit-panel-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 15px;
  overflow: auto;
  min-height: 0;
  background: var(--card-bg-color, #fff);
}

/* 文本块 */
.text-block {
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: 100%;
}

/* 文本列头部 */
.text-column-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--border-color, #e9ecef);
}

.column-title {
  font-weight: 600;
  font-size: 14px;
  color: var(--text-color, #495057);
}

.original-text-column .column-title {
  color: #e74c3c;
}

.translated-text-column .column-title {
  color: #27ae60;
}

/* 重新OCR/翻译按钮 */
.re-ocr-btn,
.re-translate-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background: var(--bg-color, #f8f9fa);
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.re-ocr-btn:hover,
.re-translate-btn:hover {
  background: #3498db;
  color: #fff;
}

/* 文本编辑器 */
.text-editor {
  flex: 1;
  width: 100%;
  min-height: 60px;
  padding: 12px;
  border: 2px solid var(--border-color, #e9ecef);
  border-radius: 8px;
  font-size: 15px;
  line-height: 1.6;
  resize: none;
  transition:
    border-color 0.2s,
    box-shadow 0.2s;
  box-sizing: border-box;
  font-family: inherit;
}

.text-editor:focus {
  outline: none;
  border-color: #3498db;
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.15);
}

.original-editor {
  background: #fff8f8;
  font-family: 'Noto Sans JP', 'Yu Gothic', 'MS Gothic', sans-serif;
}

.translated-editor {
  background: #f8fff8;
}

/* 文本操作按钮 */
.text-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  justify-content: flex-end;
}

.text-actions button {
  padding: 6px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  background: var(--card-bg-color, white);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.15s;
}

.text-actions button:hover {
  background: var(--bg-color, #f8f9fa);
  border-color: #adb5bd;
}

.apply-text-btn {
  background: #27ae60 !important;
  color: white !important;
  border-color: #27ae60 !important;
}

.apply-text-btn:hover {
  background: #219a52 !important;
}

.keyboard-toggle-btn {
  background: var(--bg-color, #f8f9fa);
}

/* ============ 样式设置区 ============ */

.style-settings-section {
  width: 100%;
  padding: 16px;
  background: #f5f6fb;
  border-radius: 10px;
  border: 1px solid rgba(82, 92, 105, 0.12);
  overflow-y: auto;
  box-sizing: border-box;
}

/* ============ Office风格工具栏 ============ */

.office-toolbar {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px;
  background: #ffffff;
  border: 1px solid rgba(96, 110, 140, 0.22);
  border-radius: 12px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
}

.toolbar-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.toolbar-row-top .combo-control {
  flex: 1;
  min-width: 160px;
}

.toolbar-row-actions,
.toolbar-row-bottom {
  gap: 8px;
  padding: 8px 10px;
  border: 1px solid rgba(226, 232, 240, 0.9);
  border-radius: 10px;
  background: linear-gradient(180deg, #fbfcff 0%, #f4f6ff 100%);
}

.combo-control {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 11px;
  color: #57607c;
}

.combo-control label {
  font-weight: 600;
  letter-spacing: 0.2px;
}

.size-input-wrap {
  display: flex;
  align-items: center;
  gap: 6px;
}

.toolbar-divider {
  width: 1px;
  height: 26px;
  background: rgba(15, 23, 42, 0.08);
}

.toolbar-divider.vertical {
  height: 34px;
  margin: 0 2px;
}

.toolbar-icon-group,
.toolbar-color-group,
.toolbar-stroke-cluster {
  display: flex;
  align-items: center;
  gap: 6px;
}

/* 字体选择器 */
.toolbar-font-select {
  min-width: 160px;
  height: 36px;
  padding: 0 10px;
  border: 1px solid #cfd6e4;
  border-radius: 8px;
  font-size: 13px;
  background: #fff;
  color: #1f2430;
  cursor: pointer;
  transition:
    border-color 0.15s,
    box-shadow 0.15s;
}

.toolbar-font-select:hover {
  border-color: #8aa0f6;
}

.toolbar-font-select:focus {
  outline: none;
  border-color: #5b73f2;
  box-shadow: 0 0 0 2px rgba(88, 125, 255, 0.18);
}

/* 字号输入 */
.toolbar-fontsize-input {
  width: 60px;
  height: 36px;
  border: 1px solid #cfd6e4;
  border-radius: 8px;
  padding: 0 8px;
  font-size: 14px;
  text-align: center;
  background: #fff;
  color: #1f2430;
}

.toolbar-fontsize-input:focus {
  outline: none;
  border-color: #5b73f2;
  box-shadow: 0 0 0 2px rgba(88, 125, 255, 0.15);
}

.toolbar-fontsize-btns {
  display: flex;
  gap: 6px;
}

.toolbar-fontsize-btn {
  min-width: 50px;
  height: 34px;
  border: 1px solid #d0d7ea;
  border-radius: 8px;
  background: #f2f4ff;
  color: #2f46c8;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  transition: all 0.15s;
}

.toolbar-fontsize-btn:hover {
  background: #dfe4ff;
  border-color: #9aaefc;
  color: #1d34a8;
}

/* 工具栏按钮 */
.toolbar-btn {
  width: 34px;
  height: 34px;
  border: 1px solid rgba(119, 130, 161, 0.35);
  border-radius: 8px;
  background: #fff;
  color: #3b3f4f;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.12s;
  padding: 0;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.03);
}

.toolbar-btn:hover {
  border-color: #7d96ff;
  color: #2b4bff;
  box-shadow: 0 2px 8px rgba(107, 125, 255, 0.25);
}

.toolbar-btn[data-active='true'],
.toolbar-btn.active {
  background: linear-gradient(135deg, #e8edff, #d9e2ff);
  border-color: #5670ff;
  color: #3040c2;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
}

.toolbar-btn:active {
  transform: scale(0.95);
}

.toolbar-btn svg {
  pointer-events: none;
}

.toolbar-small-btn {
  width: 24px;
  height: 24px;
  font-size: 11px;
  font-weight: 600;
}

/* 颜色选择器 */
.toolbar-color-picker {
  position: relative;
  display: inline-flex;
}

.toolbar-color-btn {
  flex-direction: column;
  gap: 4px;
  height: 34px;
  padding: 4px;
}

.color-indicator {
  width: 26px;
  height: 6px;
  border-radius: 999px;
  border: 1px solid rgba(0, 0, 0, 0.2);
}

.hidden-color-input {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}

/* 描边选项 */
.toolbar-stroke-options {
  transition: opacity 0.2s;
}

.toolbar-stroke-options.hidden {
  opacity: 0.4;
  pointer-events: none;
}

/* 背景修复方式选择器 */
.toolbar-inpaint-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.toolbar-inpaint-select {
  height: 34px;
  padding: 0 10px;
  border: 1px solid #cfd6e4;
  border-radius: 8px;
  font-size: 12px;
  background: #fff;
  color: #1f2430;
  cursor: pointer;
  transition:
    border-color 0.15s,
    box-shadow 0.15s;
}

.toolbar-inpaint-select:hover {
  border-color: #8aa0f6;
}

.toolbar-inpaint-select:focus {
  outline: none;
  border-color: #5b73f2;
  box-shadow: 0 0 0 2px rgba(88, 125, 255, 0.18);
}

.toolbar-solid-color-options {
  transition:
    opacity 0.2s,
    visibility 0.2s;
}

.toolbar-solid-color-options.hidden {
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
}

.toolbar-stroke-width {
  display: flex;
  align-items: center;
  gap: 4px;
}

.toolbar-mini-input {
  width: 46px;
  height: 32px;
  border: 1px solid #cfd6e4;
  border-radius: 6px;
  padding: 0 6px;
  font-size: 12px;
  text-align: center;
  background: #fff;
  color: #1f2430;
}

.toolbar-mini-input:focus {
  outline: none;
  border-color: #5b73f2;
  box-shadow: 0 0 0 2px rgba(88, 125, 255, 0.2);
}

.toolbar-unit {
  font-size: 11px;
  color: #596071;
}

/* 旋转控制组 */
.toolbar-rotation-group {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.toolbar-rotation-input {
  width: 58px;
}

/* 位置控制组 */
.toolbar-position-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.toolbar-position-value {
  font-size: 12px;
  color: #4a4f63;
  min-width: 48px;
  text-align: center;
  padding: 0 6px;
  border-radius: 6px;
  background: #eef1ff;
}

/* 字号预设面板 */
.fontsize-presets-panel {
  margin-top: 12px;
  border-top: 1px solid var(--border-color, #e0e0e0);
  padding-top: 12px;
}

.fontsize-presets-panel summary {
  cursor: pointer;
  font-size: 13px;
  color: var(--text-color, #495057);
  font-weight: 500;
  padding: 4px 0;
}

.font-size-presets {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
}

.preset-btn {
  padding: 6px 12px;
  background: #f2f4ff;
  border: 1px solid #d0d7ea;
  border-radius: 6px;
  color: #2f46c8;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
}

.preset-btn:hover {
  background: #dfe4ff;
  border-color: #9aaefc;
}

.preset-btn.active {
  background: linear-gradient(135deg, #e8edff, #d9e2ff);
  border-color: #5670ff;
  color: #3040c2;
}

/* 操作按钮 */
.edit-action-buttons {
  display: flex;
  gap: 10px;
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid var(--border-color, #e0e0e0);
}

.btn-apply,
.btn-apply-all,
.btn-reset {
  flex: 1;
  padding: 10px 16px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-apply {
  background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
  border: none;
  color: white;
}

.btn-apply:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
}

.btn-apply-all {
  background: linear-gradient(135deg, #3498db 0%, #5dade2 100%);
  border: none;
  color: white;
}

.btn-apply-all:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
}

.btn-reset {
  background: var(--card-bg-color, #fff);
  border: 1px solid var(--border-color, #ddd);
  color: var(--text-color, #495057);
}

.btn-reset:hover {
  background: var(--bg-color, #f8f9fa);
  border-color: #adb5bd;
}
</style>
