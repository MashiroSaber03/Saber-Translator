<script setup lang="ts">

import UiInput from '@/components/ui/UiInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
/**
 * 阅读器控制组件
 * 包含页码指示器、阅读设置面板、章节导航、回到顶部按钮、键盘快捷键
 */
import { ref, onMounted, onUnmounted } from 'vue'

// 阅读设置接口
export interface ReaderSettings {
  /** 图片宽度百分比 (50-100) */
  imageWidth: number
  /** 图片间距像素 (0-50) */
  imageGap: number
  /** 背景颜色 */
  bgColor: string
}

interface StoredReaderSettings extends ReaderSettings {
  readerSettingsSchemaVersion: 1
}

const READER_SETTINGS_KEY = 'readerSettings'
const READER_SETTINGS_SCHEMA_VERSION = 1
const DEFAULT_READER_SETTINGS: ReaderSettings = {
  imageWidth: 100,
  imageGap: 8,
  bgColor: '#1a1a2e'
}

// 组件属性
const props = defineProps<{
  /** 当前页码 */
  currentPage: number
  /** 总页数 */
  totalPages: number
  /** 是否有上一章 */
  hasPrevChapter: boolean
  /** 是否有下一章 */
  hasNextChapter: boolean
  /** 是否显示章节导航 */
  showChapterNav: boolean
}>()

// 组件事件
const emit = defineEmits<{
  /** 导航到上一章/下一章 */
  (e: 'navigateChapter', direction: 'prev' | 'next'): void
  /** 设置变化 */
  (e: 'settingsChange', settings: ReaderSettings): void
}>()

// ==================== 状态定义 ====================

// 阅读设置
const settings = ref<ReaderSettings>({ ...DEFAULT_READER_SETTINGS })

// 设置面板显示状态
const isSettingsPanelOpen = ref(false)

// 回到顶部按钮显示状态
const showScrollTopBtn = ref(false)

// 背景颜色预设
const bgColorPresets = [
  { color: '#1a1a2e', name: '深蓝' },
  { color: '#ffffff', name: '白色' },
  { color: '#f5f5dc', name: '米色' },
  { color: '#2d2d2d', name: '深灰' }
]
const bgColorValues = new Set(bgColorPresets.map((preset) => preset.color))

// ==================== 方法 ====================

/**
 * 打开设置面板
 */
function openSettings() {
  isSettingsPanelOpen.value = true
}

/**
 * 关闭设置面板
 */
function closeSettings() {
  isSettingsPanelOpen.value = false
}

/**
 * 回到顶部
 */
function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

/**
 * 处理滚动事件
 */
function handleScroll() {
  const scrollTop = window.scrollY
  showScrollTopBtn.value = scrollTop > 500
}

/**
 * 处理键盘事件
 */
function handleKeydown(e: KeyboardEvent) {
  switch (e.key) {
    case 'Escape':
      closeSettings()
      break
    case 'ArrowLeft':
      if (props.hasPrevChapter) {
        emit('navigateChapter', 'prev')
      }
      break
    case 'ArrowRight':
      if (props.hasNextChapter) {
        emit('navigateChapter', 'next')
      }
      break
    case 'Home':
      window.scrollTo({ top: 0, behavior: 'smooth' })
      break
    case 'End':
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
      break
  }
}

function isNumberInRange(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max
}

function isStoredReaderSettings(value: unknown): value is StoredReaderSettings {
  if (!value || typeof value !== 'object') return false

  const candidate = value as Partial<StoredReaderSettings>
  return (
    candidate.readerSettingsSchemaVersion === READER_SETTINGS_SCHEMA_VERSION &&
    isNumberInRange(candidate.imageWidth, 50, 100) &&
    isNumberInRange(candidate.imageGap, 0, 50) &&
    typeof candidate.bgColor === 'string' &&
    bgColorValues.has(candidate.bgColor)
  )
}

/**
 * 加载设置
 */
function loadSettings() {
  const saved = localStorage.getItem(READER_SETTINGS_KEY)
  if (saved) {
    try {
      const parsed: unknown = JSON.parse(saved)
      if (isStoredReaderSettings(parsed)) {
        settings.value = {
          imageWidth: parsed.imageWidth,
          imageGap: parsed.imageGap,
          bgColor: parsed.bgColor
        }
      }
    } catch (e) {
      console.error('加载阅读设置失败:', e)
    }
  }
  applySettings()
}

/**
 * 保存设置
 */
function saveSettings() {
  const payload: StoredReaderSettings = {
    readerSettingsSchemaVersion: READER_SETTINGS_SCHEMA_VERSION,
    ...settings.value
  }
  localStorage.setItem(READER_SETTINGS_KEY, JSON.stringify(payload))
}

/**
 * 应用阅读器设置到页面级 reader owner 变量
 */
function applySettings() {
  document.documentElement.style.setProperty('--reader-page-background', settings.value.bgColor)
  document.documentElement.style.setProperty('--reader-image-width', `${settings.value.imageWidth}%`)
  document.documentElement.style.setProperty('--reader-gap', `${settings.value.imageGap}px`)
  emit('settingsChange', settings.value)
}

/**
 * 更新图片宽度设置
 */
function updateImageWidth(value: number) {
  settings.value.imageWidth = value
  applySettings()
  saveSettings()
}

/**
 * 更新图片间距设置
 */
function updateImageGap(value: number) {
  settings.value.imageGap = value
  applySettings()
  saveSettings()
}

/**
 * 更新背景颜色设置
 */
function updateBgColor(color: string) {
  settings.value.bgColor = color
  applySettings()
  saveSettings()
}

/**
 * 导航到上一章/下一章
 */
function navigateChapter(direction: 'prev' | 'next') {
  emit('navigateChapter', direction)
}

// ==================== 生命周期 ====================

onMounted(() => {
  // 加载设置
  loadSettings()

  // 初始化事件监听
  window.addEventListener('scroll', handleScroll)
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  // 移除事件监听
  window.removeEventListener('scroll', handleScroll)
  document.removeEventListener('keydown', handleKeydown)

  document.documentElement.style.removeProperty('--reader-page-background')
  document.documentElement.style.removeProperty('--reader-image-width')
  document.documentElement.style.removeProperty('--reader-gap')
})

// 暴露方法给父组件
defineExpose({
  openSettings,
  closeSettings,
  settings
})
</script>

<template>
  <!-- 章节导航 -->
  <OverlayLayer v-if="showChapterNav" class="reader-controls__chapter-nav-layer" passthrough>
    <nav id="chapterNav" class="reader-controls__chapter-nav">
      <UiButton
        variant="toolbar"
        id="prevChapterBtn"
        class="reader-controls__nav-button"
        :disabled="!hasPrevChapter"
        @click="navigateChapter('prev')"
      >
        <span class="reader-controls__nav-icon">◀</span>
        <span class="reader-controls__nav-text">上一章</span>
      </UiButton>
      <UiButton
        variant="toolbar"
        id="nextChapterBtn"
        class="reader-controls__nav-button"
        :disabled="!hasNextChapter"
        @click="navigateChapter('next')"
      >
        <span class="reader-controls__nav-text">下一章</span>
        <span class="reader-controls__nav-icon">▶</span>
      </UiButton>
    </nav>
  </OverlayLayer>

  <!-- 回到顶部按钮 -->
  <OverlayLayer v-show="showScrollTopBtn" class="reader-controls__scroll-top-layer" passthrough>
    <UiButton
      variant="toolbar"
      id="scrollTopBtn"
      class="reader-controls__scroll-top-button"
      title="回到顶部"
      aria-label="回到顶部"
      @click="scrollToTop"
    >
      <span>↑</span>
    </UiButton>
  </OverlayLayer>

  <!-- 阅读设置面板 -->
  <OverlayLayer v-if="isSettingsPanelOpen" id="settingsPanel" class="reader-controls__settings-panel" level="popover">
    <div class="reader-controls__settings-overlay" @click="closeSettings"></div>
    <div class="reader-controls__settings-content">
      <div class="reader-controls__settings-header">
        <h3>阅读设置</h3>
        <UiButton variant="toolbar" class="reader-controls__close-button" aria-label="关闭阅读设置" @click="closeSettings">×</UiButton>
      </div>
      <div class="reader-controls__settings-body">
        <!-- 图片宽度设置 -->
        <div class="reader-controls__setting-item">
          <label>图片宽度</label>
          <div class="reader-controls__setting-control">
            <UiInput
              type="range"
              id="imageWidthSlider"
              min="50"
              max="100"
              :value="settings.imageWidth"
              @input="updateImageWidth(Number(($event.target as HTMLInputElement).value))"
            />
            <span id="imageWidthValue">{{ settings.imageWidth }}%</span>
          </div>
        </div>

        <!-- 图片间距设置 -->
        <div class="reader-controls__setting-item">
          <label>图片间距</label>
          <div class="reader-controls__setting-control">
            <UiInput
              type="range"
              id="imageGapSlider"
              min="0"
              max="50"
              :value="settings.imageGap"
              @input="updateImageGap(Number(($event.target as HTMLInputElement).value))"
            />
            <span id="imageGapValue">{{ settings.imageGap }}px</span>
          </div>
        </div>

        <!-- 背景颜色设置 -->
        <div class="reader-controls__setting-item">
          <label>背景颜色</label>
          <div class="reader-controls__setting-control reader-controls__bg-options">
            <UiButton
              variant="toolbar"
              v-for="preset in bgColorPresets"
              :key="preset.color"
              class="reader-controls__bg-option"
              :class="{ active: settings.bgColor === preset.color }"
              :data-bg="preset.color"
              :style="{ background: preset.color }"
              :title="preset.name"
              :aria-label="`设置背景颜色为${preset.name}`"
              @click="updateBgColor(preset.color)"
            ></UiButton>
          </div>
        </div>
      </div>
    </div>
  </OverlayLayer>
</template>

<style scoped>
/* ==================== ReaderControls样式 ==================== */

/* 章节导航 */
.reader-controls__chapter-nav-layer,
.reader-controls__scroll-top-layer,
.reader-controls__settings-panel {
  --reader-controls-border-default: rgba(255, 255, 255, .2);
  --reader-controls-border-strong: rgba(255, 255, 255, .1);
  --reader-controls-shadow-default: rgba(102, 126, 234, .5);
  --reader-controls-shadow-raised: rgba(0, 0, 0, .3);
  --reader-controls-shadow-floating: rgba(102, 126, 234, .3);
  --reader-controls-surface-base: rgba(26, 26, 46, .95);
  --reader-controls-surface-raised: rgba(26, 26, 46, .8);
  --reader-controls-surface-muted: rgba(255, 255, 255, .1);
  --reader-controls-surface-subtle: rgba(255, 255, 255, .2);
  --reader-controls-surface-hover: rgba(0, 0, 0, .5);
  --reader-controls-surface-active: #2d2d44;
  --reader-controls-text-primary: rgba(255, 255, 255, .7);
}

.reader-controls__chapter-nav-layer {
  display: flex;
  align-items: flex-end;
  justify-content: center;
}

.reader-controls__chapter-nav {
  height: 60px;
  width: 100%;
  background: linear-gradient(to top, var(--reader-controls-surface-base), var(--reader-controls-surface-raised));
  backdrop-filter: blur(10px);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 24px;
  padding: 0 16px;
}

.reader-controls__nav-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  background: var(--reader-controls-surface-muted);
  border: 1px solid var(--reader-controls-border-default);
  border-radius: 8px;
  color: white;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.reader-controls__nav-button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.reader-controls__nav-button:hover:not(:disabled) {
  background: var(--reader-controls-surface-subtle);
}

.reader-controls__nav-icon {
  font-size: 12px;
}

/* 回到顶部按钮 */
.reader-controls__scroll-top-layer {
  display: flex;
  align-items: flex-end;
  justify-content: flex-end;
  padding: 0 24px 80px 0;
}

.reader-controls__scroll-top-button {
  width: 48px;
  height: 48px;
  background: var(--color-action-primary, var(--color-surface-brand-gradient-start));
  border: none;
  border-radius: 50%;
  color: white;
  font-size: 20px;
  cursor: pointer;
  box-shadow: 0 4px 12px var(--shadow-brand-soft);
  transition: all 0.3s;
  z-index: var(--z-dropdown);
}

.reader-controls__scroll-top-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px var(--reader-controls-shadow-default);
}

/* 设置面板 */
.reader-controls__settings-panel {
  display: block;
}

.reader-controls__settings-overlay {
  position: absolute;
  inset: 0;
  background: var(--reader-controls-surface-hover);
}

.reader-controls__settings-content {
  position: absolute;
  top: 56px;
  right: 16px;
  width: 300px;
  background: var(--reader-controls-surface-active);
  border-radius: 12px;
  box-shadow: 0 8px 32px var(--reader-controls-shadow-raised);
  overflow: hidden;
}

.reader-controls__settings-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid var(--reader-controls-border-strong);
  background: var(--reader-controls-surface-active);
}

.reader-controls__settings-header h3 {
  margin: 0;
  color: white;
  font-size: 16px;
  font-weight: 500;
}

.reader-controls__close-button {
  width: 28px;
  height: 28px;
  background: var(--reader-controls-surface-muted);
  border: none;
  border-radius: 50%;
  color: white;
  font-size: 18px;
  cursor: pointer;
  transition: background 0.2s;
}

.reader-controls__close-button:hover {
  background: var(--reader-controls-surface-subtle);
}

.reader-controls__settings-body {
  padding: 16px;
  background: var(--reader-controls-surface-active);
}

.reader-controls__setting-item {
  margin-bottom: 20px;
}

.reader-controls__setting-item:last-child {
  margin-bottom: 0;
}

.reader-controls__setting-item label {
  display: block;
  color: var(--reader-controls-text-primary);
  font-size: 13px;
  margin-bottom: 8px;
}

.reader-controls__setting-control {
  display: flex;
  align-items: center;
  gap: 12px;
}

.reader-controls__setting-control input[type="range"] {
  flex: 1;
  height: 4px;
  appearance: none;
  background: var(--reader-controls-surface-subtle);
  border-radius: 2px;
  outline: none;
}

.reader-controls__setting-control input[type="range"]::-webkit-slider-thumb {
  appearance: none;
  width: 16px;
  height: 16px;
  background: var(--color-surface-brand-gradient-start);
  border-radius: 50%;
  cursor: pointer;
}

.reader-controls__setting-control span {
  color: white;
  font-size: 13px;
  min-width: 45px;
  text-align: right;
}

.reader-controls__bg-options {
  display: flex;
  gap: 8px;
}

.reader-controls__bg-option {
  width: 32px;
  height: 32px;
  border: 2px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.reader-controls__bg-option:hover {
  transform: scale(1.1);
}

.reader-controls__bg-option.active {
  border-color: var(--color-border-brand-gradient);
  box-shadow: 0 0 0 2px var(--reader-controls-shadow-floating);
}

/* 响应式设计 */
@media (--breakpoint-md-down) {
  .reader-controls__settings-content {
    right: 8px;
    left: 8px;
    width: auto;
  }

  .reader-controls__nav-button {
    padding: 10px 16px;
    font-size: 13px;
  }

  .reader-controls__scroll-top-button {
    right: 16px;
    bottom: 72px;
    width: 40px;
    height: 40px;
  }
}

@media (--breakpoint-xs-down) {
  .reader-header {
    padding: 0 8px;
  }

  .book-info {
    display: none;
  }

  .view-mode-toggle {
    gap: 0;
  }
}
</style>
