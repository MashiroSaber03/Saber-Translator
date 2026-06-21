<script setup lang="ts">

import UiInput from '@/components/ui/UiInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
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

// localStorage 存储键名
const READER_SETTINGS_KEY = 'readerSettings'

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
const settings = ref<ReaderSettings>({
  imageWidth: 100,
  imageGap: 8,
  bgColor: '#1a1a2e'
})

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

/**
 * 加载设置
 */
function loadSettings() {
  const saved = localStorage.getItem(READER_SETTINGS_KEY)
  if (saved) {
    try {
      const parsed = JSON.parse(saved)
      settings.value = { ...settings.value, ...parsed }
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
  localStorage.setItem(READER_SETTINGS_KEY, JSON.stringify(settings.value))
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
  <nav v-if="showChapterNav" id="chapterNav" class="chapter-nav">
    <UiButton
      variant="toolbar" 
      id="prevChapterBtn" 
      class="nav-btn" 
      :disabled="!hasPrevChapter"
      @click="navigateChapter('prev')"
    >
      <span class="nav-icon">◀</span>
      <span class="nav-text">上一章</span>
    </UiButton>
    <UiButton
      variant="toolbar" 
      id="nextChapterBtn" 
      class="nav-btn" 
      :disabled="!hasNextChapter"
      @click="navigateChapter('next')"
    >
      <span class="nav-text">下一章</span>
      <span class="nav-icon">▶</span>
    </UiButton>
  </nav>

  <!-- 回到顶部按钮 -->
  <UiButton
    variant="toolbar" 
    v-show="showScrollTopBtn"
    id="scrollTopBtn" 
    class="scroll-top-btn" 
    title="回到顶部"
    @click="scrollToTop"
  >
    <span>↑</span>
  </UiButton>

  <!-- 阅读设置面板 -->
  <div id="settingsPanel" class="settings-panel" :class="{ active: isSettingsPanelOpen }">
    <div class="settings-overlay" @click="closeSettings"></div>
    <div class="settings-content">
      <div class="settings-header">
        <h3>阅读设置</h3>
        <UiButton variant="toolbar" class="close-btn" @click="closeSettings">×</UiButton>
      </div>
      <div class="settings-body">
        <!-- 图片宽度设置 -->
        <div class="setting-item">
          <label>图片宽度</label>
          <div class="setting-control">
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
        <div class="setting-item">
          <label>图片间距</label>
          <div class="setting-control">
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
        <div class="setting-item">
          <label>背景颜色</label>
          <div class="setting-control bg-options">
            <UiButton
              variant="toolbar" 
              v-for="preset in bgColorPresets"
              :key="preset.color"
              class="bg-option" 
              :class="{ active: settings.bgColor === preset.color }"
              :data-bg="preset.color" 
              :style="{ background: preset.color }"
              :title="preset.name"
              @click="updateBgColor(preset.color)"
            ></UiButton>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>/* ==================== ReaderControls样式 ==================== */

/* 章节导航 */
.chapter-nav {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  height: 60px;
  background: linear-gradient(to top, var(--reader-controls-surface-base), var(--reader-controls-surface-raised));
  backdrop-filter: blur(10px);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 24px;
  padding: 0 16px;
}

.nav-btn {
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

.nav-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.nav-btn:hover:not(:disabled) {
  background: var(--reader-controls-surface-subtle);
}

.nav-icon {
  font-size: 12px;
}

/* 回到顶部按钮 */
.scroll-top-btn {
  position: fixed;
  right: 24px;
  bottom: 80px;
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

.scroll-top-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px var(--reader-controls-shadow-default);
}

/* 设置面板 */
.settings-panel {
  position: fixed;
  inset: 0;
  z-index: var(--z-popover);
  display: none;
}

.settings-panel.active {
  display: block;
}

.settings-overlay {
  position: absolute;
  inset: 0;
  background: var(--reader-controls-surface-hover);
}

.settings-content {
  position: absolute;
  top: 56px;
  right: 16px;
  width: 300px;

  /* 修复：使用固定的深色背景，不依赖可能未定义的CSS变量 */
  background: var(--reader-controls-surface-active);
  border-radius: 12px;
  box-shadow: 0 8px 32px var(--reader-controls-shadow-raised);
  overflow: hidden;
}

.settings-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid var(--reader-controls-border-strong);

  /* 修复：确保头部背景也是深色 */
  background: var(--reader-controls-surface-active);
}

.settings-header h3 {
  margin: 0;
  color: white;
  font-size: 16px;
  font-weight: 500;
}

.close-btn {
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

.close-btn:hover {
  background: var(--reader-controls-surface-subtle);
}

.settings-body {
  padding: 16px;

  /* 修复：确保主体背景也是深色 */
  background: var(--reader-controls-surface-active);
}

.setting-item {
  margin-bottom: 20px;
}

.setting-item:last-child {
  margin-bottom: 0;
}

.setting-item label {
  display: block;

  /* 修复：确保标签文字是淡白色，在深色背景上可见 */
  color: var(--reader-controls-text-primary);
  font-size: 13px;
  margin-bottom: 8px;
}

.setting-control {
  display: flex;
  align-items: center;
  gap: 12px;
}

.setting-control input[type="range"] {
  flex: 1;
  height: 4px;
  appearance: none;
  background: var(--reader-controls-surface-subtle);
  border-radius: 2px;
  outline: none;
}

.setting-control input[type="range"]::-webkit-slider-thumb {
  appearance: none;
  width: 16px;
  height: 16px;
  background: var(--color-surface-brand-gradient-start);
  border-radius: 50%;
  cursor: pointer;
}

.setting-control span {
  color: white;
  font-size: 13px;
  min-width: 45px;
  text-align: right;
}

.bg-options {
  display: flex;
  gap: 8px;
}

.bg-option {
  width: 32px;
  height: 32px;
  border: 2px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.bg-option:hover {
  transform: scale(1.1);
}

.bg-option.active {
  border-color: var(--color-border-brand-gradient);
  box-shadow: 0 0 0 2px var(--reader-controls-shadow-floating);
}

/* 响应式设计 */
@media (--breakpoint-md-down) {
  .settings-content {
    right: 8px;
    left: 8px;
    width: auto;
  }
  
  .nav-btn {
    padding: 10px 16px;
    font-size: 13px;
  }
  
  .scroll-top-btn {
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
