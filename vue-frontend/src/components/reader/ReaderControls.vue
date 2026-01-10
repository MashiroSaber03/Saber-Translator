<script setup lang="ts">
/**
 * 阅读器控制组件
 * 包含页码指示器、阅读设置面板、章节导航、回到顶部按钮、键盘快捷键
 */
import { ref, computed, onMounted, onUnmounted } from 'vue'

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

// ==================== 计算属性 ====================

/**
 * 页码显示文本
 */
const pageInfoText = computed(() => {
  if (props.totalPages === 0) {
    return '- / -'
  }
  return `${props.currentPage} / ${props.totalPages}`
})

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
 * 应用设置到页面
 * 使用 CSS 变量而不是直接设置 body 样式，避免污染全局
 */
function applySettings() {
  // 使用 CSS 变量，让 ReaderView 的容器来应用背景色
  document.documentElement.style.setProperty('--reader-bg-color', settings.value.bgColor)
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
  
  // 清除阅读器的 CSS 变量
  document.documentElement.style.removeProperty('--reader-bg-color')
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
    <button 
      id="prevChapterBtn" 
      class="nav-btn" 
      :disabled="!hasPrevChapter"
      @click="navigateChapter('prev')"
    >
      <span class="nav-icon">◀</span>
      <span class="nav-text">上一章</span>
    </button>
    <button 
      id="nextChapterBtn" 
      class="nav-btn" 
      :disabled="!hasNextChapter"
      @click="navigateChapter('next')"
    >
      <span class="nav-text">下一章</span>
      <span class="nav-icon">▶</span>
    </button>
  </nav>

  <!-- 回到顶部按钮 -->
  <button 
    v-show="showScrollTopBtn"
    id="scrollTopBtn" 
    class="scroll-top-btn" 
    title="回到顶部"
    @click="scrollToTop"
  >
    <span>↑</span>
  </button>

  <!-- 阅读设置面板 -->
  <div id="settingsPanel" class="settings-panel" :class="{ active: isSettingsPanelOpen }">
    <div class="settings-overlay" @click="closeSettings"></div>
    <div class="settings-content">
      <div class="settings-header">
        <h3>阅读设置</h3>
        <button class="close-btn" @click="closeSettings">×</button>
      </div>
      <div class="settings-body">
        <!-- 图片宽度设置 -->
        <div class="setting-item">
          <label>图片宽度</label>
          <div class="setting-control">
            <input 
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
            <input 
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
            <button 
              v-for="preset in bgColorPresets"
              :key="preset.color"
              class="bg-option" 
              :class="{ active: settings.bgColor === preset.color }"
              :data-bg="preset.color" 
              :style="{ background: preset.color }"
              :title="preset.name"
              @click="updateBgColor(preset.color)"
            ></button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ==================== ReaderControls 完整样式 - 从 reader.css 迁移 ==================== */

/* 头部样式 */
.reader-header {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 56px;
  background: var(--header-bg, linear-gradient(135deg, #667eea 0%, #764ba2 100%));
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
  z-index: 1000;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

.header-left,
.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.header-center {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
}

.header-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.15);
  border: none;
  border-radius: 8px;
  color: white;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.header-btn:hover {
  background: rgba(255, 255, 255, 0.25);
}

.header-btn.primary {
  background: rgba(255, 255, 255, 0.9);
  color: #667eea;
}

.header-btn.primary:hover {
  background: white;
}

.btn-icon {
  font-size: 16px;
}

.book-info {
  display: flex;
  align-items: center;
  gap: 8px;
  color: white;
  font-size: 14px;
}

.book-title {
  font-weight: 600;
  max-width: 200px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.separator {
  opacity: 0.6;
}

.chapter-title {
  opacity: 0.9;
  max-width: 150px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.page-info {
  color: white;
  font-size: 14px;
  opacity: 0.9;
}

/* 查看模式切换 */
.view-mode-toggle {
  display: flex;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  overflow: hidden;
}

.mode-btn {
  padding: 8px 16px;
  background: transparent;
  border: none;
  color: rgba(255, 255, 255, 0.7);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
}

.mode-btn:hover {
  color: white;
}

.mode-btn.active {
  background: rgba(255, 255, 255, 0.9);
  color: #667eea;
  font-weight: 500;
}

/* 章节导航 */
.chapter-nav {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  height: 60px;
  background: linear-gradient(to top, rgba(26, 26, 46, 0.95), rgba(26, 26, 46, 0.8));
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
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 8px;
  color: white;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.nav-btn:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.2);
}

.nav-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
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
  background: var(--primary-color, #667eea);
  border: none;
  border-radius: 50%;
  color: white;
  font-size: 20px;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
  transition: all 0.3s;
  z-index: 100;
}

.scroll-top-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(102, 126, 234, 0.5);
}

/* 设置面板 */
.settings-panel {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 2000;
  display: none;
}

.settings-panel.active {
  display: block;
}

.settings-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
}

.settings-content {
  position: absolute;
  top: 56px;
  right: 16px;
  width: 300px;
  /* 修复：使用固定的深色背景，不依赖可能未定义的CSS变量 */
  background: #2d2d44;
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  overflow: hidden;
}

.settings-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  /* 修复：确保头部背景也是深色 */
  background: #2d2d44;
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
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 50%;
  color: white;
  font-size: 18px;
  cursor: pointer;
  transition: background 0.2s;
}

.close-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.settings-body {
  padding: 16px;
  /* 修复：确保主体背景也是深色 */
  background: #2d2d44;
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
  color: rgba(255, 255, 255, 0.7);
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
  -webkit-appearance: none;
  appearance: none;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 2px;
  outline: none;
}

.setting-control input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  background: #667eea;
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
  border-color: #667eea;
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.3);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .header-btn .btn-text {
    display: none;
  }
  
  .book-title {
    max-width: 120px;
  }
  
  .chapter-title {
    max-width: 80px;
  }
  
  .header-center {
    display: none;
  }
  
  .mode-btn {
    padding: 8px 12px;
    font-size: 12px;
  }
  
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

@media (max-width: 480px) {
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
