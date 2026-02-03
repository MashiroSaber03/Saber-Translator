<!--
  编辑模式工具栏组件
  包含双行布局：第一行导航和视图控制，第二行操作工具
  从 EditWorkspace.vue 拆分出来
-->
<template>
  <div class="edit-toolbar-wrapper">
    <!-- 第一行：导航和视图控制 -->
    <div class="edit-toolbar toolbar-row-1">
      <!-- 图片导航 -->
      <div class="image-navigator">
        <button
          class="nav-btn"
          :disabled="!canGoPrevious"
          @click="$emit('go-previous-image')"
          title="上一张图片 (PageUp)"
        >
          ◀◀
        </button>
        <span
          class="image-indicator"
          @click="$emit('toggle-thumbnails')"
          title="点击展开缩略图"
        >
          图 <span>{{ currentImageIndex + 1 }}</span> / <span>{{ imageCount }}</span>
        </span>
        <button
          class="nav-btn"
          :disabled="!canGoNext"
          @click="$emit('go-next-image')"
          title="下一张图片 (PageDown)"
        >
          ▶▶
        </button>
        <button
          class="thumb-toggle-btn"
          :class="{ active: showThumbnails }"
          @click="$emit('toggle-thumbnails')"
          title="显示/隐藏缩略图"
        >
          ☷
        </button>
      </div>

      <div class="toolbar-divider"></div>

      <!-- 气泡导航 -->
      <div class="bubble-navigator">
        <button
          id="prevBubbleBtn"
          class="nav-btn"
          :disabled="!hasBubbles || selectedBubbleIndex <= 0"
          @click="$emit('select-previous-bubble')"
          title="上一个气泡 (←)"
        >
          ◀
        </button>
        <span class="bubble-indicator">
          气泡 <span id="currentBubbleNum">{{ selectedBubbleIndex >= 0 ? selectedBubbleIndex + 1 : 0 }}</span> / <span id="totalBubbleNum">{{ bubbleCount }}</span>
        </span>
        <button
          id="nextBubbleBtn"
          class="nav-btn"
          :disabled="!hasBubbles || selectedBubbleIndex >= bubbleCount - 1"
          @click="$emit('select-next-bubble')"
          title="下一个气泡 (→)"
        >
          ▶
        </button>
      </div>

      <div class="toolbar-divider"></div>

      <!-- 视图控制 -->
      <div class="view-controls">
        <button
          class="layout-toggle-btn"
          @click="$emit('toggle-layout')"
          title="切换布局：左右/上下"
        >
          <svg v-if="layoutMode === 'horizontal'" viewBox="0 0 20 20" width="16" height="16">
            <rect x="1" y="2" width="8" height="16" rx="1" fill="none" stroke="currentColor" stroke-width="1.5" />
            <rect x="11" y="2" width="8" height="16" rx="1" fill="none" stroke="currentColor" stroke-width="1.5" />
          </svg>
          <svg v-else viewBox="0 0 20 20" width="16" height="16">
            <rect x="2" y="1" width="16" height="8" rx="1" fill="none" stroke="currentColor" stroke-width="1.5" />
            <rect x="2" y="11" width="16" height="8" rx="1" fill="none" stroke="currentColor" stroke-width="1.5" />
          </svg>
        </button>
        <button
          class="view-mode-btn"
          @click="$emit('toggle-view-mode')"
          title="切换视图模式"
        >
          <span class="dual-icon">⧉</span>
        </button>
        <button
          :class="{ active: syncEnabled }"
          @click="$emit('toggle-sync')"
          title="同步缩放/拖动"
          style="font-size: 12px;"
        >
          🔗
        </button>
        <button @click="$emit('fit-to-screen')" title="适应屏幕 (双击)">⛶</button>
        <button @click="$emit('zoom-in')" title="放大 (+)">+</button>
        <span id="zoomLevel">{{ Math.round(scale * 100) }}%</span>
        <button @click="$emit('zoom-out')" title="缩小 (-)">−</button>
        <button @click="$emit('reset-zoom')" title="原始大小">1:1</button>
        <button 
            id="openSettingsBtn"
            class="settings-header-btn" 
            title="打开设置"
            @click="openSettings()"
          >
            <span class="icon">⚙️</span>
         </button>
      </div>

      <div class="toolbar-spacer"></div>

      <button class="secondary-btn" @click="$emit('exit-edit-mode')">退出编辑</button>
    </div>

    <!-- 第二行：操作工具 -->
    <div class="edit-toolbar toolbar-row-2">
      <!-- 气泡操作工具组 -->
      <div class="annotation-tools">
        <button
          class="annotation-btn detect-btn"
          @click="$emit('auto-detect-bubbles')"
          title="自动检测当前图片的文本框"
        >
          <svg viewBox="0 0 16 16" width="14" height="14">
            <circle cx="6" cy="6" r="4" fill="none" stroke="currentColor" stroke-width="1.5" />
            <path d="M9 9l4 4" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
          </svg>
          <span>检测</span>
        </button>
        <button
          class="annotation-btn detect-btn"
          @click="$emit('detect-all-images')"
          title="批量检测所有图片"
        >
          <svg viewBox="0 0 16 16" width="14" height="14">
            <circle cx="5" cy="5" r="2.5" fill="none" stroke="currentColor" stroke-width="1" />
            <path d="M7 7l2 2" stroke="currentColor" stroke-width="1" stroke-linecap="round" />
            <circle cx="10" cy="10" r="2.5" fill="none" stroke="currentColor" stroke-width="1" />
            <path d="M12 12l2 2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" />
          </svg>
          <span>批量检测</span>
        </button>
        <button
          class="annotation-btn primary-action-btn"
          @click="$emit('translate-with-bubbles')"
          title="使用当前文本框翻译此图片"
        >
          <svg viewBox="0 0 16 16" width="14" height="14">
            <path d="M2 3h5M4.5 3v7M2 6h5" stroke="currentColor" stroke-width="1.2" fill="none" />
            <path d="M9 13l2-7 2 7M9.5 11h3" stroke="currentColor" stroke-width="1.2" fill="none" />
          </svg>
          <span>翻译</span>
        </button>

        <div class="toolbar-divider"></div>

        <button
          class="annotation-btn"
          :class="{ active: isDrawingMode }"
          @click="$emit('toggle-drawing-mode')"
          title="添加气泡框（或中键拖拽绘制）"
        >
          <svg viewBox="0 0 16 16" width="14" height="14">
            <rect x="3" y="3" width="10" height="10" rx="1" fill="none" stroke="currentColor" stroke-width="1.5" />
            <path d="M8 5v6M5 8h6" stroke="currentColor" stroke-width="1.5" />
          </svg>
          <span>添加</span>
        </button>
        <button
          class="annotation-btn"
          :disabled="!hasSelection"
          @click="$emit('delete-selected-bubbles')"
          title="删除选中气泡框 (Delete)"
        >
          <svg viewBox="0 0 16 16" width="14" height="14">
            <rect x="3" y="3" width="10" height="10" rx="1" fill="none" stroke="currentColor" stroke-width="1.5" />
            <path d="M5 8h6" stroke="currentColor" stroke-width="1.5" />
          </svg>
          <span>删除</span>
        </button>
        <button
          class="annotation-btn"
          :class="{ 'is-loading': isRepairLoading }"
          :disabled="!hasSelection || isRepairLoading"
          @click="$emit('repair-selected-bubble')"
          title="修复选中气泡背景 (R)"
        >
          <svg viewBox="0 0 16 16" width="14" height="14" :class="{ 'spin-icon': isRepairLoading }">
            <path d="M2 14l3-3m0 0l6-6 3 3-6 6m-3 0l-1 1 1-1z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
            <path d="M11 5l-1-1 2-2 2 2-2 2-1-1z" fill="currentColor" />
          </svg>
          <span>修复</span>
        </button>

        <div class="toolbar-divider"></div>

        <!-- 笔刷工具 -->
        <button
          class="annotation-btn brush-btn"
          :class="{ active: brushMode === 'repair' }"
          @click="$emit('activate-repair-brush')"
          title="修复笔刷 (按住R+左键拖拽)"
        >
          <svg viewBox="0 0 16 16" width="14" height="14">
            <circle cx="8" cy="8" r="5" fill="none" stroke="currentColor" stroke-width="1.5" />
            <circle cx="8" cy="8" r="2" fill="currentColor" />
          </svg>
          <span>修复笔刷</span>
        </button>
        <button
          class="annotation-btn brush-btn"
          :class="{ active: brushMode === 'restore' }"
          @click="$emit('activate-restore-brush')"
          title="还原笔刷 (按住U+左键拖拽)"
        >
          <svg viewBox="0 0 16 16" width="14" height="14">
            <circle cx="8" cy="8" r="5" fill="none" stroke="currentColor" stroke-width="1.5" />
            <path d="M5 8h6M8 5v6" stroke="currentColor" stroke-width="1" transform="rotate(45 8 8)" />
          </svg>
          <span>还原笔刷</span>
        </button>
        <span v-if="brushMode" class="brush-size-indicator">
          笔刷: {{ brushSize }}px
        </span>

        <!-- 快捷键帮助 -->
        <div class="help-tooltip-container">
          <button class="help-tooltip-btn" title="快捷键操作帮助">
            <svg viewBox="0 0 16 16" width="14" height="14">
              <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" stroke-width="1.2" />
              <text x="8" y="11" text-anchor="middle" font-size="9" font-weight="bold" fill="currentColor">?</text>
            </svg>
            <span class="help-btn-text">快捷键</span>
          </button>
          <div class="help-tooltip-popup">
            <div class="help-section">
              <div class="help-title">🖱️ 鼠标操作</div>
              <div class="help-item"><span class="help-key">左键点击气泡</span><span class="help-desc">选择气泡</span></div>
              <div class="help-item"><span class="help-key">Shift+左键点击</span><span class="help-desc">多选气泡</span></div>
              <div class="help-item"><span class="help-key">左键拖拽四角/边</span><span class="help-desc">调整大小</span></div>
              <div class="help-item"><span class="help-key">左键拖拽框内部</span><span class="help-desc">移动气泡框</span></div>
              <div class="help-item"><span class="help-key">中键拖拽</span><span class="help-desc">绘制新气泡框</span></div>
            </div>
            <div class="help-section">
              <div class="help-title">⌨️ 快捷键</div>
              <div class="help-item"><span class="help-key">A / D</span><span class="help-desc">切换上/下一张图片</span></div>
              <div class="help-item"><span class="help-key">Ctrl+Enter</span><span class="help-desc">应用并跳转下一张</span></div>
              <div class="help-item"><span class="help-key">Delete / Backspace</span><span class="help-desc">删除选中气泡</span></div>
              <div class="help-item"><span class="help-key">按住R+左键拖拽</span><span class="help-desc">修复笔刷</span></div>
              <div class="help-item"><span class="help-key">按住U+左键拖拽</span><span class="help-desc">还原笔刷</span></div>
              <div class="help-item"><span class="help-key">笔刷模式下滚轮</span><span class="help-desc">调整笔刷大小</span></div>
            </div>
          </div>
        </div>
      </div>

      <!-- 笔刷光标 -->
      <div
        v-if="brushMode"
        class="brush-cursor"
        :style="brushCursorStyle"
      ></div>

      <!-- 笔刷模式提示 -->
      <div v-if="brushMode" class="brush-mode-hint">
        {{ brushMode === 'repair' ? '修复笔刷 (R)' : '还原笔刷 (U)' }} - 滚轮调整大小
      </div>

      <!-- 进度条显示（紧跟快捷键图标右侧，复刻原版位置） -->
      <div 
        v-if="isProcessing" 
        class="edit-progress-container"
        :class="{ completed: isProgressCompleted }"
      >
        <div class="edit-progress-info">
          <span class="edit-progress-text">{{ progressText }}</span>
          <span class="edit-progress-count">{{ progressCurrent }}/{{ progressTotal }}</span>
        </div>
        <div class="edit-progress-bar">
          <div 
            class="edit-progress-fill" 
            :class="{ animating: !isProgressCompleted }"
            :style="{ width: progressPercent + '%' }"
          ></div>
        </div>
      </div>

      <div class="toolbar-spacer"></div>

      <!-- 快捷操作 -->
      <div class="quick-actions">
        <button class="primary-btn" @click="$emit('apply-and-next')" title="应用更改并跳转下一张 (Ctrl+Enter)">
          应用并下一张
        </button>
      </div>
    </div>

        <!-- 设置模态框 -->
    <SettingsModal 
      v-model="showSettingsModal"
      :initial-tab="settingsInitialTab"
      @save="handleSettingsSave"
    />
    
  </div>
</template>

<script setup lang="ts">
/**
 * 编辑模式工具栏组件
 * 包含双行布局：第一行导航和视图控制，第二行操作工具
 */
import { computed, ref } from 'vue'
import { showToast } from '@/utils/toast';
import SettingsModal from '@/components/settings/SettingsModal.vue'

// ============================================================
// Props
// ============================================================

const props = defineProps<{
  /** 当前图片索引 */
  currentImageIndex: number
  /** 图片总数 */
  imageCount: number
  /** 是否可以切换到上一张 */
  canGoPrevious: boolean
  /** 是否可以切换到下一张 */
  canGoNext: boolean
  /** 是否显示缩略图 */
  showThumbnails: boolean
  /** 是否有气泡 */
  hasBubbles: boolean
  /** 选中的气泡索引 */
  selectedBubbleIndex: number
  /** 气泡总数 */
  bubbleCount: number
  /** 布局模式 */
  layoutMode: 'horizontal' | 'vertical'
  /** 是否同步 */
  syncEnabled: boolean
  /** 缩放比例 */
  scale: number
  /** 是否处于绘制模式 */
  isDrawingMode: boolean
  /** 是否有选中 */
  hasSelection: boolean
  /** 笔刷模式 */
  brushMode: 'repair' | 'restore' | null
  /** 笔刷大小 */
  brushSize: number
  /** 鼠标X坐标 */
  mouseX: number
  /** 鼠标Y坐标 */
  mouseY: number
  /** 是否正在处理 */
  isProcessing: boolean
  /** 进度文本 */
  progressText: string
  /** 当前进度 */
  progressCurrent: number
  /** 总进度 */
  progressTotal: number
  /** 修复气泡背景中 */
  isRepairLoading?: boolean
}>()

// ============================================================
// Emits
// ============================================================

defineEmits<{
  /** 切换到上一张图片 */
  (e: 'go-previous-image'): void
  /** 切换到下一张图片 */
  (e: 'go-next-image'): void
  /** 切换缩略图显示 */
  (e: 'toggle-thumbnails'): void
  /** 选择上一个气泡 */
  (e: 'select-previous-bubble'): void
  /** 选择下一个气泡 */
  (e: 'select-next-bubble'): void
  /** 切换布局 */
  (e: 'toggle-layout'): void
  /** 切换视图模式 */
  (e: 'toggle-view-mode'): void
  /** 切换同步 */
  (e: 'toggle-sync'): void
  /** 适应屏幕 */
  (e: 'fit-to-screen'): void
  /** 放大 */
  (e: 'zoom-in'): void
  /** 缩小 */
  (e: 'zoom-out'): void
  /** 重置缩放 */
  (e: 'reset-zoom'): void
  /** 退出编辑模式 */
  (e: 'exit-edit-mode'): void
  /** 自动检测气泡 */
  (e: 'auto-detect-bubbles'): void
  /** 批量检测所有图片 */
  (e: 'detect-all-images'): void
  /** 使用当前气泡翻译 */
  (e: 'translate-with-bubbles'): void
  /** 切换绘制模式 */
  (e: 'toggle-drawing-mode'): void
  /** 删除选中气泡 */
  (e: 'delete-selected-bubbles'): void
  /** 修复选中气泡 */
  (e: 'repair-selected-bubble'): void
  /** 激活修复笔刷 */
  (e: 'activate-repair-brush'): void
  /** 激活还原笔刷 */
  (e: 'activate-restore-brush'): void
  /** 应用并下一张 */
  (e: 'apply-and-next'): void
  /** 打开设置 */
  (e: 'open-settings'): void
}>()

// ============================================================
// Methods
// ============================================================


/** 是否显示设置模态框 */
const showSettingsModal = ref(false)
/** 设置模态框初始Tab（用于插件管理直接跳转） */
const settingsInitialTab = ref<string | undefined>(undefined)

/** 打开设置面板 */
function openSettings(initialTab?: string) {
  settingsInitialTab.value = initialTab
  showSettingsModal.value = true
}

function handleSettingsSave() {
  showToast('设置已保存', 'success')
}



// ============================================================
// 计算属性
// ============================================================

/** 进度百分比 */
const progressPercent = computed(() => {
  if (props.progressTotal === 0) return 0
  return Math.round((props.progressCurrent / props.progressTotal) * 100)
})

/** 进度是否完成（复刻原版状态控制） */
const isProgressCompleted = computed(() => {
  return props.progressTotal > 0 && props.progressCurrent >= props.progressTotal
})

/** 笔刷光标样式 */
const brushCursorStyle = computed(() => {
  const color = props.brushMode === 'repair' 
    ? { fill: 'rgba(76, 175, 80, 0.4)', border: '#4CAF50' }
    : { fill: 'rgba(33, 150, 243, 0.4)', border: '#2196F3' }
  
  return {
    position: 'fixed' as const,
    left: `${props.mouseX}px`,
    top: `${props.mouseY}px`,
    width: `${props.brushSize}px`,
    height: `${props.brushSize}px`,
    borderRadius: '50%',
    border: `2px solid ${color.border}`,
    backgroundColor: color.fill,
    pointerEvents: 'none' as const,
    zIndex: 99999,
    transform: 'translate(-50%, -50%)',
    display: props.brushMode ? 'block' : 'none'
  }
})
</script>

<style scoped>
/* 顶部工具栏 */
.edit-toolbar-wrapper {
  flex-shrink: 0;
  background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.edit-toolbar {
  display: flex;
  align-items: center;
  padding: 8px 15px;
  gap: 10px;
}

.toolbar-row-1 {
  border-bottom: 1px solid rgba(255,255,255,0.05);
}

.toolbar-row-2 {
  background: rgba(0,0,0,0.15);
}

.toolbar-spacer {
  flex: 1;
}

.toolbar-divider {
  width: 1px;
  height: 24px;
  background: rgba(255,255,255,0.2);
  margin: 0 5px;
}

/* 图片导航 */
.image-navigator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.image-navigator .nav-btn {
  width: 36px;
  height: 32px;
  border: none;
  border-radius: 6px;
  background: rgba(102, 126, 234, 0.3);
  color: #fff;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.image-navigator .nav-btn:hover {
  background: rgba(102, 126, 234, 0.5);
}

.image-navigator .nav-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.image-indicator {
  color: #fff;
  font-size: 14px;
  padding: 6px 12px;
  background: rgba(102, 126, 234, 0.2);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.image-indicator:hover {
  background: rgba(102, 126, 234, 0.4);
}

.image-indicator span {
  font-weight: bold;
  color: #667eea;
}

.thumb-toggle-btn {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 6px;
  background: rgba(255,255,255,0.1);
  color: #fff;
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s;
}

.thumb-toggle-btn:hover {
  background: rgba(255,255,255,0.2);
}

.thumb-toggle-btn.active {
  background: rgba(102, 126, 234, 0.5);
}

/* 气泡导航 */
.bubble-navigator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.bubble-navigator .nav-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background: rgba(102, 126, 234, 0.3);
  color: #fff;
  cursor: pointer;
  font-size: 10px;
  transition: all 0.2s;
}

.bubble-navigator .nav-btn:hover {
  background: rgba(102, 126, 234, 0.5);
}

.bubble-navigator .nav-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.bubble-indicator {
  color: #fff;
  font-size: 13px;
  padding: 4px 10px;
  background: rgba(0,0,0,0.3);
  border-radius: 6px;
}

.bubble-indicator span {
  font-weight: bold;
  color: #00ff88;
}

/* 视图控制按钮组 */
.view-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

.view-controls button {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 6px;
  background: rgba(255,255,255,0.1);
  color: #fff;
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s;
}

.view-controls button:hover {
  background: rgba(255,255,255,0.2);
}

.view-controls #zoomLevel {
  min-width: 50px;
  text-align: center;
  color: #fff;
  font-size: 13px;
  padding: 0 8px;
}

.view-mode-btn {
  font-size: 18px !important;
}

/* 快捷操作 */
.quick-actions {
  display: flex;
  gap: 10px;
}

.primary-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
  color: #1a1a2e;
  font-weight: 600;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.primary-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 255, 136, 0.3);
}

.secondary-btn {
  padding: 8px 16px;
  border: 1px solid rgba(255,255,255,0.3);
  border-radius: 6px;
  background: transparent;
  color: #fff;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.secondary-btn:hover {
  background: rgba(255,255,255,0.1);
  border-color: rgba(255,255,255,0.5);
}

/* 导航按钮样式 - 与原版一致 */
.image-navigator .nav-btn,
.bubble-navigator .nav-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background: rgba(102, 126, 234, 0.3);
  color: #fff;
  cursor: pointer;
  font-size: 10px;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  line-height: 1;
}

.image-navigator .nav-btn:hover,
.bubble-navigator .nav-btn:hover {
  background: rgba(102, 126, 234, 0.5);
}

/* 进度条样式（完整复刻原版） */
.edit-progress-container {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 16px;
  margin-left: 12px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 20px;
  min-width: 200px;
  max-width: 350px;
  animation: progressFadeIn 0.3s ease;
}

@keyframes progressFadeIn {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.edit-progress-info {
  display: flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
}

.edit-progress-text {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.9);
  font-weight: 500;
}

.edit-progress-count {
  font-size: 12px;
  color: #00ff88;
  font-weight: 600;
  font-family: 'Consolas', 'Monaco', monospace;
}

.edit-progress-bar {
  flex: 1;
  height: 6px;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 3px;
  overflow: hidden;
  min-width: 80px;
}

.edit-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #00ff88, #00d4ff);
  border-radius: 3px;
  transition: width 0.3s ease;
  box-shadow: 0 0 8px rgba(0, 255, 136, 0.5);
}

/* 进度条动画效果（仅在进行中时播放） */
.edit-progress-fill.animating {
  background: linear-gradient(90deg, #00ff88, #00d4ff, #00ff88);
  background-size: 200% 100%;
  animation: progressShine 1.5s ease-in-out infinite;
}

@keyframes progressShine {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* 完成状态 */
.edit-progress-container.completed .edit-progress-fill {
  background: #00ff88;
  animation: none;
}

.edit-progress-container.completed .edit-progress-text {
  color: #00ff88;
}

/* 修复按钮 Loading 状态 */
.annotation-btn.is-loading {
  opacity: 0.7;
  cursor: wait;
  pointer-events: none;
}

.annotation-btn.is-loading .spin-icon {
  animation: spin-repair-icon 1s linear infinite;
}

@keyframes spin-repair-icon {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

/* 笔刷大小指示器 */
.brush-size-indicator {
  color: #fff;
  font-size: 12px;
  padding: 4px 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  margin-left: 8px;
}

/* 激活状态按钮 */
.annotation-btn.active,
.brush-btn.active {
  background: rgba(102, 126, 234, 0.5) !important;
  border-color: #667eea !important;
}

/* 笔刷光标 */
.brush-cursor {
  pointer-events: none;
  transition: width 0.1s, height 0.1s;
}

/* 笔刷模式提示 */
.brush-mode-hint {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  padding: 8px 16px;
  background: rgba(0, 0, 0, 0.8);
  color: #fff;
  border-radius: 6px;
  font-size: 13px;
  z-index: 10000;
  pointer-events: none;
}

/* ============ 快捷键帮助提示框样式 ============ */

.help-tooltip-container {
  position: relative;
  display: inline-flex;
}

.help-tooltip-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  height: 28px;
  padding: 0 10px;
  border: 1px solid #cfd6e4;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.9);
  color: #666;
  cursor: pointer;
  transition: all 0.2s;
}

.help-btn-text {
  font-size: 12px;
  font-weight: 500;
  white-space: nowrap;
}

.help-tooltip-btn:hover {
  background: #fff;
  border-color: #5b73f2;
  color: #5b73f2;
}

.help-tooltip-popup {
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 8px;
  min-width: 260px;
  padding: 12px 14px;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  opacity: 0;
  visibility: hidden;
  transform: translateY(-5px);
  transition: all 0.2s ease;
}

.help-tooltip-container:hover .help-tooltip-popup,
.help-tooltip-popup:hover {
  opacity: 1;
  visibility: visible;
  transform: translateY(0);
}

.help-section {
  margin-bottom: 10px;
}

.help-section:last-child {
  margin-bottom: 0;
}

.help-title {
  font-size: 12px;
  font-weight: 600;
  color: #374151;
  margin-bottom: 6px;
  padding-bottom: 4px;
  border-bottom: 1px solid #e5e7eb;
}

.help-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 3px 0;
  font-size: 11px;
}

.help-key {
  color: #6b7280;
  font-family: 'SF Mono', Monaco, 'Courier New', monospace;
  background: #f3f4f6;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
}

.help-desc {
  color: #374151;
}

/* annotation-tools 样式 */
.annotation-tools {
  display: flex;
  align-items: center;
  gap: 6px;
}

.annotation-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 10px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.annotation-btn:hover {
  background: rgba(255, 255, 255, 0.2);
  border-color: rgba(255, 255, 255, 0.3);
}

.annotation-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.annotation-btn svg {
  flex-shrink: 0;
}

.annotation-btn span {
  white-space: nowrap;
}

.detect-btn {
  background: rgba(102, 126, 234, 0.3);
  border-color: rgba(102, 126, 234, 0.5);
}

.detect-btn:hover {
  background: rgba(102, 126, 234, 0.5);
}

.primary-action-btn {
  background: rgba(0, 255, 136, 0.2);
  border-color: rgba(0, 255, 136, 0.4);
  color: #00ff88;
}

.primary-action-btn:hover {
  background: rgba(0, 255, 136, 0.3);
}

.brush-btn {
  background: rgba(255, 193, 7, 0.2);
  border-color: rgba(255, 193, 7, 0.4);
}

.brush-btn:hover {
  background: rgba(255, 193, 7, 0.3);
}

.image-navigator .nav-btn:disabled,
.bubble-navigator .nav-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

/* sync按钮激活状态 */
.view-controls button.active {
  background: rgba(102, 126, 234, 0.5);
}
</style>
