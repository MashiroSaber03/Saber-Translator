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
        <UiButton
          variant="toolbar"
          class="nav-btn"
          :disabled="!canGoPrevious"
          @click="$emit('go-previous-image')"
          title="上一张图片 (A)"
        >
          ◀◀
        </UiButton>
        <span
          class="image-indicator"
          @click="$emit('toggle-thumbnails')"
          title="点击展开缩略图"
        >
          图 <span>{{ currentImageIndex + 1 }}</span> / <span>{{ imageCount }}</span>
        </span>
        <UiButton
          variant="toolbar"
          class="nav-btn"
          :disabled="!canGoNext"
          @click="$emit('go-next-image')"
          title="下一张图片 (D)"
        >
          ▶▶
        </UiButton>
        <UiButton
          variant="toolbar"
          class="thumb-toggle-btn"
          :class="{ active: showThumbnails }"
          @click="$emit('toggle-thumbnails')"
          title="显示/隐藏缩略图"
        >
          ☷
        </UiButton>
      </div>

      <div class="toolbar-divider"></div>

      <!-- 气泡导航 -->
      <div class="bubble-navigator">
        <UiButton
          variant="toolbar"
          id="prevBubbleBtn"
          class="nav-btn"
          :disabled="!hasBubbles || selectedBubbleIndex <= 0"
          @click="$emit('select-previous-bubble')"
          title="上一个气泡"
        >
          ◀
        </UiButton>
        <span class="bubble-indicator">
          气泡 <span id="currentBubbleNum">{{ selectedBubbleIndex >= 0 ? selectedBubbleIndex + 1 : 0 }}</span> / <span id="totalBubbleNum">{{ bubbleCount }}</span>
        </span>
        <UiButton
          variant="toolbar"
          id="nextBubbleBtn"
          class="nav-btn"
          :disabled="!hasBubbles || selectedBubbleIndex >= bubbleCount - 1"
          @click="$emit('select-next-bubble')"
          title="下一个气泡"
        >
          ▶
        </UiButton>
      </div>

      <div class="toolbar-divider"></div>

      <!-- 视图控制 -->
      <div class="view-controls">
        <UiButton
          variant="toolbar"
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
        </UiButton>
        <UiButton
          variant="toolbar"
          class="view-mode-btn"
          @click="$emit('toggle-view-mode')"
          title="切换视图模式"
        >
          <span class="dual-icon">⧉</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          :class="{ active: syncEnabled }"
          @click="$emit('toggle-sync')"
          title="同步缩放/拖动"
          style="font-size: 12px;"
        >
          🔗
        </UiButton>
        <UiButton variant="toolbar" @click="$emit('fit-to-screen')" title="适应屏幕 (双击)">⛶</UiButton>
        <UiButton variant="toolbar" @click="$emit('zoom-in')" title="放大 (+)">+</UiButton>
        <span id="zoomLevel" class="zoom-level">{{ Math.round(scale * 100) }}%</span>
        <UiButton variant="toolbar" @click="$emit('zoom-out')" title="缩小 (-)">−</UiButton>
        <UiButton variant="toolbar" @click="$emit('reset-zoom')" title="原始大小">1:1</UiButton>
      </div>

      <div class="toolbar-spacer"></div>

      <UiButton variant="toolbar" class="action-secondary" @click="$emit('exit-edit-mode')">退出编辑</UiButton>
    </div>

    <!-- 第二行：操作工具 -->
    <div class="edit-toolbar toolbar-row-2">
      <!-- 气泡操作工具组 -->
      <div class="annotation-tools">
        <UiButton
          variant="toolbar"
          class="annotation-btn detect-btn"
          @click="$emit('auto-detect-bubbles')"
          title="自动检测当前图片的文本框"
        >
          <svg viewBox="0 0 16 16" width="14" height="14">
            <circle cx="6" cy="6" r="4" fill="none" stroke="currentColor" stroke-width="1.5" />
            <path d="M9 9l4 4" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
          </svg>
          <span>检测</span>
        </UiButton>
        <UiButton
          variant="toolbar"
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
        </UiButton>
        <UiButton
          variant="toolbar"
          class="annotation-btn primary-action-btn"
          @click="$emit('translate-with-bubbles')"
          title="使用当前文本框翻译此图片"
        >
          <svg viewBox="0 0 16 16" width="14" height="14">
            <path d="M2 3h5M4.5 3v7M2 6h5" stroke="currentColor" stroke-width="1.2" fill="none" />
            <path d="M9 13l2-7 2 7M9.5 11h3" stroke="currentColor" stroke-width="1.2" fill="none" />
          </svg>
          <span>翻译</span>
        </UiButton>

        <div class="toolbar-divider"></div>

        <UiButton
          variant="toolbar"
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
        </UiButton>
        <UiButton
          variant="toolbar"
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
        </UiButton>
        <UiButton
          variant="toolbar"
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
        </UiButton>

        <div class="toolbar-divider"></div>

        <!-- 笔刷工具 -->
        <UiButton
          variant="toolbar"
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
        </UiButton>
        <UiButton
          variant="toolbar"
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
        </UiButton>
        <span v-if="brushMode" class="brush-size-indicator">
          笔刷: {{ brushSize }}px
        </span>

        <!-- 快捷键帮助 -->
        <div class="help-tooltip-container">
          <UiButton variant="toolbar" class="help-tooltip-btn" title="快捷键操作帮助">
            <svg viewBox="0 0 16 16" width="14" height="14">
              <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" stroke-width="1.2" />
              <text x="8" y="11" text-anchor="middle" font-size="9" font-weight="bold" fill="currentColor">?</text>
            </svg>
            <span class="help-btn-text">快捷键</span>
          </UiButton>
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
      <OverlayLayer v-if="brushMode" class="brush-mode-hint-layer" passthrough>
        <div class="brush-mode-hint">
          {{ brushMode === 'repair' ? '修复笔刷 (R)' : '还原笔刷 (U)' }} - 滚轮调整大小
        </div>
      </OverlayLayer>

      <!-- 进度条显示（紧跟快捷键图标右侧，业务契约位置） -->
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
        <UiButton variant="toolbar" class="action-primary" @click="$emit('apply-and-next')" title="应用更改并跳转下一张 (Ctrl+Enter)">
          应用并下一张
        </UiButton>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">

import UiButton from '@/components/ui/UiButton.vue'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
/**
 * 编辑模式工具栏组件
 * 包含双行布局：第一行导航和视图控制，第二行操作工具
 */
import { computed } from 'vue'

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
}>()

// ============================================================
// 计算属性
// ============================================================

/** 进度百分比 */
const progressPercent = computed(() => {
  if (props.progressTotal === 0) return 0
  return Math.round((props.progressCurrent / props.progressTotal) * 100)
})

/** 进度是否完成（业务契约状态控制） */
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

<style scoped>/* 顶部工具栏 */
.edit-toolbar-wrapper {
  /* owner tokens: edit-toolbar */
  --edit-toolbar-border-default: #e5e7eb;
  --edit-toolbar-border-strong: rgba(255, 255, 255, .2);
  --edit-toolbar-border-muted: rgba(255, 255, 255, .3);
  --edit-toolbar-border-subtle: rgba(102, 126, 234, .5);
  --edit-toolbar-border-hover: rgba(0, 255, 136, .4);
  --edit-toolbar-border-active: rgba(255, 193, 7, .4);
  --edit-toolbar-shadow-default: rgba(0, 255, 136, .5);
  --edit-toolbar-shadow-raised: rgba(0, 0, 0, .15);
  --edit-toolbar-surface-base: #00d4ff;
  --edit-toolbar-surface-raised: rgba(0, 0, 0, .8);
  --edit-toolbar-surface-muted: rgba(255, 255, 255, .9);
  --edit-toolbar-surface-subtle: rgba(0, 255, 136, .2);
  --edit-toolbar-surface-hover: rgba(0, 255, 136, .3);
  --edit-toolbar-surface-active: rgba(255, 193, 7, .2);
  --edit-toolbar-surface-selected: rgba(255, 193, 7, .3);
  --edit-toolbar-text-primary: #667eea;
  --edit-toolbar-text-secondary: rgba(255, 255, 255, .9);
  --edit-toolbar-text-muted: #5b73f2;
  --edit-toolbar-text-subtle: #374151;
  --edit-toolbar-text-supporting: #6b7280;

  flex-shrink: 0;
  background: linear-gradient(135deg, var(--color-edit-shell-start) 0%, var(--color-edit-shell-end) 100%);
  border-bottom: 1px solid var(--color-edit-shell-divider);
}

.edit-toolbar {
  display: flex;
  align-items: center;
  padding: 8px 15px;
  gap: 10px;
}

.toolbar-row-1 {
  border-bottom: 1px solid var(--color-edit-shell-divider-soft);
}

.toolbar-row-2 {
  background: var(--color-edit-shell-overlay);
}

.toolbar-spacer {
  flex: 1;
}

.toolbar-divider {
  width: 1px;
  height: 24px;
  background: var(--color-surface-overlay-medium-muted);
  margin: 0 5px;
}

/* 图片导航 */
.image-navigator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.image-indicator {
  color: var(--color-text-inverse);
  font-size: 14px;
  padding: 6px 12px;
  background: var(--color-edit-shell-chip);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.image-indicator:hover {
  background: var(--color-edit-shell-chip-hover);
}

.image-indicator span {
  font-weight: bold;
  color: var(--edit-toolbar-text-primary);
}

.thumb-toggle-btn {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 6px;
  background: var(--color-surface-overlay-light-muted);
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s;
}

.thumb-toggle-btn:hover {
  background: var(--color-surface-overlay-medium-muted);
}

.thumb-toggle-btn.active {
  background: var(--color-edit-shell-chip-active);
}

/* 气泡导航 */
.bubble-navigator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.bubble-indicator {
  color: var(--color-text-inverse);
  font-size: 13px;
  padding: 4px 10px;
  background: var(--color-edit-shell-progress-2);
  border-radius: 6px;
}

.bubble-indicator span {
  font-weight: bold;
  color: var(--color-edit-accent);
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
  background: var(--color-surface-overlay-light-muted);
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s;
}

.view-controls button:hover {
  background: var(--color-surface-overlay-medium-muted);
}

.view-controls .zoom-level {
  min-width: 50px;
  text-align: center;
  color: var(--color-text-inverse);
  font-size: 13px;
  padding: 0 8px;
}

.view-mode-btn {
  font-size: 18px;
}

/* 快捷操作 */
.quick-actions {
  display: flex;
  gap: 10px;
}

.action-primary {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  background: linear-gradient(135deg, var(--color-edit-action-start) 0%, var(--color-edit-action-end) 100%);
  color: var(--color-edit-action-text);
  font-weight: 600;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.action-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px var(--shadow-edit-action);
}

.action-secondary {
  padding: 8px 16px;
  border: 1px solid var(--color-edit-action-border);
  border-radius: 6px;
  background: transparent;
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.action-secondary:hover {
  background: var(--color-surface-overlay-light-muted);
  border-color: var(--color-edit-action-border-hover);
}

/* 导航按钮样式 - 按业务契约 */
.image-navigator .nav-btn,
.bubble-navigator .nav-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background: var(--color-edit-shell-control);
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 10px;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  line-height: 1;
}

.image-navigator .nav-btn:disabled,
.bubble-navigator .nav-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.image-navigator .nav-btn:not(:disabled):hover,
.bubble-navigator .nav-btn:not(:disabled):hover {
  background: var(--color-edit-shell-chip-active);
}

/* 编辑进度条 */
.edit-progress-container {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 16px;
  margin-left: 12px;
  background: var(--color-edit-shell-progress);
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
  color: var(--edit-toolbar-text-secondary);
  font-weight: 500;
}

.edit-progress-count {
  font-size: 12px;
  color: var(--color-edit-accent);
  font-weight: 600;
  font-family: var(--font-mono);
}

.edit-progress-bar {
  flex: 1;
  height: 6px;
  background: var(--color-surface-overlay-light-soft);
  border-radius: 3px;
  overflow: hidden;
  min-width: 80px;
}

.edit-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--color-edit-action-start), var(--edit-toolbar-surface-base));
  border-radius: 3px;
  transition: width 0.3s ease;
  box-shadow: 0 0 8px var(--edit-toolbar-shadow-default);
}

/* 进度条动画效果（仅在进行中时播放） */
.edit-progress-fill.animating {
  background: linear-gradient(90deg, var(--color-edit-action-start), var(--edit-toolbar-surface-base), var(--color-edit-action-start));
  background-size: 200% 100%;
  animation: progressShine 1.5s ease-in-out infinite;
}

@keyframes progressShine {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* 完成状态 */
.edit-progress-container.completed .edit-progress-fill {
  background: var(--color-edit-action-start);
  animation: none;
}

.edit-progress-container.completed .edit-progress-text {
  color: var(--color-edit-accent);
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

/* 笔刷大小指示器 */

.brush-size-indicator {
  color: var(--color-text-inverse);
  font-size: 12px;
  padding: 4px 8px;
  background: var(--color-surface-overlay-light);
  border-radius: 4px;
  margin-left: 8px;
}

/* 激活状态按钮 */
.annotation-btn.active,
.brush-btn.active {
  background: var(--color-edit-shell-chip-active);
  border-color: var(--color-border-brand-gradient);
}

/* 笔刷光标 */
.brush-cursor {
  pointer-events: none;
  transition: width 0.1s, height 0.1s;
}

/* 笔刷模式提示 */
.brush-mode-hint-layer {
  display: flex;
  align-items: flex-end;
  justify-content: center;
  padding-bottom: 20px;
}

.brush-mode-hint {
  padding: 8px 16px;
  background: var(--edit-toolbar-surface-raised);
  color: var(--color-text-inverse);
  border-radius: 6px;
  font-size: 13px;
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
  border: 1px solid var(--color-edit-input-border);
  border-radius: 14px;
  background: var(--edit-toolbar-surface-muted);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: all 0.2s;
}

.help-btn-text {
  font-size: 12px;
  font-weight: 500;
  white-space: nowrap;
}

.help-tooltip-btn:hover {
  background: var(--color-surface-base);
  border-color: var(--color-edit-input-border-focus);
  color: var(--edit-toolbar-text-muted);
}

.help-tooltip-popup {
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 8px;
  min-width: 260px;
  padding: 12px 14px;
  background: var(--color-surface-base);
  border: 1px solid var(--color-border-muted);
  border-radius: 10px;
  box-shadow: 0 4px 20px var(--edit-toolbar-shadow-raised);
  z-index: var(--z-overlay);
  opacity: 0;
  visibility: hidden;
  transform: translateY(-5px);
  transition: all 0.2s ease;
}

.help-tooltip-container:hover .help-tooltip-popup {
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
  color: var(--edit-toolbar-text-subtle);
  margin-bottom: 6px;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--edit-toolbar-border-default);
}

.help-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 3px 0;
  font-size: 11px;
}

.help-key {
  color: var(--edit-toolbar-text-supporting);
  font-family: var(--font-mono);
  background: var(--color-surface-muted);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
}

.help-desc {
  color: var(--edit-toolbar-text-subtle);
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
  border: 1px solid var(--edit-toolbar-border-strong);
  border-radius: 6px;
  background: var(--color-surface-overlay-light);
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.annotation-btn:hover {
  background: var(--color-surface-overlay-medium);
  border-color: var(--edit-toolbar-border-muted);
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
  background: var(--color-edit-shell-control);
  border-color: var(--edit-toolbar-border-subtle);
}

.detect-btn:hover {
  background: var(--color-edit-shell-chip-active);
}

.primary-action-btn {
  background: var(--edit-toolbar-surface-subtle);
  border-color: var(--edit-toolbar-border-hover);
  color: var(--color-edit-accent);
}

.primary-action-btn:hover {
  background: var(--edit-toolbar-surface-hover);
}

.brush-btn {
  background: var(--edit-toolbar-surface-active);
  border-color: var(--edit-toolbar-border-active);
}

.brush-btn:hover {
  background: var(--edit-toolbar-surface-selected);
}

/* sync按钮激活状态 */
.view-controls button.active {
  background: var(--color-edit-shell-chip-active);
}
</style>
