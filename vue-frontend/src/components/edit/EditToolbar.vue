<template>
  <div class="edit-toolbar-wrapper">
    <div class="edit-toolbar toolbar-row-1">
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
        <UiButton
          variant="toolbar"
          class="image-indicator"
          aria-label="显示或隐藏缩略图"
          @click="$emit('toggle-thumbnails')"
          title="点击展开缩略图"
        >
          图 <span>{{ currentImageIndex + 1 }}</span> / <span>{{ imageCount }}</span>
        </UiButton>
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

      <div class="view-controls">
        <UiButton
          variant="toolbar"
          class="view-control-btn layout-toggle-btn"
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
          class="view-control-btn view-mode-btn"
          @click="$emit('toggle-view-mode')"
          title="切换视图模式"
        >
          <span class="dual-icon">⧉</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          class="view-control-btn sync-toggle-btn"
          :class="{ active: syncEnabled }"
          @click="$emit('toggle-sync')"
          title="同步缩放/拖动"
        >
          🔗
        </UiButton>
        <UiButton variant="toolbar" class="view-control-btn" @click="$emit('fit-to-screen')" title="适应屏幕 (双击)">⛶</UiButton>
        <UiButton variant="toolbar" class="view-control-btn" @click="$emit('zoom-in')" title="放大 (+)">+</UiButton>
        <span id="zoomLevel" class="zoom-level">{{ Math.round(scale * 100) }}%</span>
        <UiButton variant="toolbar" class="view-control-btn" @click="$emit('zoom-out')" title="缩小 (-)">−</UiButton>
        <UiButton variant="toolbar" class="view-control-btn" @click="$emit('reset-zoom')" title="原始大小">1:1</UiButton>
      </div>

      <div class="toolbar-spacer"></div>

      <UiButton variant="toolbar" class="action-secondary" @click="$emit('exit-edit-mode')">退出编辑</UiButton>
    </div>

    <div class="edit-toolbar toolbar-row-2">
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

      <div
        v-if="brushMode"
        class="brush-cursor"
        :style="brushCursorStyle"
      ></div>

      <OverlayLayer v-if="brushMode" class="brush-mode-hint-layer" passthrough>
        <div class="brush-mode-hint">
          {{ brushMode === 'repair' ? '修复笔刷 (R)' : '还原笔刷 (U)' }} - 滚轮调整大小
        </div>
      </OverlayLayer>

      <div
        v-if="isProcessing"
        class="edit-progress-container"
        :class="{ completed: isProgressCompleted }"
      >
        <div class="edit-progress-info">
          <span class="edit-progress-text">{{ progressText }}</span>
          <span class="edit-progress-count">{{ progressCurrent }}/{{ progressTotal }}</span>
        </div>
        <div
          class="edit-progress-bar"
          role="progressbar"
          aria-label="编辑处理进度"
          aria-valuemin="0"
          :aria-valuemax="progressAriaMax"
          :aria-valuenow="progressAriaValue"
        >
          <div
            class="edit-progress-fill"
            :class="{ animating: !isProgressCompleted }"
            :style="{ width: progressPercent + '%' }"
          ></div>
        </div>
      </div>

      <div class="toolbar-spacer"></div>

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
import { computed } from 'vue'

const props = defineProps<{
  currentImageIndex: number
  imageCount: number
  canGoPrevious: boolean
  canGoNext: boolean
  showThumbnails: boolean
  hasBubbles: boolean
  selectedBubbleIndex: number
  bubbleCount: number
  layoutMode: 'horizontal' | 'vertical'
  syncEnabled: boolean
  scale: number
  isDrawingMode: boolean
  hasSelection: boolean
  brushMode: 'repair' | 'restore' | null
  brushSize: number
  mouseX: number
  mouseY: number
  isProcessing: boolean
  progressText: string
  progressCurrent: number
  progressTotal: number
  isRepairLoading?: boolean
}>()

defineEmits<{
  (e: 'go-previous-image'): void
  (e: 'go-next-image'): void
  (e: 'toggle-thumbnails'): void
  (e: 'select-previous-bubble'): void
  (e: 'select-next-bubble'): void
  (e: 'toggle-layout'): void
  (e: 'toggle-view-mode'): void
  (e: 'toggle-sync'): void
  (e: 'fit-to-screen'): void
  (e: 'zoom-in'): void
  (e: 'zoom-out'): void
  (e: 'reset-zoom'): void
  (e: 'exit-edit-mode'): void
  (e: 'auto-detect-bubbles'): void
  (e: 'detect-all-images'): void
  (e: 'translate-with-bubbles'): void
  (e: 'toggle-drawing-mode'): void
  (e: 'delete-selected-bubbles'): void
  (e: 'repair-selected-bubble'): void
  (e: 'activate-repair-brush'): void
  (e: 'activate-restore-brush'): void
  (e: 'apply-and-next'): void
}>()

const progressAriaMax = computed(() => Math.max(0, props.progressTotal))
const progressAriaValue = computed(() => Math.max(0, Math.min(props.progressCurrent, progressAriaMax.value)))

const progressPercent = computed(() => {
  if (progressAriaMax.value === 0) return 0
  const percent = Math.round((progressAriaValue.value / progressAriaMax.value) * 100)
  return Math.max(0, Math.min(100, percent))
})

const isProgressCompleted = computed(() => {
  return props.progressTotal > 0 && props.progressCurrent >= props.progressTotal
})

const brushCursorStyle = computed(() => {
  const color = props.brushMode === 'repair'
    ? { fill: 'var(--edit-toolbar-brush-repair-fill)', border: 'var(--edit-toolbar-brush-repair-border)' }
    : { fill: 'var(--edit-toolbar-brush-restore-fill)', border: 'var(--edit-toolbar-brush-restore-border)' }

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
    zIndex: 'var(--z-toast)',
    transform: 'translate(-50%, -50%)',
    display: props.brushMode ? 'block' : 'none'
  }
})
</script>

<style scoped>
.edit-toolbar-wrapper {
  --edit-toolbar-shell-start: #16213e;
  --edit-toolbar-shell-end: #1a1a2e;
  --edit-toolbar-shell-divider: rgba(255, 255, 255, .1);
  --edit-toolbar-shell-divider-soft: rgba(255, 255, 255, .05);
  --edit-toolbar-row-overlay: rgba(0, 0, 0, .15);
  --edit-toolbar-chip-background: rgba(102, 126, 234, .2);
  --edit-toolbar-chip-hover-background: rgba(102, 126, 234, .4);
  --edit-toolbar-chip-active-background: rgba(102, 126, 234, .5);
  --edit-toolbar-control-background: rgba(102, 126, 234, .3);
  --edit-toolbar-progress-background: rgba(0, 0, 0, .3);
  --edit-toolbar-accent: #0f8;
  --edit-toolbar-primary-action-start: #0f8;
  --edit-toolbar-primary-action-end: #00cc6a;
  --edit-toolbar-primary-action-text: #1a1a2e;
  --edit-toolbar-primary-action-shadow: rgba(0, 255, 136, .3);
  --edit-toolbar-secondary-action-border: rgba(255, 255, 255, .3);
  --edit-toolbar-secondary-action-border-hover: rgba(255, 255, 255, .5);
  --edit-toolbar-help-border: #cfd6e4;
  --edit-toolbar-help-border-focus: #5b73f2;
  --edit-toolbar-button-border-default: #e5e7eb;
  --edit-toolbar-button-border-strong: rgba(255, 255, 255, .2);
  --edit-toolbar-button-border-muted: rgba(255, 255, 255, .3);
  --edit-toolbar-button-border-subtle: rgba(102, 126, 234, .5);
  --edit-toolbar-button-border-hover: rgba(0, 255, 136, .4);
  --edit-toolbar-button-border-active: rgba(255, 193, 7, .4);
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
  --edit-toolbar-brush-repair-fill: rgba(76, 175, 80, .4);
  --edit-toolbar-brush-repair-border: #4caf50;
  --edit-toolbar-brush-restore-fill: rgba(33, 150, 243, .4);
  --edit-toolbar-brush-restore-border: #2196f3;

  flex-shrink: 0;
  background: linear-gradient(135deg, var(--edit-toolbar-shell-start) 0%, var(--edit-toolbar-shell-end) 100%);
  border-bottom: 1px solid var(--edit-toolbar-shell-divider);
}

.edit-toolbar {
  display: flex;
  align-items: center;
  padding: 8px 15px;
  gap: 10px;
}

.toolbar-row-1 {
  border-bottom: 1px solid var(--edit-toolbar-shell-divider-soft);
}

.toolbar-row-2 {
  background: var(--edit-toolbar-row-overlay);
}

.toolbar-spacer {
  flex: 1;
}

.toolbar-divider {
  width: 1px;
  height: 24px;
  background: var(--color-overlay-inverse-muted);
  margin: 0 5px;
}

.image-navigator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.image-indicator {
  color: var(--color-text-inverse);
  font-size: 14px;
  padding: 6px 12px;
  background: var(--edit-toolbar-chip-background);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.image-indicator:hover {
  background: var(--edit-toolbar-chip-hover-background);
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
  background: var(--color-overlay-inverse-subtle);
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s;
}

.thumb-toggle-btn:hover {
  background: var(--color-overlay-inverse-muted);
}

.thumb-toggle-btn.active {
  background: var(--edit-toolbar-chip-active-background);
}

.bubble-navigator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.bubble-indicator {
  color: var(--color-text-inverse);
  font-size: 13px;
  padding: 4px 10px;
  background: var(--edit-toolbar-progress-background);
  border-radius: 6px;
}

.bubble-indicator span {
  font-weight: bold;
  color: var(--edit-toolbar-accent);
}

.view-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

.view-control-btn {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 6px;
  background: var(--color-overlay-inverse-subtle);
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s;
}

.view-control-btn:hover {
  background: var(--color-overlay-inverse-muted);
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

.sync-toggle-btn {
  font-size: 12px;
}

.quick-actions {
  display: flex;
  gap: 10px;
}

.action-primary {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  background: linear-gradient(135deg, var(--edit-toolbar-primary-action-start) 0%, var(--edit-toolbar-primary-action-end) 100%);
  color: var(--edit-toolbar-primary-action-text);
  font-weight: 600;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.action-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px var(--edit-toolbar-primary-action-shadow);
}

.action-secondary {
  padding: 8px 16px;
  border: 1px solid var(--edit-toolbar-secondary-action-border);
  border-radius: 6px;
  background: transparent;
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.action-secondary:hover {
  background: var(--color-overlay-inverse-subtle);
  border-color: var(--edit-toolbar-secondary-action-border-hover);
}

.image-navigator .nav-btn,
.bubble-navigator .nav-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background: var(--edit-toolbar-control-background);
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
  background: var(--edit-toolbar-chip-active-background);
}

.edit-progress-container {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 16px;
  margin-left: 12px;
  background: var(--edit-toolbar-progress-background);
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
  color: var(--edit-toolbar-accent);
  font-weight: 600;
  font-family: var(--font-mono);
}

.edit-progress-bar {
  flex: 1;
  height: 6px;
  background: var(--color-overlay-inverse-soft);
  border-radius: 3px;
  overflow: hidden;
  min-width: 80px;
}

.edit-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--edit-toolbar-primary-action-start), var(--edit-toolbar-surface-base));
  border-radius: 3px;
  transition: width 0.3s ease;
  box-shadow: 0 0 8px var(--edit-toolbar-shadow-default);
}

.edit-progress-fill.animating {
  background: linear-gradient(90deg, var(--edit-toolbar-primary-action-start), var(--edit-toolbar-surface-base), var(--edit-toolbar-primary-action-start));
  background-size: 200% 100%;
  animation: progressShine 1.5s ease-in-out infinite;
}

@keyframes progressShine {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

.edit-progress-container.completed .edit-progress-fill {
  background: var(--edit-toolbar-primary-action-start);
  animation: none;
}

.edit-progress-container.completed .edit-progress-text {
  color: var(--edit-toolbar-accent);
}

.annotation-btn.is-loading {
  opacity: 0.7;
  cursor: wait;
  pointer-events: none;
}

.annotation-btn.is-loading .spin-icon {
  animation: spin-repair-icon 1s linear infinite;
}

.brush-size-indicator {
  color: var(--color-text-inverse);
  font-size: 12px;
  padding: 4px 8px;
  background: var(--color-overlay-inverse-subtle);
  border-radius: 4px;
  margin-left: 8px;
}

.annotation-btn.active,
.brush-btn.active {
  background: var(--edit-toolbar-chip-active-background);
  border-color: var(--color-border-brand-gradient);
}

.brush-cursor {
  pointer-events: none;
  transition: width 0.1s, height 0.1s;
}

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
  border: 1px solid var(--edit-toolbar-help-border);
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
  border-color: var(--edit-toolbar-help-border-focus);
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
  border-bottom: 1px solid var(--edit-toolbar-button-border-default);
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
  border: 1px solid var(--edit-toolbar-button-border-strong);
  border-radius: 6px;
  background: var(--color-overlay-inverse-subtle);
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.annotation-btn:hover {
  background: var(--color-overlay-inverse-muted);
  border-color: var(--edit-toolbar-button-border-muted);
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
  background: var(--edit-toolbar-control-background);
  border-color: var(--edit-toolbar-button-border-subtle);
}

.detect-btn:hover {
  background: var(--edit-toolbar-chip-active-background);
}

.primary-action-btn {
  background: var(--edit-toolbar-surface-subtle);
  border-color: var(--edit-toolbar-button-border-hover);
  color: var(--edit-toolbar-accent);
}

.primary-action-btn:hover {
  background: var(--edit-toolbar-surface-hover);
}

.brush-btn {
  background: var(--edit-toolbar-surface-active);
  border-color: var(--edit-toolbar-button-border-active);
}

.brush-btn:hover {
  background: var(--edit-toolbar-surface-selected);
}

.view-control-btn.active {
  background: var(--edit-toolbar-chip-active-background);
}
</style>
