<template>
  <div class="edit-toolbar">
    <div class="edit-toolbar__row edit-toolbar__row--primary">
      <div class="edit-toolbar__image-navigator">
        <UiIconButton
          variant="inverse"
          size="sm"
          class="edit-toolbar__nav-action"
          :disabled="!canGoPrevious"
          label="上一张图片"
          title="上一张图片 (A)"
          @click="$emit('go-previous-image')"
        >
          <UiIcon name="chevrons-left" size="16" />
        </UiIconButton>
        <UiButton
          variant="toolbar"
          class="edit-toolbar__image-indicator"
          aria-label="显示或隐藏缩略图"
          :aria-pressed="String(showThumbnails)"
          @click="$emit('toggle-thumbnails')"
          title="点击展开缩略图"
        >
          图 <span class="edit-toolbar__image-indicator-value">{{ currentImageIndex + 1 }}</span> / <span class="edit-toolbar__image-indicator-value">{{ imageCount }}</span>
        </UiButton>
        <UiIconButton
          variant="inverse"
          size="sm"
          class="edit-toolbar__nav-action"
          :disabled="!canGoNext"
          label="下一张图片"
          title="下一张图片 (D)"
          @click="$emit('go-next-image')"
        >
          <UiIcon name="chevrons-right" size="16" />
        </UiIconButton>
        <UiIconButton
          variant="inverse"
          size="sm"
          class="edit-toolbar__thumbnail-toggle"
          :active="showThumbnails"
          :pressed="showThumbnails"
          label="显示或隐藏缩略图"
          title="显示/隐藏缩略图"
          @click="$emit('toggle-thumbnails')"
        >
          <UiIcon name="list" size="16" />
        </UiIconButton>
      </div>

      <div class="edit-toolbar__divider"></div>

      <div class="edit-toolbar__bubble-navigator">
        <UiIconButton
          variant="inverse"
          size="sm"
          class="edit-toolbar__nav-action"
          :disabled="!hasBubbles || selectedBubbleIndex <= 0"
          label="上一个气泡"
          title="上一个气泡"
          @click="$emit('select-previous-bubble')"
        >
          <UiIcon name="chevron-left" size="16" />
        </UiIconButton>
        <span class="edit-toolbar__bubble-indicator">
          气泡 <span class="edit-toolbar__bubble-indicator-value">{{ selectedBubbleIndex >= 0 ? selectedBubbleIndex + 1 : 0 }}</span> / <span class="edit-toolbar__bubble-indicator-value">{{ bubbleCount }}</span>
        </span>
        <UiIconButton
          variant="inverse"
          size="sm"
          class="edit-toolbar__nav-action"
          :disabled="!hasBubbles || selectedBubbleIndex >= bubbleCount - 1"
          label="下一个气泡"
          title="下一个气泡"
          @click="$emit('select-next-bubble')"
        >
          <UiIcon name="chevron-right" size="16" />
        </UiIconButton>
      </div>

      <div class="edit-toolbar__divider"></div>

      <div class="edit-toolbar__view-controls">
        <UiIconButton
          variant="inverse"
          size="md"
          class="edit-toolbar__view-action edit-toolbar__view-action--layout"
          label="切换布局"
          title="切换布局：左右/上下"
          @click="$emit('toggle-layout')"
        >
          <UiIcon :name="layoutMode === 'horizontal' ? 'columns' : 'rows'" size="16" />
        </UiIconButton>
        <UiIconButton
          variant="inverse"
          size="md"
          class="edit-toolbar__view-action edit-toolbar__view-action--mode"
          label="切换视图模式"
          title="切换视图模式"
          @click="$emit('toggle-view-mode')"
        >
          <UiIcon name="image" size="16" />
        </UiIconButton>
        <UiIconButton
          variant="inverse"
          size="md"
          class="edit-toolbar__view-action edit-toolbar__view-action--sync"
          :active="syncEnabled"
          :pressed="syncEnabled"
          label="同步缩放和拖动"
          title="同步缩放/拖动"
          @click="$emit('toggle-sync')"
        >
          <UiIcon name="link" size="16" />
        </UiIconButton>
        <UiIconButton variant="inverse" size="md" class="edit-toolbar__view-action" label="适应屏幕" title="适应屏幕 (双击)" @click="$emit('fit-to-screen')">
          <UiIcon name="maximize" size="16" />
        </UiIconButton>
        <UiIconButton variant="inverse" size="md" class="edit-toolbar__view-action" label="放大" title="放大 (+)" @click="$emit('zoom-in')">
          <UiIcon name="plus" size="16" />
        </UiIconButton>
        <span class="edit-toolbar__zoom-level">{{ Math.round(scale * 100) }}%</span>
        <UiIconButton variant="inverse" size="md" class="edit-toolbar__view-action" label="缩小" title="缩小 (-)" @click="$emit('zoom-out')">
          <UiIcon name="minus" size="16" />
        </UiIconButton>
        <UiIconButton variant="inverse" size="md" class="edit-toolbar__view-action" label="原始大小" title="原始大小" @click="$emit('reset-zoom')">
          <UiIcon name="home" size="16" />
        </UiIconButton>
      </div>

      <div class="edit-toolbar__spacer"></div>

      <UiButton variant="inverse" size="sm" @click="$emit('exit-edit-mode')">退出编辑</UiButton>
    </div>

    <div class="edit-toolbar__row edit-toolbar__row--secondary">
      <div class="edit-toolbar__annotation-tools">
        <UiButton
          variant="toolbar"
          class="edit-toolbar__annotation-action edit-toolbar__annotation-action--detect"
          @click="$emit('auto-detect-bubbles')"
          title="自动检测当前图片的文本框"
        >
          <UiIcon name="scan-search" class="edit-toolbar__annotation-icon" size="14" />
          <span class="edit-toolbar__annotation-label">检测</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          class="edit-toolbar__annotation-action edit-toolbar__annotation-action--detect"
          @click="$emit('detect-all-images')"
          title="批量检测所有图片"
        >
          <UiIcon name="scan-line" class="edit-toolbar__annotation-icon" size="14" />
          <span class="edit-toolbar__annotation-label">批量检测</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          class="edit-toolbar__annotation-action edit-toolbar__annotation-action--primary"
          @click="$emit('translate-with-bubbles')"
          title="使用当前文本框翻译此图片"
        >
          <UiIcon name="languages" class="edit-toolbar__annotation-icon" size="14" />
          <span class="edit-toolbar__annotation-label">翻译</span>
        </UiButton>

        <div class="edit-toolbar__divider"></div>

        <UiButton
          variant="toolbar"
          class="edit-toolbar__annotation-action"
          :class="{ 'edit-toolbar__annotation-action--active': isDrawingMode }"
          :aria-pressed="String(isDrawingMode)"
          @click="$emit('toggle-drawing-mode')"
          title="添加气泡框（或中键拖拽绘制）"
        >
          <UiIcon name="square-plus" class="edit-toolbar__annotation-icon" size="14" />
          <span class="edit-toolbar__annotation-label">添加</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          class="edit-toolbar__annotation-action"
          :disabled="!hasSelection"
          @click="$emit('delete-selected-bubbles')"
          title="删除选中气泡框 (Delete)"
        >
          <UiIcon name="square-minus" class="edit-toolbar__annotation-icon" size="14" />
          <span class="edit-toolbar__annotation-label">删除</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          class="edit-toolbar__annotation-action"
          :class="{ 'edit-toolbar__annotation-action--loading': isRepairLoading }"
          :disabled="!hasSelection || isRepairLoading"
          @click="$emit('repair-selected-bubble')"
          title="修复选中气泡背景 (R)"
        >
          <UiIcon name="wand-sparkles" class="edit-toolbar__annotation-icon" size="14" :class="{ 'edit-toolbar__repair-icon--spinning': isRepairLoading }" />
          <span class="edit-toolbar__annotation-label">修复</span>
        </UiButton>

        <div class="edit-toolbar__divider"></div>

        <UiButton
          variant="toolbar"
          class="edit-toolbar__annotation-action edit-toolbar__annotation-action--brush"
          :class="{ 'edit-toolbar__annotation-action--active': brushMode === 'repair' }"
          :aria-pressed="String(brushMode === 'repair')"
          @click="$emit('activate-repair-brush')"
          title="修复笔刷 (按住R+左键拖拽)"
        >
          <UiIcon name="brush" class="edit-toolbar__annotation-icon" size="14" />
          <span class="edit-toolbar__annotation-label">修复笔刷</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          class="edit-toolbar__annotation-action edit-toolbar__annotation-action--brush"
          :class="{ 'edit-toolbar__annotation-action--active': brushMode === 'restore' }"
          :aria-pressed="String(brushMode === 'restore')"
          @click="$emit('activate-restore-brush')"
          title="还原笔刷 (按住U+左键拖拽)"
        >
          <UiIcon name="eraser" class="edit-toolbar__annotation-icon" size="14" />
          <span class="edit-toolbar__annotation-label">还原笔刷</span>
        </UiButton>
        <span v-if="brushMode" class="edit-toolbar__brush-size">
          笔刷: {{ brushSize }}px
        </span>

        <EditToolbarHelp />
      </div>

      <div
        v-if="brushMode"
        class="edit-toolbar__brush-cursor"
        :style="brushCursorStyle"
      ></div>

      <OverlayLayer v-if="brushMode" class="edit-toolbar__brush-mode-hint-layer" passthrough>
        <div class="edit-toolbar__brush-mode-hint">
          {{ brushMode === 'repair' ? '修复笔刷 (R)' : '还原笔刷 (U)' }} - 滚轮调整大小
        </div>
      </OverlayLayer>

      <div
        v-if="isProcessing"
        class="edit-toolbar__progress"
        :class="{ 'edit-toolbar__progress--completed': isProgressCompleted }"
      >
        <div class="edit-toolbar__progress-info">
          <span class="edit-toolbar__progress-text">{{ progressText }}</span>
          <span class="edit-toolbar__progress-count">{{ progressCurrent }}/{{ progressTotal }}</span>
        </div>
        <UiProgressBar
          class="edit-toolbar__progress-bar"
          :value="progressAriaValue"
          :max="progressAriaMax"
          label="编辑处理进度"
          tone="success"
          size="sm"
          :striped="!isProgressCompleted"
          :animated="!isProgressCompleted"
        />
      </div>

      <div class="edit-toolbar__spacer"></div>

      <div class="edit-toolbar__quick-actions">
        <UiButton variant="primary" tone="success" size="sm" @click="$emit('apply-and-next')" title="应用更改并跳转下一张 (Ctrl+Enter)">
          应用并下一张
        </UiButton>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">

import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
import { computed } from 'vue'
import EditToolbarHelp from './EditToolbarHelp.vue'

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
.edit-toolbar {
  --edit-toolbar-shell-start: var(--color-surface-inverse-panel);
  --edit-toolbar-shell-end: var(--color-surface-inverse);
  --edit-toolbar-shell-divider: var(--color-overlay-inverse-subtle);
  --edit-toolbar-shell-divider-soft: color-mix(in srgb, var(--color-overlay-inverse-subtle) 50%, transparent);
  --edit-toolbar-row-overlay: color-mix(in srgb, var(--color-overlay-scrim-subtle) 50%, transparent);
  --edit-toolbar-chip-background: color-mix(in srgb, var(--color-action-brand) 20%, transparent);
  --edit-toolbar-chip-hover-background: color-mix(in srgb, var(--color-action-brand) 40%, transparent);
  --edit-toolbar-chip-active-background: color-mix(in srgb, var(--color-action-brand) 50%, transparent);
  --edit-toolbar-control-background: color-mix(in srgb, var(--color-action-brand) 30%, transparent);
  --edit-toolbar-progress-background: var(--color-overlay-scrim-subtle);
  --edit-toolbar-status-accent: var(--color-action-success-bright);
  --edit-toolbar-progress-fill-start: var(--color-action-success-bright);
  --edit-toolbar-annotation-button-border: color-mix(in srgb, var(--color-text-inverse) 20%, transparent);
  --edit-toolbar-annotation-button-hover-border: color-mix(in srgb, var(--color-text-inverse) 30%, transparent);
  --edit-toolbar-detect-button-border: color-mix(in srgb, var(--color-action-brand) 50%, transparent);
  --edit-toolbar-translate-button-background: color-mix(in srgb, var(--color-action-success-bright) 20%, transparent);
  --edit-toolbar-translate-button-hover-background: color-mix(in srgb, var(--color-action-success-bright) 30%, transparent);
  --edit-toolbar-translate-button-border: color-mix(in srgb, var(--color-action-success-bright) 40%, transparent);
  --edit-toolbar-brush-button-background: color-mix(in srgb, var(--color-status-warning) 20%, transparent);
  --edit-toolbar-brush-button-hover-background: color-mix(in srgb, var(--color-status-warning) 30%, transparent);
  --edit-toolbar-brush-button-border: color-mix(in srgb, var(--color-status-warning) 40%, transparent);
  --edit-toolbar-image-index-text: var(--color-action-brand);
  --edit-toolbar-progress-text: color-mix(in srgb, var(--color-text-inverse) 90%, transparent);
  --edit-toolbar-progress-highlight: var(--color-status-info-bright);
  --edit-toolbar-brush-hint-background: color-mix(in srgb, var(--color-overlay-backdrop-solid) 80%, transparent);
  --edit-toolbar-brush-repair-fill: color-mix(in srgb, var(--color-status-success) 40%, transparent);
  --edit-toolbar-brush-repair-border: var(--color-status-success);
  --edit-toolbar-brush-restore-fill: color-mix(in srgb, var(--color-status-info) 40%, transparent);
  --edit-toolbar-brush-restore-border: var(--color-status-info);
  --ui-icon-button-active-background: var(--edit-toolbar-chip-active-background);
  --ui-icon-button-active-border: var(--color-border-brand-gradient);
  --ui-icon-button-active-hover-background: var(--edit-toolbar-chip-hover-background);

  flex-shrink: 0;
  background: linear-gradient(135deg, var(--edit-toolbar-shell-start) 0%, var(--edit-toolbar-shell-end) 100%);
  border-bottom: 1px solid var(--edit-toolbar-shell-divider);
}

.edit-toolbar__row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  padding: 8px 15px;
  gap: 10px;
}

.edit-toolbar__row--primary {
  border-bottom: 1px solid var(--edit-toolbar-shell-divider-soft);
}

.edit-toolbar__row--secondary {
  background: var(--edit-toolbar-row-overlay);
}

.edit-toolbar__spacer {
  flex: 1;
  min-width: 0;
}

.edit-toolbar__divider {
  width: 1px;
  height: 24px;
  background: var(--color-overlay-inverse-muted);
  margin: 0 5px;
}

.edit-toolbar__image-navigator {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.edit-toolbar__image-indicator {
  color: var(--color-text-inverse);
  font-size: 14px;
  padding: 6px 12px;
  background: var(--edit-toolbar-chip-background);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.edit-toolbar__image-indicator:hover {
  background: var(--edit-toolbar-chip-hover-background);
}

.edit-toolbar__image-indicator-value {
  font-weight: 700;
  color: var(--edit-toolbar-image-index-text);
}

.edit-toolbar__bubble-navigator {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.edit-toolbar__bubble-indicator {
  color: var(--color-text-inverse);
  font-size: 13px;
  padding: 4px 10px;
  background: var(--edit-toolbar-progress-background);
  border-radius: 6px;
}

.edit-toolbar__bubble-indicator-value {
  font-weight: 700;
  color: var(--edit-toolbar-status-accent);
}

.edit-toolbar__view-controls {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.edit-toolbar__view-controls .edit-toolbar__zoom-level {
  min-width: 50px;
  text-align: center;
  color: var(--color-text-inverse);
  font-size: 13px;
  padding: 0 8px;
}

.edit-toolbar__quick-actions {
  display: flex;
  gap: 10px;
}

.edit-toolbar__progress {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 16px;
  margin-left: 12px;
  background: var(--edit-toolbar-progress-background);
  border-radius: 20px;
  min-width: min(100%, 200px);
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

.edit-toolbar__progress-info {
  display: flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
}

.edit-toolbar__progress-text {
  font-size: 12px;
  color: var(--edit-toolbar-progress-text);
  font-weight: 500;
}

.edit-toolbar__progress-count {
  font-size: 12px;
  color: var(--edit-toolbar-status-accent);
  font-weight: 600;
  font-family: var(--font-mono);
}

.edit-toolbar__progress-bar {
  flex: 1;
  min-width: 80px;

  --ui-progress-bar-track: var(--color-overlay-inverse-soft);
  --ui-progress-bar-stripe: color-mix(in srgb, var(--color-text-inverse) 18%, transparent);
  --ui-progress-bar-height: 6px;
  --ui-progress-bar-fill: linear-gradient(90deg, var(--edit-toolbar-progress-fill-start), var(--edit-toolbar-progress-highlight));
}

.edit-toolbar__progress--completed .edit-toolbar__progress-bar {
  --ui-progress-bar-fill: var(--edit-toolbar-progress-fill-start);
}

.edit-toolbar__progress--completed .edit-toolbar__progress-text {
  color: var(--edit-toolbar-status-accent);
}

.edit-toolbar__annotation-action--loading {
  opacity: 0.7;
  cursor: wait;
  pointer-events: none;
}

.edit-toolbar__annotation-action--loading .edit-toolbar__repair-icon--spinning {
  animation: spin-repair-icon 1s linear infinite;
}

.edit-toolbar__brush-size {
  color: var(--color-text-inverse);
  font-size: 12px;
  padding: 4px 8px;
  background: var(--color-overlay-inverse-subtle);
  border-radius: 4px;
  margin-left: 8px;
}

.edit-toolbar__annotation-action--active {
  background: var(--edit-toolbar-chip-active-background);
  border-color: var(--color-border-brand-gradient);
}

.edit-toolbar__brush-cursor {
  pointer-events: none;
  transition: width 0.1s, height 0.1s;
}

.edit-toolbar__brush-mode-hint-layer {
  display: flex;
  align-items: flex-end;
  justify-content: center;
  padding-bottom: 20px;
}

.edit-toolbar__brush-mode-hint {
  padding: 8px 16px;
  background: var(--edit-toolbar-brush-hint-background);
  color: var(--color-text-inverse);
  border-radius: 6px;
  font-size: 13px;
  pointer-events: none;
}

.edit-toolbar__annotation-tools {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}

.edit-toolbar__annotation-action {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 10px;
  border: 1px solid var(--edit-toolbar-annotation-button-border);
  border-radius: 6px;
  background: var(--color-overlay-inverse-subtle);
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.edit-toolbar__annotation-action:hover {
  background: var(--color-overlay-inverse-muted);
  border-color: var(--edit-toolbar-annotation-button-hover-border);
}

.edit-toolbar__annotation-action:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.edit-toolbar__annotation-icon {
  flex-shrink: 0;
}

.edit-toolbar__annotation-label {
  white-space: nowrap;
}

.edit-toolbar__annotation-action--detect {
  background: var(--edit-toolbar-control-background);
  border-color: var(--edit-toolbar-detect-button-border);
}

.edit-toolbar__annotation-action--detect:hover {
  background: var(--edit-toolbar-chip-active-background);
}

.edit-toolbar__annotation-action--primary {
  background: var(--edit-toolbar-translate-button-background);
  border-color: var(--edit-toolbar-translate-button-border);
  color: var(--edit-toolbar-status-accent);
}

.edit-toolbar__annotation-action--primary:hover {
  background: var(--edit-toolbar-translate-button-hover-background);
}

.edit-toolbar__annotation-action--brush {
  background: var(--edit-toolbar-brush-button-background);
  border-color: var(--edit-toolbar-brush-button-border);
}

.edit-toolbar__annotation-action--brush:hover {
  background: var(--edit-toolbar-brush-button-hover-background);
}

</style>
