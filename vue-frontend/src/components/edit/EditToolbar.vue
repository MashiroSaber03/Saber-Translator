<template>
  <div class="edit-toolbar" @pointerup="handlePointerUp">
    <div class="edit-toolbar__row edit-toolbar__row--primary">
      <div class="edit-toolbar__image-navigator">
        <UiIconButton
          variant="inverse"
          size="sm"
          class="edit-toolbar__nav-action"
          :disabled="isBusy || !!brushMode || !canGoPrevious"
          label="上一张图片"
          title="上一张图片 (A)"
          @click="$emit('go-previous-image')"
        >
          <span class="edit-toolbar__text-glyph edit-toolbar__text-glyph--compact" aria-hidden="true">◀◀</span>
        </UiIconButton>
        <UiButton
          variant="toolbar"
          class="edit-toolbar__image-indicator"
          aria-label="显示或隐藏缩略图"
          :aria-pressed="showThumbnails"
          @click="$emit('toggle-thumbnails')"
          title="点击展开缩略图"
        >
          图 <span class="edit-toolbar__image-indicator-value">{{ currentImageIndex + 1 }}</span> / <span class="edit-toolbar__image-indicator-value">{{ imageCount }}</span>
        </UiButton>
        <UiIconButton
          variant="inverse"
          size="sm"
          class="edit-toolbar__nav-action"
          :disabled="isBusy || !!brushMode || !canGoNext"
          label="下一张图片"
          title="下一张图片 (D)"
          @click="$emit('go-next-image')"
        >
          <span class="edit-toolbar__text-glyph edit-toolbar__text-glyph--compact" aria-hidden="true">▶▶</span>
        </UiIconButton>
      </div>

      <div class="edit-toolbar__divider"></div>

      <div class="edit-toolbar__bubble-navigator">
        <UiIconButton
          variant="inverse"
          size="sm"
          class="edit-toolbar__nav-action"
          :disabled="isBusy || !hasBubbles || selectedBubbleIndex <= 0"
          label="上一个气泡"
          title="上一个气泡"
          @click="$emit('select-previous-bubble')"
        >
          <span class="edit-toolbar__text-glyph" aria-hidden="true">◀</span>
        </UiIconButton>
        <span class="edit-toolbar__bubble-indicator">
          气泡 <span class="edit-toolbar__bubble-indicator-value">{{ selectedBubbleIndex >= 0 ? selectedBubbleIndex + 1 : 0 }}</span> / <span class="edit-toolbar__bubble-indicator-value">{{ bubbleCount }}</span>
        </span>
        <UiIconButton
          variant="inverse"
          size="sm"
          class="edit-toolbar__nav-action"
          :disabled="isBusy || !hasBubbles || selectedBubbleIndex >= bubbleCount - 1"
          label="下一个气泡"
          title="下一个气泡"
          @click="$emit('select-next-bubble')"
        >
          <span class="edit-toolbar__text-glyph" aria-hidden="true">▶</span>
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
          <span class="edit-toolbar__text-glyph edit-toolbar__text-glyph--view-mode" aria-hidden="true">⧉</span>
        </UiIconButton>
        <UiButton
          variant="toolbar"
          class="edit-toolbar__annotation-action"
          :disabled="!canPreviewClean"
          :aria-pressed="!!showCleanPreview"
          :title="!canPreviewClean ? '尚未生成修复图，请先进行背景修复' : showCleanPreview ? '返回查看译图' : '查看不含译文的修复图'"
          @click="$emit('toggle-clean-preview')"
        >
          <span class="edit-toolbar__annotation-label">仅看修复</span>
        </UiButton>
        <label class="edit-toolbar__source-overlay" title="0% 隐藏原文，100% 显示原文；译文显示在原文上方">
          <span>原文</span>
          <UiInput
            class="edit-toolbar__source-opacity"
            type="range"
            min="0"
            max="100"
            step="1"
            aria-label="原文透明度"
            :aria-valuetext="`${originalTextOpacity ?? 0}% 原文透明度`"
            :model-value="originalTextOpacity ?? 0"
            :disabled="!canOverlaySource"
            @update:model-value="emit('update-original-text-opacity', Number($event))"
          />
          <span class="edit-toolbar__source-opacity-value">{{ originalTextOpacity ?? 0 }}%</span>
        </label>
        <label class="edit-toolbar__source-overlay" title="0% 隐藏译文，100% 显示译文；背景保持不变">
          <span>译文</span>
          <UiInput
            class="edit-toolbar__source-opacity"
            type="range"
            min="0"
            max="100"
            step="1"
            aria-label="译文透明度"
            :aria-valuetext="`${translatedTextOpacity ?? 100}% 译文透明度`"
            :model-value="translatedTextOpacity ?? 100"
            :disabled="!canFadeTranslated"
            @update:model-value="emit('update-translated-text-opacity', Number($event))"
          />
          <span class="edit-toolbar__source-opacity-value">{{ translatedTextOpacity ?? 100 }}%</span>
        </label>
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
          <span class="edit-toolbar__text-glyph edit-toolbar__text-glyph--sync" aria-hidden="true">🔗</span>
        </UiIconButton>
        <UiIconButton variant="inverse" size="md" class="edit-toolbar__view-action" label="适应屏幕" title="适应屏幕 (双击)" @click="$emit('fit-to-screen')">
          <span class="edit-toolbar__text-glyph edit-toolbar__text-glyph--fit" aria-hidden="true">⛶</span>
        </UiIconButton>
        <UiIconButton variant="inverse" size="md" class="edit-toolbar__view-action" label="放大" title="放大 (+)" @click="$emit('zoom-in')">
          <UiIcon name="plus" size="16" />
        </UiIconButton>
        <span class="edit-toolbar__zoom-level">{{ Math.round(scale * 100) }}%</span>
        <UiIconButton variant="inverse" size="md" class="edit-toolbar__view-action" label="缩小" title="缩小 (-)" @click="$emit('zoom-out')">
          <UiIcon name="minus" size="16" />
        </UiIconButton>
        <UiIconButton variant="inverse" size="md" class="edit-toolbar__view-action" label="原始大小" title="原始大小" @click="$emit('reset-zoom')">
          <span class="edit-toolbar__text-glyph edit-toolbar__text-glyph--reset" aria-hidden="true">1:1</span>
        </UiIconButton>
      </div>

      <div class="edit-toolbar__spacer"></div>

      <UiButton
        variant="inverse"
        size="sm"
        :disabled="isBusy || !!brushMode"
        @click="$emit('exit-edit-mode')"
      >
        退出编辑
      </UiButton>
    </div>

    <slot v-if="maskMode" name="mask-tools" />
    <div v-else class="edit-toolbar__row edit-toolbar__row--secondary">
      <div class="edit-toolbar__annotation-tools">
        <UiButton variant="toolbar" class="edit-toolbar__annotation-action" title="查看和编辑背景修复范围" :disabled="isBusy" @click="$emit('toggle-mask-mode')">
          <UiIcon name="target" class="edit-toolbar__annotation-icon" size="14" />
          <span class="edit-toolbar__annotation-label">修复范围</span>
        </UiButton>
        <div class="edit-toolbar__divider"></div>
        <UiButton
          variant="toolbar"
          class="edit-toolbar__annotation-action edit-toolbar__annotation-action--detect"
          :disabled="isBusy"
          @click="$emit('auto-detect-bubbles')"
          title="自动检测当前图片的文本框"
        >
          <UiIcon name="scan-search" class="edit-toolbar__annotation-icon" size="14" />
          <span class="edit-toolbar__annotation-label">检测</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          class="edit-toolbar__annotation-action edit-toolbar__annotation-action--detect"
          :disabled="isBusy"
          @click="$emit('detect-all-images')"
          title="批量检测所有图片"
        >
          <UiIcon name="scan-line" class="edit-toolbar__annotation-icon" size="14" />
          <span class="edit-toolbar__annotation-label">批量检测</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          class="edit-toolbar__annotation-action edit-toolbar__annotation-action--primary"
          :disabled="isBusy"
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
          :aria-pressed="isDrawingMode"
          :disabled="isBusy"
          @click="$emit('toggle-drawing-mode')"
          title="添加气泡框（或中键拖拽绘制）"
        >
          <UiIcon name="square-plus" class="edit-toolbar__annotation-icon" size="14" />
          <span class="edit-toolbar__annotation-label">添加</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          class="edit-toolbar__annotation-action"
          :disabled="isBusy || !hasSelection"
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
          :disabled="isBusy || !hasSelection || isRepairLoading"
          @click="$emit('repair-selected-bubble')"
          title="修复选中气泡背景"
        >
          <UiIcon name="wand-sparkles" class="edit-toolbar__annotation-icon" size="14" :class="{ 'edit-toolbar__repair-icon--spinning': isRepairLoading }" />
          <span class="edit-toolbar__annotation-label">修复</span>
        </UiButton>

        <div class="edit-toolbar__divider"></div>

        <UiIconButton
          variant="inverse"
          size="sm"
          class="edit-toolbar__annotation-action edit-toolbar__annotation-action--brush"
          :pressed="brushMode === 'repair'"
          label="修复笔刷"
          :disabled="isBusy"
          @click="$emit('activate-repair-brush')"
          title="修复笔刷 (按住R+左键拖拽)"
        >
          <UiIcon name="brush" class="edit-toolbar__annotation-icon" size="16" />
        </UiIconButton>
        <UiIconButton
          variant="inverse"
          size="sm"
          class="edit-toolbar__annotation-action edit-toolbar__annotation-action--brush"
          :pressed="brushMode === 'restore'"
          label="还原笔刷"
          :disabled="isBusy"
          @click="$emit('activate-restore-brush')"
          title="还原笔刷 (按住U+左键拖拽)"
        >
          <UiIcon name="eraser" class="edit-toolbar__annotation-icon" size="16" />
        </UiIconButton>
        <span v-if="brushMode" class="edit-toolbar__brush-size" title="再次点击笔刷或按 Esc 退出">
          {{ brushSize }}px · 滚轮调大小 · Esc 退出
        </span>

        <EditToolbarHelp />
      </div>

      <div class="edit-toolbar__spacer"></div>

      <div class="edit-toolbar__quick-actions">
        <UiButton
          variant="primary"
          tone="success"
          size="sm"
          :disabled="isBusy || !!brushMode"
          :title="canGoNext ? '应用更改并跳转下一张 (Ctrl+Enter)' : '应用当前图片的更改 (Ctrl+Enter)'"
          @click="$emit('apply-and-next')"
        >
          {{ canGoNext ? '应用并下一张' : '应用更改' }}
        </UiButton>
      </div>
    </div>
    <div v-if="brushMode" class="edit-toolbar__brush-cursor" :style="brushCursorStyle"></div>
  </div>
</template>

<script setup lang="ts">

import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
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
  isBusy: boolean
  isRepairLoading?: boolean
  maskMode?: boolean
  canPreviewClean?: boolean
  showCleanPreview?: boolean
  originalTextOpacity?: number
  canOverlaySource?: boolean
  translatedTextOpacity?: number
  canFadeTranslated?: boolean
}>()

const emit = defineEmits<{
  (e: 'go-previous-image'): void
  (e: 'go-next-image'): void
  (e: 'toggle-thumbnails'): void
  (e: 'select-previous-bubble'): void
  (e: 'select-next-bubble'): void
  (e: 'toggle-layout'): void
  (e: 'toggle-view-mode'): void
  (e: 'toggle-clean-preview'): void
  (e: 'update-original-text-opacity', value: number): void
  (e: 'update-translated-text-opacity', value: number): void
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
  (e: 'pointer-action-complete'): void
  (e: 'toggle-mask-mode'): void
}>()

function handlePointerUp(event: PointerEvent): void {
  if (event.target instanceof Element && event.target.closest('input:not([type="range"]), [role="combobox"], [role="listbox"]')) return
  emit('pointer-action-complete')
}

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
  --edit-toolbar-indicator-background: var(--color-overlay-scrim-subtle);
  --edit-toolbar-status-accent: var(--color-action-success-bright);
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
  background: var(--edit-toolbar-indicator-background);
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

.edit-toolbar__source-overlay {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  border: 1px solid var(--edit-toolbar-annotation-button-border);
  border-radius: 6px;
  color: var(--color-text-inverse);
  font-size: 12px;
  white-space: nowrap;
}

.edit-toolbar__source-opacity {
  width: 72px;
  accent-color: var(--color-action-brand);
}

.edit-toolbar__source-opacity-value {
  min-width: 4ch;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.edit-toolbar__text-glyph {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  line-height: 1;
}

.edit-toolbar__text-glyph--compact {
  font-size: 10px;
  letter-spacing: -2px;
}

.edit-toolbar__text-glyph--fit {
  font-size: 16px;
}

.edit-toolbar__text-glyph--view-mode {
  font-size: 18px;
}

.edit-toolbar__text-glyph--sync {
  font-size: 12px;
}

.edit-toolbar__text-glyph--reset {
  font-size: 11px;
  white-space: nowrap;
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

.edit-toolbar__brush-cursor {
  pointer-events: none;
  transition: width 0.1s, height 0.1s;
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
  padding: 0;
  background: var(--edit-toolbar-brush-button-background);
  border-color: var(--edit-toolbar-brush-button-border);
}

.edit-toolbar__annotation-action--brush:hover {
  background: var(--edit-toolbar-brush-button-hover-background);
}

.edit-toolbar__annotation-action[aria-pressed='true'] {
  background: var(--edit-toolbar-chip-active-background);
  border-color: var(--color-border-brand-gradient);
}

</style>
