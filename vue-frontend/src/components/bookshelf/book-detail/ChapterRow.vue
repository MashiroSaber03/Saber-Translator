<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import type { ChapterData } from '@/types/api'

defineProps<{
  chapter: ChapterData
  index: number
  isDragging: boolean
  isDragOver: boolean
}>()

defineEmits<{
  (event: 'dragStart', dragEvent: DragEvent, index: number): void
  (event: 'dragOver', dragEvent: DragEvent, index: number): void
  (event: 'dragLeave'): void
  (event: 'drop', dragEvent: DragEvent, index: number): void
  (event: 'dragEnd'): void
  (event: 'translate', chapterId: string): void
  (event: 'read', chapterId: string): void
  (event: 'edit', chapterId: string): void
  (event: 'delete', chapterId: string): void
}>()
</script>

<template>
  <div
    class="chapter-item"
    :class="{
      dragging: isDragging,
      'drag-over': isDragOver
    }"
    draggable="true"
    @dragstart="$emit('dragStart', $event, index)"
    @dragover="$emit('dragOver', $event, index)"
    @dragleave="$emit('dragLeave')"
    @drop="$emit('drop', $event, index)"
    @dragend="$emit('dragEnd')"
  >
    <div class="chapter-drag-handle" title="拖拽排序">⋮⋮</div>
    <div class="chapter-info">
      <span class="chapter-order">#{{ index + 1 }}</span>
      <span class="chapter-title">{{ chapter.title }}</span>
      <span class="chapter-meta">{{ chapter.image_count || chapter.imageCount || 0 }} 张图片</span>
    </div>
    <div class="chapter-actions">
      <UiButton
        variant="toolbar"
        class="chapter-action-btn chapter-enter-btn"
        @click="$emit('translate', chapter.id)"
      >
        进入翻译
      </UiButton>
      <UiButton
        variant="toolbar"
        class="chapter-action-btn chapter-read-btn"
        :disabled="(chapter.image_count || chapter.imageCount || 0) === 0"
        @click="$emit('read', chapter.id)"
      >
        进入阅读
      </UiButton>
      <UiButton
        variant="toolbar"
        class="chapter-action-btn"
        @click="$emit('edit', chapter.id)"
      >
        编辑
      </UiButton>
      <UiButton
        variant="danger"
        class="chapter-action-btn"
        @click="$emit('delete', chapter.id)"
      >
        删除
      </UiButton>
    </div>
  </div>
</template>

<style scoped>
.chapter-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 16px;
  border-radius: 8px;
  background: var(--color-surface-interactive-hover);
  transition: all 0.2s ease;
}

.chapter-item:hover {
  background: var(--color-border-muted);
}

.chapter-info {
  display: flex;
  flex: 1;
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.chapter-order {
  flex-shrink: 0;
  min-width: 32px;
  color: var(--color-text-supporting);
  font-size: 0.8rem;
}

.chapter-title {
  overflow: hidden;
  color: var(--color-text-default);
  font-weight: 500;
  font-size: 0.9rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chapter-meta {
  color: var(--color-text-supporting);
  font-size: 0.75rem;
}

.chapter-actions {
  display: flex;
  flex-shrink: 0;
  gap: 6px;
  opacity: 1;
}

.chapter-action-btn {
  padding: 6px 10px;
  border: none;
  border-radius: 4px;
  background: none;
  color: var(--color-text-supporting);
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s;
}

.chapter-action-btn:hover {
  background: var(--color-surface-card);
  color: var(--color-text-default);
}

.chapter-action-btn.danger:hover {
  color: var(--color-text-danger);
}

.chapter-enter-btn {
  background: linear-gradient(135deg, var(--color-surface-brand-gradient-start) 0%, var(--color-surface-brand-gradient-end) 100%);
  color: var(--color-text-inverse);
  font-weight: 500;
}

.chapter-enter-btn:hover {
  background: linear-gradient(135deg, var(--book-detail-modal-surface-base) 0%, var(--book-detail-modal-surface-raised) 100%);
  color: var(--color-text-inverse);
  transform: scale(1.02);
  box-shadow: 0 4px 12px var(--book-detail-modal-shadow-raised);
}

.chapter-read-btn {
  background: linear-gradient(135deg, var(--color-surface-success-gradient-start) 0%, var(--color-surface-success-gradient-end) 100%);
  color: var(--color-text-inverse);
  font-weight: 500;
}

.chapter-read-btn:disabled {
  background: var(--color-border-muted);
  color: var(--color-text-supporting);
  opacity: 0.6;
  cursor: not-allowed;
}

.chapter-read-btn:hover:not(:disabled) {
  background: linear-gradient(135deg, var(--book-detail-modal-surface-muted) 0%, var(--book-detail-modal-surface-subtle) 100%);
  color: var(--color-text-inverse);
  transform: scale(1.02);
  box-shadow: 0 4px 12px var(--book-detail-modal-shadow-floating);
}
</style>
