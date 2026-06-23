<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import type { ChapterData } from '@/types/api'
import ChapterRow from './ChapterRow.vue'

defineProps<{
  chapters: ChapterData[]
  draggedChapterIndex: number | null
  dragOverChapterIndex: number | null
}>()

defineEmits<{
  (event: 'create'): void
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
  <div class="chapters-section">
    <div class="section-header">
      <h3>章节列表</h3>
      <UiButton size="sm" variant="primary" @click="$emit('create')">
        <span class="button-icon">+</span> 新建章节
      </UiButton>
    </div>
    <div v-if="chapters.length > 0" class="chapters-list">
      <ChapterRow
        v-for="(chapter, index) in chapters"
        :key="chapter.id"
        :chapter="chapter"
        :index="index"
        :is-dragging="draggedChapterIndex === index"
        :is-drag-over="dragOverChapterIndex === index && draggedChapterIndex !== index"
        @delete="$emit('delete', $event)"
        @drag-end="$emit('dragEnd')"
        @drag-leave="$emit('dragLeave')"
        @drag-over="(event, rowIndex) => $emit('dragOver', event, rowIndex)"
        @drag-start="(event, rowIndex) => $emit('dragStart', event, rowIndex)"
        @drop="(event, rowIndex) => $emit('drop', event, rowIndex)"
        @edit="$emit('edit', $event)"
        @read="$emit('read', $event)"
        @translate="$emit('translate', $event)"
      />
    </div>
    <div v-else class="empty-state-small">
      <p>暂无章节，点击上方按钮创建</p>
    </div>
  </div>
</template>

<style scoped>
.chapters-section {
  padding-top: 16px;
  border-top: 1px solid var(--color-border-muted);
}

.section-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 16px;
}

.section-header h3 {
  margin: 0;
  color: var(--color-text-default);
  font-weight: 600;
  font-size: 1.05rem;
}

.chapters-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 280px;
  padding-right: 4px;
  overflow-y: auto;
  -webkit-overflow-scrolling: touch;
}

.chapters-list::-webkit-scrollbar {
  width: 6px;
}

.chapters-list::-webkit-scrollbar-track {
  border-radius: 3px;
  background: var(--color-surface-interactive-hover);
}

.chapters-list::-webkit-scrollbar-thumb {
  border-radius: 3px;
  background: var(--color-border-muted);
}

.chapters-list::-webkit-scrollbar-thumb:hover {
  background: var(--color-text-supporting);
}

.empty-state-small {
  padding: 40px 20px;
  color: var(--color-text-supporting);
  text-align: center;
}
</style>
