<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
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
  <div class="chapter-list">
    <ProductSectionHeader title="章节列表" icon-name="book-open">
      <template #actions>
        <UiButton size="sm" variant="primary" @click="$emit('create')">
          <UiIcon name="plus" size="14" />
          <span>新建章节</span>
        </UiButton>
      </template>
    </ProductSectionHeader>
    <ProductScrollStack
      v-if="chapters.length > 0"
      class="chapter-list__list"
      aria-label="章节列表"
      gap="sm"
      padding="none"
      role="region"
    >
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
    </ProductScrollStack>
    <ProductStatusBanner
      v-else
      class="chapter-list__empty-state"
      tone="neutral"
      icon-name="book-open"
      role="note"
    >
      暂无章节，点击上方按钮创建
    </ProductStatusBanner>
  </div>
</template>

<style scoped>
.chapter-list {
  padding-top: 16px;
  border-top: 1px solid var(--color-border-muted);
}

.chapter-list__list {
  max-block-size: 280px;
  padding-right: 4px;
  -webkit-overflow-scrolling: touch;
}

.chapter-list__empty-state {
  align-items: center;
}
</style>
