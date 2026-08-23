<script setup lang="ts">
import { computed } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import type { ChapterData } from '@/types/api'
import ChapterRow from './ChapterRow.vue'

const props = withDefaults(defineProps<{
  chapters: ChapterData[]
  draggedChapterIndex: number | null
  dragOverChapterIndex: number | null
  selectedChapterIds?: Set<string>
  downloadPending?: boolean
  translationPending?: boolean
  translationAllowed?: boolean
}>(), {
  selectedChapterIds: () => new Set<string>(),
  downloadPending: false,
  translationPending: false,
  translationAllowed: true,
})

const emit = defineEmits<{
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
  (event: 'select', chapterId: string, selected: boolean): void
  (event: 'selectAll', chapterIds: string[]): void
  (event: 'downloadSelected'): void
  (event: 'translateSelected'): void
}>()

const selectableChapterIds = computed(() => props.chapters
  .filter(chapter => (chapter.imageCount ?? 0) > 0)
  .map(chapter => chapter.id))
const actionPending = computed(() => props.downloadPending || props.translationPending)
const allSelected = computed(() => (
  selectableChapterIds.value.length > 0
  && selectableChapterIds.value.every(id => props.selectedChapterIds.has(id))
))

function toggleAll() {
  emit('selectAll', allSelected.value ? [] : selectableChapterIds.value)
}
</script>

<template>
  <div class="chapter-list">
    <ProductSectionHeader title="章节列表" :heading-level="3">
      <template #actions>
        <UiButton
          v-if="chapters.length"
          size="sm"
          variant="secondary"
          :disabled="selectableChapterIds.length === 0 || actionPending"
          @click="toggleAll"
        >
          {{ allSelected ? '清空选择' : '全选有内容章节' }}
        </UiButton>
        <UiButton
          v-if="chapters.length"
          size="sm"
          variant="secondary"
          :disabled="selectedChapterIds.size === 0 || actionPending"
          :loading="downloadPending"
          @click="$emit('downloadSelected')"
        >
          下载选中章节（{{ selectedChapterIds.size }}）
        </UiButton>
        <UiButton
          v-if="chapters.length && translationAllowed"
          size="sm"
          variant="primary"
          :disabled="selectedChapterIds.size === 0 || actionPending"
          :loading="translationPending"
          @click="$emit('translateSelected')"
        >
          翻译选中章节（{{ selectedChapterIds.size }}）
        </UiButton>
        <UiButton size="sm" variant="primary" @click="$emit('create')">
          <span aria-hidden="true">+</span>
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
        selectable
        :translation-allowed="translationAllowed"
        :selected="selectedChapterIds.has(chapter.id)"
        @delete="$emit('delete', $event)"
        @drag-end="$emit('dragEnd')"
        @drag-leave="$emit('dragLeave')"
        @drag-over="(event, rowIndex) => $emit('dragOver', event, rowIndex)"
        @drag-start="(event, rowIndex) => $emit('dragStart', event, rowIndex)"
        @drop="(event, rowIndex) => $emit('drop', event, rowIndex)"
        @edit="$emit('edit', $event)"
        @read="$emit('read', $event)"
        @translate="$emit('translate', $event)"
        @select="(chapterId, selected) => $emit('select', chapterId, selected)"
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
  overflow-x: hidden;
  -webkit-overflow-scrolling: touch;
}

.chapter-list__empty-state {
  --product-status-banner-align-items: center;
  --product-status-banner-justify-content: center;
  --product-status-banner-padding: 40px 20px;
  --product-status-banner-border: 0;
  --product-status-banner-background: transparent;
  --product-status-banner-icon-display: none;
  --product-status-banner-text-align: center;
}
</style>
