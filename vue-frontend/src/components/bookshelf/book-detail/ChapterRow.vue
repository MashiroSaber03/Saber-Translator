<script setup lang="ts">
import { computed } from 'vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import type { ChapterData } from '@/types/api'
import TaskStatusBadge from '@/components/task-center/TaskStatusBadge.vue'
import { useTaskCenterStore } from '@/stores/taskCenterStore'

const props = withDefaults(defineProps<{
  chapter: ChapterData
  index: number
  isDragging: boolean
  isDragOver: boolean
  selectable?: boolean
  selected?: boolean
}>(), {
  selectable: false,
  selected: false,
})

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
  (event: 'select', chapterId: string, selected: boolean): void
}>()

const taskCenterStore = useTaskCenterStore()
const imageCount = computed(() => props.chapter.imageCount ?? 0)
const hasActiveTranslation = computed(() => {
  const live = taskCenterStore.queue.some(job => (
    job.chapterId === props.chapter.id
    && job.kind === 'translation'
    && ['queued', 'running', 'pausing', 'paused', 'cancelling', 'interrupted'].includes(job.status)
  ))
  if (live) return true
  const summary = props.chapter.jobStatusSummary || {}
  return ['queued', 'running', 'pausing', 'paused', 'cancelling', 'interrupted']
    .some(status => (summary[status as keyof typeof summary] || 0) > 0)
})
const canSelect = computed(() => imageCount.value > 0 && !hasActiveTranslation.value)
</script>

<template>
  <ProductRecordCard
    class="chapter-row"
    :class="{
      'chapter-row--dragging': isDragging,
      'chapter-row--drag-over': isDragOver
    }"
    :aria-label="`章节 ${chapter.title}，拖拽可调整排序`"
    :aria-grabbed="isDragging ? 'true' : 'false'"
    draggable="true"
    @dragstart="$emit('dragStart', $event, index)"
    @dragover="$emit('dragOver', $event, index)"
    @dragleave="$emit('dragLeave')"
    @drop="$emit('drop', $event, index)"
    @dragend="$emit('dragEnd')"
  >
    <div class="chapter-row__content">
      <div class="chapter-row__drag-handle" title="拖拽排序" aria-hidden="true">
        <UiIcon name="grip-vertical" size="18" />
      </div>
      <UiCheckbox
        v-if="selectable"
        :model-value="selected"
        :disabled="!canSelect"
        :aria-label="`选择章节：${chapter.title}`"
        @change="$emit('select', chapter.id, $event)"
      />
      <div class="chapter-row__info">
        <span class="chapter-row__order">#{{ index + 1 }}</span>
        <span class="chapter-row__title">{{ chapter.title }}</span>
        <span class="chapter-row__meta">{{ imageCount }} 张图片</span>
        <TaskStatusBadge
          :chapter-id="chapter.id"
          :summary="chapter.jobStatusSummary"
        />
      </div>
      <ProductActionRow
        class="chapter-row__actions"
        :aria-label="`${chapter.title} 章节操作`"
        justify="end"
      >
        <UiButton
          variant="primary"
          size="xs"
          @click="$emit('translate', chapter.id)"
        >
          进入翻译
        </UiButton>
        <UiButton
          variant="secondary"
          size="xs"
          :disabled="imageCount === 0"
          @click="$emit('read', chapter.id)"
        >
          进入阅读
        </UiButton>
        <UiButton
          variant="secondary"
          size="xs"
          @click="$emit('edit', chapter.id)"
        >
          编辑
        </UiButton>
        <UiButton
          variant="danger"
          size="xs"
          @click="$emit('delete', chapter.id)"
        >
          删除
        </UiButton>
      </ProductActionRow>
    </div>
  </ProductRecordCard>
</template>

<style scoped>
.chapter-row {
  --product-record-card-background: var(--color-surface-interactive-hover);
  --product-record-card-border: transparent;
  --product-record-card-accent: var(--color-border-brand-gradient);
  --product-record-card-padding: 12px 16px;
  --product-record-card-shadow-hover: none;
}

.chapter-row--dragging {
  opacity: 0.6;
}

.chapter-row--drag-over {
  --product-record-card-border: var(--color-border-brand-gradient);
  --product-record-card-background: var(--color-surface-card);
}

.chapter-row__content {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
  min-width: 0;
}

.chapter-row__info {
  display: flex;
  flex: 1 1 260px;
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.chapter-row__order {
  flex-shrink: 0;
  min-width: 32px;
  color: var(--color-text-supporting);
  font-size: 0.8rem;
}

.chapter-row__title {
  overflow: hidden;
  color: var(--color-text-default);
  font-weight: 500;
  font-size: 0.9rem;
  overflow-wrap: anywhere;
  text-overflow: ellipsis;
  white-space: normal;
}

.chapter-row__meta {
  color: var(--color-text-supporting);
  font-size: 0.75rem;
}

.chapter-row__actions {
  flex: 1 1 280px;
  min-width: 0;
}
</style>
