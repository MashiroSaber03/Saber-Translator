<script setup lang="ts">
import { computed, ref } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'

interface Chapter {
  id: string
  title: string
  startPage?: number
  endPage?: number
}

interface Props {
  chapters: Chapter[]
}

const props = defineProps<Props>()

const emit = defineEmits<{
  close: []
  select: [chapterId: string]
}>()

const selectedChapterId = ref<string>('')
const hasValidSelection = computed(() => (
  props.chapters.some(chapter => chapter.id === selectedChapterId.value)
))

function selectChapter(chapterId: string): void {
  selectedChapterId.value = chapterId
}

function chapterPageItems(chapter: Chapter): ProductChipItem[] {
  if (chapter.startPage === undefined || chapter.endPage === undefined) return []

  return [
    {
      id: `${chapter.id}-pages`,
      label: `第 ${chapter.startPage}-${chapter.endPage} 页`,
      tone: 'neutral',
    },
  ]
}

function confirmSelection(): void {
  if (hasValidSelection.value) {
    emit('select', selectedChapterId.value)
  }
}

function close(): void {
  emit('close')
}
</script>

<template>
  <BaseModal
    :model-value="true"
    title="选择章节"
    size="small"
    custom-class="chapter-select-modal"
    body-padding="spacious"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="close"
  >
    <div class="chapter-select-modal__body">
      <p class="chapter-select-modal__hint">请选择要翻译的章节：</p>
      <div class="chapter-select-modal__list">
        <ProductRecordCard
          v-for="chapter in chapters"
          :key="chapter.id"
          as="button"
          class="chapter-select-modal__choice-card"
          :class="{ 'chapter-select-modal__choice-card--selected': selectedChapterId === chapter.id }"
          :accent="selectedChapterId === chapter.id"
          :aria-label="`选择章节：${chapter.title}`"
          :aria-pressed="selectedChapterId === chapter.id"
          @click="selectChapter(chapter.id)"
        >
          <template #meta>
            <span class="chapter-select-modal__chapter-title">{{ chapter.title }}</span>
          </template>

          <template #actions>
            <UiIcon
              v-if="selectedChapterId === chapter.id"
              name="check"
              class="chapter-select-modal__check-icon"
              size="18"
            />
          </template>

          <ProductChipList
            v-if="chapterPageItems(chapter).length"
            aria-label="章节页码"
            :items="chapterPageItems(chapter)"
          />
        </ProductRecordCard>
      </div>
    </div>

    <template #footer>
      <ProductActionRow
        aria-label="章节选择操作"
        variant="dialog"
      >
        <UiButton variant="secondary" @click="close">取消</UiButton>
        <UiButton
          variant="primary"
          :disabled="!hasValidSelection"
          @click="confirmSelection"
        >
          确定
        </UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<style scoped>
.chapter-select-modal__hint {
  font-size: 14px;
  color: var(--insight-text-secondary);
  margin: 0 0 16px;
}

.chapter-select-modal__list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chapter-select-modal__choice-card {
  --product-record-card-background: var(--color-surface-muted);
  --product-record-card-border: transparent;
  --product-record-card-radius: 8px;
  --product-record-card-padding: 12px 16px;
  --product-record-card-accent: var(--insight-action-primary);
}

.chapter-select-modal__choice-card--selected {
  --product-record-card-background: var(--color-focus-brand-soft);
  --product-record-card-border: var(--insight-action-primary);
}

.chapter-select-modal__chapter-title {
  color: var(--insight-text-primary);
  font-size: 14px;
  font-weight: 500;
}

.chapter-select-modal__check-icon {
  color: var(--insight-action-primary);
}
</style>
