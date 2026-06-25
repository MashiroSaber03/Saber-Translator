<script setup lang="ts">
import { ref } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'

interface Chapter {
  id: string
  title: string
  startPage?: number
  endPage?: number
}

interface Props {
  chapters: Chapter[]
}

defineProps<Props>()

const emit = defineEmits<{
  close: []
  select: [chapterId: string]
}>()

const selectedChapterId = ref<string>('')

function selectChapter(chapterId: string): void {
  selectedChapterId.value = chapterId
}

function confirmSelection(): void {
  if (selectedChapterId.value) {
    emit('select', selectedChapterId.value)
  }
}

function close(): void {
  emit('close')
}
</script>

<template>
  <BaseModal
    title="📖 选择章节"
    size="small"
    custom-class="chapter-select-modal"
    body-padding="spacious"
    footer-gap="12px"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="close"
  >
    <div class="chapter-select-body">
      <p class="hint-text">请选择要翻译的章节：</p>
      <div class="chapters-list">
        <UiButton
          v-for="chapter in chapters"
          :key="chapter.id"
          variant="toolbar"
          type="button"
          class="chapter-item"
          :class="{ selected: selectedChapterId === chapter.id }"
          @click="selectChapter(chapter.id)"
        >
          <div class="chapter-info">
            <span class="chapter-title">{{ chapter.title }}</span>
            <span v-if="chapter.startPage && chapter.endPage" class="chapter-pages">
              第 {{ chapter.startPage }}-{{ chapter.endPage }} 页
            </span>
          </div>
          <span v-if="selectedChapterId === chapter.id" class="check-icon">✓</span>
        </UiButton>
      </div>
    </div>

    <template #footer>
      <UiButton variant="secondary" @click="close">取消</UiButton>
      <UiButton
        variant="primary"
        :disabled="!selectedChapterId"
        @click="confirmSelection"
      >
        确定
      </UiButton>
    </template>
  </BaseModal>
</template>

<style scoped>
.chapter-select-body {
  --chapter-select-modal-row-background: #f1f5f9;
  --chapter-select-modal-row-hover-border: #818cf8;
  --chapter-select-modal-title-text: #1a202c;
  --chapter-select-modal-page-text: #64748b;
}

.chapter-select-body .hint-text {
  font-size: 14px;
  color: var(--insight-text-secondary);
  margin: 0 0 16px;
}

.chapter-select-body .chapters-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chapter-select-body .chapter-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  width: 100%;
  background: var(--chapter-select-modal-row-background);
  border: 2px solid transparent;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.chapter-select-body .chapter-item:hover {
  background: var(--color-surface-quiet);
  border-color: var(--chapter-select-modal-row-hover-border);
}

.chapter-select-body .chapter-item.selected {
  background: var(--color-focus-brand-soft);
  border-color: var(--insight-action-primary);
}

.chapter-select-body .chapter-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
}

.chapter-select-body .chapter-title {
  font-size: 14px;
  font-weight: 500;
  color: var(--chapter-select-modal-title-text);
}

.chapter-select-body .chapter-pages {
  font-size: 12px;
  color: var(--chapter-select-modal-page-text);
}

.chapter-select-body .check-icon {
  font-size: 18px;
  color: var(--insight-action-primary);
  font-weight: bold;
}
</style>
