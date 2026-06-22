<script setup lang="ts">
/**
 * 章节选择弹窗组件
 * 用于在有多个章节时让用户选择要翻译的章节
 * 基于 BaseModal 实现
 */

import { ref } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'

// ============================================================
// 类型定义
// ============================================================

interface Chapter {
  id: string
  title: string
  startPage?: number
  endPage?: number
}

interface Props {
  chapters: Chapter[]
}

// ============================================================
// Props 和 Emits
// ============================================================

defineProps<Props>()

const emit = defineEmits<{
  close: []
  select: [chapterId: string]
}>()

// ============================================================
// 状态
// ============================================================

/** 选中的章节ID */
const selectedChapterId = ref<string>('')

// ============================================================
// 方法
// ============================================================

/**
 * 选择章节
 * @param chapterId - 章节ID
 */
function selectChapter(chapterId: string): void {
  selectedChapterId.value = chapterId
}

/**
 * 确认选择
 */
function confirmSelection(): void {
  if (selectedChapterId.value) {
    emit('select', selectedChapterId.value)
  }
}

/**
 * 关闭弹窗
 */
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
    :custom-style="{ '--ui-dialog-actions-gap': '12px' }"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="close"
  >
    <div class="chapter-select-body">
      <p class="hint-text">请选择要翻译的章节：</p>
      <div class="chapters-list">
        <div
          v-for="chapter in chapters"
          :key="chapter.id"
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
        </div>
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
  --chapter-select-modal-border-default: #818cf8;
  --chapter-select-modal-surface-base: #f1f5f9;
  --chapter-select-modal-text-primary: #1a202c;
  --chapter-select-modal-text-secondary: #64748b;
}

/* 章节选择弹窗特定样式 */

.chapter-select-body .hint-text {
  font-size: 14px;
  color: var(--insight-text-secondary);
  margin: 0 0 16px;
}

/* 章节列表 */
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
  background: var(--chapter-select-modal-surface-base);
  border: 2px solid transparent;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.chapter-select-body .chapter-item:hover {
  background: var(--color-surface-quiet);
  border-color: var(--chapter-select-modal-border-default);
}

.chapter-select-body .chapter-item.selected {
  background: var(--color-focus-brand-soft);
  border-color: var(--insight-color-primary);
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
  color: var(--color-text-default, var(--chapter-select-modal-text-primary));
}

.chapter-select-body .chapter-pages {
  font-size: 12px;
  color: var(--color-text-supporting, var(--chapter-select-modal-text-secondary));
}

.chapter-select-body .check-icon {
  font-size: 18px;
  color: var(--insight-color-primary);
  font-weight: bold;
}
</style>
