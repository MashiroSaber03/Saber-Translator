<script setup lang="ts">
/**
 * 章节选择弹窗组件
 * 用于在有多个章节时让用户选择要翻译的章节
 * 基于 BaseModal 实现
 */

import { ref } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'

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

const props = defineProps<Props>()

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
      <button class="btn btn-secondary" @click="close">取消</button>
      <button 
        class="btn btn-primary" 
        :disabled="!selectedChapterId"
        @click="confirmSelection"
      >
        确定
      </button>
    </template>
  </BaseModal>
</template>

<style>
/* 不使用 scoped，因为 BaseModal 使用 Teleport 将内容传送到 body */

/* 章节选择弹窗特定样式 */
.chapter-select-modal .modal-body {
  padding: 24px;
}

.chapter-select-body .hint-text {
  font-size: 14px;
  color: var(--text-secondary);
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
  background: var(--bg-tertiary, #f1f5f9);
  border: 2px solid transparent;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.chapter-select-body .chapter-item:hover {
  background: var(--bg-primary, #f8fafc);
  border-color: var(--primary-light, #818cf8);
}

.chapter-select-body .chapter-item.selected {
  background: rgb(99, 102, 241, 0.1);
  border-color: var(--color-primary);
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
  color: var(--text-primary, #1a202c);
}

.chapter-select-body .chapter-pages {
  font-size: 12px;
  color: var(--text-secondary, #64748b);
}

.chapter-select-body .check-icon {
  font-size: 18px;
  color: var(--color-primary);
  font-weight: bold;
}

.chapter-select-modal .modal-footer {
  gap: 12px;
}
</style>
