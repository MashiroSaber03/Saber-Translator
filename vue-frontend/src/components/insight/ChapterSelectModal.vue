<script setup lang="ts">
/**
 * 章节选择弹窗组件
 * 用于在有多个章节时让用户选择要翻译的章节
 */

import { ref } from 'vue'

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
  <div class="modal chapter-select-modal show">
    <div class="modal-overlay" @click="close"></div>
    <div class="modal-content">
      <div class="modal-header">
        <h2>📖 选择章节</h2>
        <button class="modal-close" @click="close">&times;</button>
      </div>
      <div class="modal-body">
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
      <div class="modal-footer">
        <button class="btn btn-secondary" @click="close">取消</button>
        <button 
          class="btn btn-primary" 
          :disabled="!selectedChapterId"
          @click="confirmSelection"
        >
          确定
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ==================== 章节选择弹窗样式 ==================== */

/* CSS变量 */
.chapter-select-modal {
  --bg-primary: #f8fafc;
  --bg-secondary: #ffffff;
  --bg-tertiary: #f1f5f9;
  --text-primary: #1a202c;
  --text-secondary: #64748b;
  --text-muted: #94a3b8;
  --border-color: #e2e8f0;
  --primary-color: #6366f1;
  --primary-light: #818cf8;
  --primary-dark: #4f46e5;
  --success-color: #22c55e;
}

/* 模态框基础样式 */
.modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 1000;
  display: none;
  align-items: center;
  justify-content: center;
}

.modal.show {
  display: flex;
}

.modal-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
}

.modal-content {
  position: relative;
  background: var(--bg-secondary);
  border-radius: 12px;
  width: 90%;
  max-width: 500px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 24px;
  border-bottom: 1px solid var(--border-color);
}

.modal-header h2 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
}

.modal-close {
  width: 32px;
  height: 32px;
  border: none;
  background: transparent;
  color: var(--text-secondary);
  font-size: 24px;
  cursor: pointer;
  border-radius: 6px;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  line-height: 1;
}

.modal-close:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
}

.hint-text {
  font-size: 14px;
  color: var(--text-secondary);
  margin: 0 0 16px 0;
}

/* 章节列表 */
.chapters-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chapter-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  background: var(--bg-tertiary);
  border: 2px solid transparent;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.chapter-item:hover {
  background: var(--bg-primary);
  border-color: var(--primary-light);
}

.chapter-item.selected {
  background: rgba(99, 102, 241, 0.1);
  border-color: var(--primary-color);
}

.chapter-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
}

.chapter-title {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.chapter-pages {
  font-size: 12px;
  color: var(--text-secondary);
}

.check-icon {
  font-size: 18px;
  color: var(--primary-color);
  font-weight: bold;
}

.modal-footer {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
  padding: 16px 24px;
  border-top: 1px solid var(--border-color);
}

/* 按钮样式 */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 10px 18px;
  font-size: 14px;
  font-weight: 500;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary {
  background: var(--primary-color);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: var(--primary-dark);
}

.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-secondary {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
}

.btn-secondary:hover {
  background: var(--border-color);
}
</style>
