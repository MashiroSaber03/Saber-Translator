<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import type { NoteData, NoteType } from '@/stores/insightStore'

defineProps<{
  note: NoteData
}>()

defineEmits<{
  (event: 'delete', noteId: string): void
  (event: 'edit', note: NoteData): void
  (event: 'showPage', pageNum: number): void
}>()

function formatDate(dateStr: string): string {
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function getNoteTypeIcon(type: NoteType): string {
  return type === 'qa' ? '💬' : '📝'
}
</script>

<template>
  <div
    class="note-item"
    :class="{ 'qa-note': note.type === 'qa' }"
  >
    <div class="note-header">
      <span class="note-type-icon">{{ getNoteTypeIcon(note.type) }}</span>
      <span class="note-date">{{ formatDate(note.createdAt) }}</span>
      <div class="note-actions">
        <UiIconButton
          label="编辑"
          size="sm"
          @click.stop="$emit('edit', note)"
        >
          ✏️
        </UiIconButton>
        <UiIconButton
          label="删除"
          variant="danger"
          size="sm"
          @click.stop="$emit('delete', note.id)"
        >
          🗑️
        </UiIconButton>
      </div>
    </div>

    <UiButton
      variant="toolbar"
      class="note-open-button"
      :aria-label="`编辑笔记：${note.title || note.question || note.content || '未命名笔记'}`"
      @click="$emit('edit', note)"
    >
      <span v-if="note.title" class="note-title">{{ note.title }}</span>
      <span v-if="note.type === 'qa'" class="note-content">
        <span class="qa-preview-text">Q: {{ note.question?.substring(0, 60) }}...</span>
      </span>
      <span v-else class="note-content">{{ note.content }}</span>

      <span v-if="note.tags && note.tags.length > 0" class="note-tags">
        <span v-for="tag in note.tags" :key="tag" class="note-tag">{{ tag }}</span>
      </span>
    </UiButton>

    <div v-if="note.type === 'qa' && note.citations && note.citations.length > 0" class="note-citations">
      <UiButton
        v-for="citation in note.citations.slice(0, 3)"
        :key="citation.page"
        variant="toolbar"
        class="citation-badge"
        :aria-label="`查看第 ${citation.page} 页`"
        @click="$emit('showPage', citation.page)"
      >
        第{{ citation.page }}页
      </UiButton>
      <span v-if="note.citations.length > 3" class="citation-badge">+{{ note.citations.length - 3 }}</span>
    </div>

    <div v-if="note.pageNum" class="note-page-link">
      <UiButton
        variant="toolbar"
        class="btn-link"
        @click.stop="$emit('showPage', note.pageNum)"
      >
        📄 第 {{ note.pageNum }} 页
      </UiButton>
    </div>
  </div>
</template>

<style scoped>
.note-item {
  margin-bottom: 10px;
  padding: 12px;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  background: var(--insight-surface-tertiary);
  transition: all 0.2s ease;
}

.note-item:hover {
  border-color: var(--insight-action-primary);
  box-shadow: 0 2px 8px var(--color-focus-brand-soft);
}

.note-item.qa-note {
  border-left: 3px solid var(--insight-action-primary);
}

.note-header {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 8px;
}

.note-type-icon {
  font-size: 14px;
}

.note-date {
  flex: 1;
  color: var(--insight-text-secondary);
  font-size: 12px;
}

.note-actions {
  display: flex;
  gap: 4px;
  margin-left: auto;
}

.note-open-button {
  display: block;
  width: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
  font: inherit;
  text-align: left;
}

.note-title {
  display: block;
  margin-bottom: 6px;
  overflow: hidden;
  color: var(--insight-text-primary);
  font-weight: 600;
  font-size: 14px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.note-content {
  display: block;
  color: var(--insight-text-secondary);
  font-size: 14px;
  line-height: 1.5;
  white-space: pre-wrap;
}

.qa-preview-text {
  color: var(--insight-text-secondary);
  font-size: 13px;
  font-style: italic;
}

.note-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 8px;
}

.note-tag {
  padding: 2px 6px;
  border-radius: 10px;
  background: var(--insight-action-primary);
  color: var(--color-text-inverse);
  font-size: 11px;
  opacity: 0.8;
}

.note-citations {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}

.citation-badge {
  display: inline-block;
  padding: 2px 8px;
  border: 0;
  border-radius: 10px;
  background: var(--insight-action-primary);
  color: var(--color-text-inverse);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
  transition: opacity 0.2s;
}

.citation-badge:hover {
  opacity: 0.8;
}

.note-page-link {
  margin-top: 8px;
}

.btn-link {
  padding: 0;
  border: none;
  background: none;
  color: var(--insight-action-primary);
  font-size: 12px;
  cursor: pointer;
}

.btn-link:hover {
  text-decoration: underline;
}
</style>
