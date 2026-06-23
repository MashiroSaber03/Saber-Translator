<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import type { BookData } from '@/types/api'

defineProps<{
  book: BookData
  chapterCount: number
  formatDate: (date?: string) => string
  getTagColor: (tagName: string) => string
}>()

defineEmits<{
  (event: 'addTag'): void
  (event: 'edit'): void
  (event: 'delete'): void
  (event: 'insight'): void
  (event: 'removeTag', tagName: string): void
}>()
</script>

<template>
  <div class="book-info-section">
    <div class="book-cover-large">
      <img
        v-if="book.cover"
        :src="book.cover"
        alt="封面"
      >
      <div v-else class="book-cover-placeholder">📖</div>
    </div>
    <div class="book-meta">
      <h3>{{ book.title }}</h3>
      <p class="meta-item">
        <span>标签：</span>
        <span v-if="book.tags && book.tags.length > 0" class="detail-tags">
          <span
            v-for="tag in book.tags"
            :key="tag"
            class="detail-tag"
            :style="{ background: getTagColor(tag) }"
          >
            {{ tag }}
            <span class="remove-detail-tag" @click.stop="$emit('removeTag', tag)">×</span>
          </span>
        </span>
        <span v-else class="no-tags-hint">暂无标签</span>
        <UiButton
          variant="toolbar"
          class="btn-add-tag"
          title="添加标签"
          @click="$emit('addTag')"
        >
          +
        </UiButton>
      </p>
      <p class="meta-item"><span>章节数：</span><span>{{ chapterCount }}</span></p>
      <p class="meta-item"><span>创建时间：</span><span>{{ formatDate(book.created_at || book.createdAt) }}</span></p>
      <p class="meta-item"><span>最后更新：</span><span>{{ formatDate(book.updated_at || book.updatedAt) }}</span></p>
      <div class="book-actions">
        <UiButton size="sm" variant="primary" @click="$emit('insight')">● 漫画分析</UiButton>
        <UiButton size="sm" variant="secondary" @click="$emit('edit')">编辑书籍</UiButton>
        <UiButton size="sm" variant="danger" @click="$emit('delete')">删除书籍</UiButton>
      </div>
    </div>
  </div>
</template>

<style scoped>
.book-info-section {
  display: flex;
  align-items: flex-start;
  gap: 24px;
}

.book-cover-large {
  flex-shrink: 0;
  width: 140px;
  overflow: hidden;
  border-radius: 12px;
  aspect-ratio: 3 / 4;
  background: linear-gradient(135deg, var(--color-surface-brand-gradient-start) 0%, var(--color-surface-brand-gradient-end) 100%);
  box-shadow: 0 8px 24px var(--book-detail-modal-shadow-default);
}

.book-cover-large img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.book-meta {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
}

.book-meta h3 {
  margin: 0 0 16px;
  color: var(--color-text-default);
  font-weight: 600;
  font-size: 1.3rem;
  line-height: 1.3;
  word-break: break-word;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 6px 0;
  color: var(--color-text-supporting);
  font-size: 0.9rem;
}

.meta-item span:first-child {
  flex-shrink: 0;
  min-width: 70px;
  color: var(--color-text-default);
  font-weight: 500;
}

.detail-tags {
  display: inline-flex;
  flex-wrap: wrap;
  gap: 6px;
}

.detail-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  color: var(--color-text-inverse);
  font-size: 0.75rem;
}

.no-tags-hint {
  color: var(--color-text-supporting);
  font-style: italic;
}

.btn-add-tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  margin-left: 6px;
  border: 1px dashed var(--color-border-muted);
  border-radius: 50%;
  background: transparent;
  color: var(--color-text-supporting);
  font-size: 0.9rem;
  cursor: pointer;
}

.btn-add-tag:hover {
  border-color: var(--color-border-brand-gradient);
  color: var(--book-detail-modal-text-primary);
}

.book-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
}
</style>
