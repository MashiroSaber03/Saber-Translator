<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import type { TagData } from '@/types/api'

defineProps<{
  availableTags: TagData[]
  filter: string
  showCreateNewTagOption: boolean
}>()

defineEmits<{
  (event: 'update:filter', value: string): void
  (event: 'add', tagName: string): void
  (event: 'submit'): void
}>()
</script>

<template>
  <div class="quick-tag-input-wrapper">
    <UiInput
      :model-value="filter"
      type="text"
      class="quick-tag-input"
      placeholder="输入标签名称进行搜索或创建..."
      autofocus
      @update:model-value="$emit('update:filter', String($event))"
      @keydown.enter="$emit('submit')"
    />
  </div>

  <div class="quick-tag-list">
    <UiButton
      v-for="tag in availableTags"
      :key="tag.name"
      variant="toolbar"
      type="button"
      class="quick-tag-item"
      :aria-label="`添加标签 ${tag.name}`"
      @click="$emit('add', tag.name)"
    >
      <span class="tag-color-dot" :style="{ background: tag.color || '#667eea' }"></span>
      <span class="quick-tag-name">{{ tag.name }}</span>
      <span class="tag-add-icon">+</span>
    </UiButton>

    <UiButton
      v-if="showCreateNewTagOption"
      variant="toolbar"
      type="button"
      class="quick-tag-item new-tag"
      :aria-label="`创建并添加标签 ${filter.trim()}`"
      @click="$emit('add', filter.trim())"
    >
      <span class="tag-icon">+</span>
      <span>创建并添加 "{{ filter.trim() }}"</span>
    </UiButton>

    <p
      v-if="availableTags.length === 0 && !showCreateNewTagOption"
      class="quick-tags-empty"
    >
      {{ filter ? '未找到匹配的标签' : '所有标签已添加或暂无标签' }}
    </p>
  </div>
</template>

<style scoped>
.quick-tag-input-wrapper {
  margin-bottom: 16px;
}

.quick-tag-input {
  width: 100%;
  padding: 12px 16px;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  background: var(--color-surface-card);
  color: var(--color-text-default);
  font-size: 0.95rem;
  transition: all 0.2s;
}

.quick-tag-input:focus {
  outline: none;
  border-color: var(--color-border-brand-gradient);
  box-shadow: 0 0 0 3px var(--book-detail-focus-shadow);
}

.quick-tag-input::placeholder {
  color: var(--color-text-supporting);
}

.quick-tag-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 260px;
  overflow-y: auto;
}

.quick-tag-item {
  display: flex;
  align-items: center;
  width: 100%;
  gap: 12px;
  padding: 12px 16px;
  border-radius: 8px;
  background: var(--color-surface-interactive-hover);
  color: inherit;
  cursor: pointer;
  text-align: left;
  transition: all 0.2s;
}

.quick-tag-item:hover {
  background: var(--color-border-muted);
  transform: translateX(4px);
}

.tag-color-dot {
  flex-shrink: 0;
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.quick-tag-name {
  flex: 1;
  color: var(--color-text-default);
  font-weight: 500;
}

.tag-add-icon {
  color: var(--book-detail-accent);
  font-weight: 600;
  font-size: 1.2rem;
  opacity: 0;
  transition: opacity 0.2s;
}

.quick-tag-item:hover .tag-add-icon {
  opacity: 1;
}

.quick-tag-item.new-tag {
  border: 1px dashed var(--book-detail-new-tag-border);
  background: linear-gradient(135deg, var(--book-detail-new-tag-background-start) 0%, var(--book-detail-new-tag-background-end) 100%);
}

.quick-tag-item.new-tag:hover {
  border-color: var(--book-detail-new-tag-border-hover);
  background: linear-gradient(135deg, var(--book-detail-new-tag-hover-start) 0%, var(--book-detail-new-tag-hover-end) 100%);
}

.tag-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  color: var(--book-detail-accent);
  font-weight: 600;
  font-size: 1.1rem;
}

.quick-tags-empty {
  margin: 0;
  padding: 24px 16px;
  color: var(--color-text-supporting);
  font-style: italic;
  text-align: center;
}
</style>
