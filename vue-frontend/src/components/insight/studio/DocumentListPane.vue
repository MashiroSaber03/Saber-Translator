<template>
  <div class="pane">
    <div class="pane-head">
      <h3>角色文档</h3>
      <span>{{ documents.length }}</span>
    </div>
    <div v-if="documents.length === 0" class="empty-copy">当前书还没有角色文档。</div>
    <div v-else class="list">
      <UiButton
        variant="toolbar"
        v-for="item in documents"
        :key="item.id"
        class="item"
        :class="{ active: currentDocumentId === item.id, opening: openingDocumentId === item.id }"
        :disabled="!!openingDocumentId"
        @click="$emit('open', item.id)"
      >
        <div class="item-main">
          <strong>{{ item.title }}</strong>
          <div class="item-meta">
            <span>{{ formatOrigin(item.origin) }}</span>
            <span>{{ formatTime(item.updated_at) }}</span>
          </div>
        </div>
        <div class="item-badges">
          <span v-if="openingDocumentId === item.id" class="opening-pill">打开中...</span>
          <span v-if="item.is_favorite" class="favorite-pill">收藏</span>
          <span v-if="item.source_character" class="source-pill">{{ item.source_character }}</span>
        </div>
      </UiButton>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import type { CharacterStudioSummary } from '@/types/characterStudio'

defineProps<{
  documents: CharacterStudioSummary[]
  currentDocumentId: string
  openingDocumentId: string
}>()

defineEmits<{
  (e: 'open', docId: string): void
}>()

function formatOrigin(origin: CharacterStudioSummary['origin']) {
  if (origin === 'analysis') return '分析生成'
  if (origin === 'imported') return '外部导入'
  return '手工创建'
}

function formatTime(value: string) {
  if (!value) return '未更新'
  return value.slice(0, 16).replace('T', ' ')
}
</script>

<style scoped>
.pane {
  --document-list-pane-border-default: rgba(37, 99, 199, .24);
  --document-list-pane-shadow-default: rgba(31, 70, 120, .08);
  --document-list-pane-surface-base: rgba(255, 255, 255, .95);
  --document-list-pane-surface-raised: rgba(255, 178, 46, .16);
  --document-list-pane-text-primary: #122b47;
  --document-list-pane-text-secondary: #9a6708;
}

.pane {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.pane-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.pane-head h3 {
  margin: 0;
  font-size: 14px;
}

.pane-head span {
  font-size: 12px;
  color: var(--studio-text-subtle);
}

.empty-copy {
  color: var(--studio-text-subtle);
  font-size: 13px;
  line-height: 1.6;
  padding: 8px 0;
}

.list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  width: 100%;
  text-align: left;
  border: 1px solid transparent;
  background: var(--color-surface-raised);
  border-radius: 16px;
  padding: 12px 12px;
  cursor: pointer;
}

.item:disabled {
  opacity: 0.74;
  cursor: wait;
}

.item.active {
  border-color: var(--document-list-pane-border-default);
  background: var(--document-list-pane-surface-base);
  box-shadow: 0 12px 24px var(--document-list-pane-shadow-default);
}

.item.opening {
  cursor: wait;
}

.item-main strong {
  display: block;
  color: var(--document-list-pane-text-primary);
  font-size: 13px;
}

.item-meta {
  display: flex;
  gap: 8px;
  margin-top: 6px;
  color: var(--studio-text-subtle);
  font-size: 11px;
  flex-wrap: wrap;
}

.item-badges {
  display: flex;
  flex-direction: column;
  gap: 6px;
  align-items: flex-end;
}

.opening-pill,
.favorite-pill,
.source-pill {
  border-radius: 999px;
  padding: 3px 8px;
  font-size: 10px;
}

.opening-pill {
  background: var(--studio-surface-tint-strong);
  color: var(--color-text-primary-strong);
}

.favorite-pill {
  background: var(--document-list-pane-surface-raised);
  color: var(--document-list-pane-text-secondary);
}

.source-pill {
  background: var(--studio-surface-tint);
  color: var(--color-text-primary-strong);
}
</style>
