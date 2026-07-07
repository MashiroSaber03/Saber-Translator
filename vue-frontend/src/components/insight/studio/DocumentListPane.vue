<template>
  <div class="document-list-pane">
    <div class="document-list-pane__head">
      <h3 class="document-list-pane__title">角色文档</h3>
      <span class="document-list-pane__count">{{ documents.length }}</span>
    </div>
    <ProductEmptyState
      v-if="documents.length === 0"
      icon-name="file-text"
      role="note"
      size="compact"
      title="当前书还没有角色文档"
    />
    <div v-else class="document-list-pane__list">
      <ProductRecordCard
        v-for="item in documents"
        :key="item.id"
        as="button"
        class="document-list-pane__item"
        :class="{
          'document-list-pane__item--active': currentDocumentId === item.id,
          'document-list-pane__item--opening': openingDocumentId === item.id,
        }"
        :aria-current="currentDocumentId === item.id ? 'true' : undefined"
        :disabled="!!openingDocumentId"
        @click="$emit('open', item.id)"
      >
        <div class="document-list-pane__item-body">
          <div class="document-list-pane__item-main">
            <strong class="document-list-pane__item-title">{{ item.title }}</strong>
            <div class="document-list-pane__item-meta">
              <span>{{ formatOrigin(item.origin) }}</span>
              <span>{{ formatTime(item.updated_at) }}</span>
            </div>
          </div>
          <ProductChipList
            v-if="documentChips(item).length > 0"
            class="document-list-pane__item-badges"
            :items="documentChips(item)"
            aria-label="角色文档状态"
          />
        </div>
      </ProductRecordCard>
    </div>
  </div>
</template>

<script setup lang="ts">
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import type { CharacterStudioSummary } from '@/types/characterStudio'

const props = defineProps<{
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

function documentChips(item: CharacterStudioSummary): ProductChipItem[] {
  const chips: ProductChipItem[] = []

  if (props.openingDocumentId === item.id) {
    chips.push({ id: `${item.id}-opening`, label: '打开中...', tone: 'primary' })
  }

  if (item.is_favorite) {
    chips.push({ id: `${item.id}-favorite`, label: '收藏', tone: 'warning' })
  }

  if (item.source_character) {
    chips.push({ id: `${item.id}-source`, label: item.source_character, tone: 'primary' })
  }

  return chips
}
</script>

<style scoped>
.document-list-pane {
  --document-list-pane-active-border: color-mix(in srgb, var(--color-action-primary) 24%, transparent);
  --document-list-pane-active-shadow: var(--studio-shadow-floating);
  --document-list-pane-active-background: var(--color-surface-raised);
  --document-list-pane-title-text: var(--studio-text-strong);

  display: flex;
  flex-direction: column;
  gap: 10px;
}

.document-list-pane__head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.document-list-pane__title {
  margin: 0;
  font-size: 14px;
}

.document-list-pane__count {
  font-size: 12px;
  color: var(--studio-text-subtle);
}

.document-list-pane__list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.document-list-pane__item {
  --product-record-card-background: var(--color-surface-raised);
  --product-record-card-border: transparent;
  --product-record-card-radius: 16px;
  --product-record-card-padding: 12px;

  width: 100%;
}

.document-list-pane__item-body {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
  width: 100%;
  text-align: left;
}

.document-list-pane__item--active {
  --product-record-card-background: var(--document-list-pane-active-background);
  --product-record-card-border: var(--document-list-pane-active-border);
  --product-record-card-shadow: 0 12px 24px var(--document-list-pane-active-shadow);
}

.document-list-pane__item--opening {
  cursor: wait;
}

.document-list-pane__item-main {
  flex: 1 1 180px;
  min-width: 0;
}

.document-list-pane__item-title {
  display: block;
  color: var(--document-list-pane-title-text);
  font-size: 13px;
}

.document-list-pane__item-meta {
  display: flex;
  gap: 8px;
  margin-top: 6px;
  color: var(--studio-text-subtle);
  font-size: 11px;
  flex-wrap: wrap;
}

.document-list-pane__item-badges {
  justify-content: flex-end;
}
</style>
