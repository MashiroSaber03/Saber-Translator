<template>
  <div class="candidate-list-pane">
    <div class="candidate-list-pane__head">
      <h3 class="candidate-list-pane__title">分析候选</h3>
      <span class="candidate-list-pane__count">{{ candidates.length }}</span>
    </div>
    <ProductEmptyState
      v-if="!hasTimeline"
      icon-name="bar-chart"
      role="note"
      size="compact"
      title="暂无增强时间线"
    />
    <ProductEmptyState
      v-else-if="candidates.length === 0"
      role="note"
      size="compact"
      title="没有可用候选角色"
    />
    <div v-else class="candidate-list-pane__list">
      <ProductRecordCard v-for="item in candidates" :key="item.id" class="candidate-list-pane__row">
        <div class="candidate-list-pane__row-body">
          <div class="candidate-list-pane__candidate-main">
            <strong class="candidate-list-pane__candidate-name">{{ item.name }}</strong>
            <div class="candidate-list-pane__candidate-meta">
              首登 {{ item.first_appearance_page ?? '-' }} 页 · 关键时刻
              {{ item.key_moment_count }} · 相关页 {{ item.related_page_count }}
              <template v-if="item.related_page_numbers.length > 0">
                （{{ item.related_page_numbers.slice(0, 3).join(' / ') }}）
              </template>
            </div>
          </div>
          <ProductActionRow appearance="accent" aria-label="候选角色操作">
            <UiButton
              variant="secondary"
              :disabled="!!creatingCandidateId"
              size="sm"
              @click="$emit('create', item.id)"
            >
              {{ creatingCandidateId === item.id ? '创建中...' : '创建' }}
            </UiButton>
          </ProductActionRow>
        </div>
      </ProductRecordCard>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import type { CharacterStudioCandidate } from '@/types/characterStudio'

defineProps<{
  candidates: CharacterStudioCandidate[]
  hasTimeline: boolean
  creatingCandidateId: string
}>()

defineEmits<{
  (e: 'create', candidateId: string): void
}>()
</script>

<style scoped>
.candidate-list-pane {
  --candidate-list-pane-name-text: var(--studio-text-strong);
  --product-empty-state-align-items: flex-start;
  --product-empty-state-justify-content: flex-start;
  --product-empty-state-margin-inline: 0;
  --product-empty-state-min-height: 0;
  --product-empty-state-padding: 8px 0;
  --product-empty-state-text-align: left;
  --product-empty-state-title: var(--color-text-supporting);
  --product-empty-state-title-font-weight: 400;

  display: flex;
  flex-direction: column;
  gap: 10px;
}

.candidate-list-pane__head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.candidate-list-pane__title {
  margin: 0;
  font-size: 14px;
}

.candidate-list-pane__count {
  font-size: 12px;
  color: var(--studio-text-subtle);
}

.candidate-list-pane__list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.candidate-list-pane__row {
  --product-record-card-background: var(--color-surface-raised);
  --product-record-card-border: transparent;
  --product-record-card-radius: 16px;
  --product-record-card-padding: 12px;
}

.candidate-list-pane__row-body {
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}

.candidate-list-pane__candidate-main {
  flex: 1 1 180px;
  min-width: 0;
}

.candidate-list-pane__candidate-name {
  display: block;
  color: var(--candidate-list-pane-name-text);
  font-size: 13px;
}

.candidate-list-pane__candidate-meta {
  margin-top: 6px;
  color: var(--studio-text-subtle);
  font-size: 11px;
  line-height: 1.5;
}
</style>
