<template>
  <ProductRecordCard class="character-studio-entry-panel" aria-label="角色工坊入口">
    <template #meta>
      <div class="character-studio-entry-panel__eyebrow">角色工坊 2.0</div>
    </template>

    <template #actions>
      <UiButton variant="primary" @click="openStudio">打开角色工坊</UiButton>
    </template>

    <div class="character-studio-entry-panel__content">
      <h3 class="character-studio-entry-panel__title">角色工坊已升级为独立工作台</h3>
      <p class="character-studio-entry-panel__description">新的工作台会在独立页面中提供角色候选、世界书树、问候语、正则脚本、状态任务、聊天预览和卡片助手的完整闭环体验。</p>
    </div>
  </ProductRecordCard>
</template>

<script setup lang="ts">
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { useRouter } from 'vue-router'
import { useInsightStore } from '@/stores/insightStore'

const router = useRouter()
const insightStore = useInsightStore()

function openStudio() {
  if (!insightStore.currentBookId) return
  void router.push({
    name: 'character-studio',
    query: { book: insightStore.currentBookId },
  })
}
</script>

<style scoped>
.character-studio-entry-panel {
  --product-record-card-background:
    radial-gradient(circle at top right, color-mix(in srgb, var(--color-action-primary) 12%, transparent), transparent 24%),
    linear-gradient(180deg, color-mix(in srgb, var(--color-surface-base) 94%, transparent), color-mix(in srgb, var(--color-surface-quiet) 88%, transparent));
  --product-record-card-border: color-mix(in srgb, var(--color-text-link-strong) 14%, transparent);
  --product-record-card-radius: 28px;
  --product-record-card-shadow: 0 24px 48px var(--shadow-soft);
  --product-record-card-display: grid;
  --product-record-card-grid-template-columns: minmax(0, 1fr) auto;
  --product-record-card-grid-template-rows: auto auto;
  --product-record-card-align-items: center;
  --product-record-card-column-gap: 20px;
  --product-record-card-row-gap: 8px;
  --product-record-card-header-display: contents;
  --product-record-card-meta-grid-column: 1;
  --product-record-card-meta-grid-row: 1;
  --product-record-card-actions-grid-column: 2;
  --product-record-card-actions-grid-row: 1 / span 2;
  --product-record-card-actions-align-self: center;
  --product-record-card-body-grid-column: 1;
  --product-record-card-body-grid-row: 2;
  --ui-button-primary-background: linear-gradient(135deg, var(--color-action-primary-hover), var(--color-action-primary-soft));
  --ui-button-primary-hover-background: linear-gradient(135deg, var(--color-action-primary-hover), var(--color-action-primary-soft));
  --ui-button-primary-shadow: 0 10px 22px var(--shadow-action-brand);
  --ui-button-radius: 14px;
  --ui-button-padding: 12px 20px;
  --ui-button-width: auto;
  --ui-button-white-space: nowrap;

  width: auto;
  min-height: 0;
  margin: 20px;
  padding: 28px 30px;
}

.character-studio-entry-panel__eyebrow {
  font-size: 11px;
  letter-spacing: 0.12em;
  color: var(--color-text-link-strong);
  font-weight: 600;
}

.character-studio-entry-panel__title {
  margin: 10px 0 0;
  font-size: 24px;
  color: var(--color-text-strong);
}

.character-studio-entry-panel__description {
  margin: 12px 0 0;
  color: var(--color-text-secondary);
  max-width: 760px;
  line-height: 1.7;
}

@media (--breakpoint-lg-down) {
  .character-studio-entry-panel {
    padding: 22px;
  }
}
</style>
