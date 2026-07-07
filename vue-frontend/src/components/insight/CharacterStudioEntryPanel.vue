<template>
  <ProductRecordCard class="character-studio-entry-panel" aria-label="角色工坊入口">
    <template #meta>
      <div class="character-studio-entry-panel__eyebrow">角色工坊</div>
    </template>

    <template #actions>
      <UiButton variant="primary" @click="openStudio">打开角色工坊</UiButton>
    </template>

    <div class="character-studio-entry-panel__content">
      <h3 class="character-studio-entry-panel__title">独立角色工作台</h3>
      <p class="character-studio-entry-panel__description">在独立页面中集中管理角色候选、世界书树、问候语、正则脚本、状态任务、聊天预览和卡片助手。</p>
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
  width: 100%;
  min-height: 180px;
  padding: 28px 30px;
}

.character-studio-entry-panel__eyebrow {
  font-size: 11px;
  letter-spacing: 0;
  color: var(--color-text-supporting);
  font-weight: 600;
}

.character-studio-entry-panel__title {
  margin: 0;
  font-size: 24px;
  color: var(--color-text-primary);
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
