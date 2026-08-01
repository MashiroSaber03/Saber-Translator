<script setup lang="ts">
import { computed } from 'vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { PlotThread } from '@/types/insight'

const props = defineProps<{
  plotThreads: PlotThread[]
  storySummary: string
}>()

const themeItems = computed<ProductChipItem[]>(() => {
  return props.plotThreads.slice(0, 5).map((thread) => ({
    id: thread.id,
    label: thread.name,
    tone: 'inverse',
  }))
})
</script>

<template>
  <div class="timeline-summary-card">
    <h4 class="timeline-summary-card__title">
      <UiIcon name="book-open" size="16" />
      <span>故事概要</span>
    </h4>
    <p class="timeline-summary-card__summary">{{ storySummary }}</p>
    <ProductChipList
      v-if="themeItems.length"
      aria-label="故事主题"
      label="主题："
      :items="themeItems"
    />
  </div>
</template>

<style scoped>
.timeline-summary-card {
  --product-chip-list-text: var(--color-text-inverse);
  --product-chip-list-label-text: var(--color-text-inverse);

  background: linear-gradient(135deg, var(--insight-action-primary) 0%, var(--insight-action-primary-strong) 100%);
  color: var(--color-text-inverse);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 24px;
}

.timeline-summary-card__title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 12px;
  font-size: 16px;
  font-weight: 600;
}

.timeline-summary-card__summary {
  font-size: 15px;
  line-height: 1.6;
  margin-bottom: 12px;
}
</style>
