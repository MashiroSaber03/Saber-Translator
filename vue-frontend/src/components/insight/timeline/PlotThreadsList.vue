<script setup lang="ts">
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import type { PlotThread } from './timelineTypes'

defineProps<{
  threads: PlotThread[]
}>()

function isResolved(thread: PlotThread): boolean {
  return thread.status === '已解决'
}

function threadStatus(thread: PlotThread): string {
  return thread.status || '进行中'
}

function threadName(thread: PlotThread): string {
  return thread.name || '未命名线索'
}

function threadChips(thread: PlotThread): ProductChipItem[] {
  const items: ProductChipItem[] = [
    {
      id: `${thread.id}-status`,
      label: threadStatus(thread),
      tone: isResolved(thread) ? 'success' : 'warning',
    },
  ]

  if (thread.introduced_at) {
    items.push({
      id: `${thread.id}-introduced`,
      label: `第 ${thread.introduced_at} 页引入`,
      tone: 'neutral',
    })
  }

  return items
}
</script>

<template>
  <div class="plot-threads-list">
    <ProductRecordCard
      v-for="thread in threads"
      :key="thread.id"
      accent
      class="plot-threads-list__card"
      :class="{ 'plot-threads-list__card--resolved': isResolved(thread) }"
      :aria-label="`线索：${threadName(thread)}`"
    >
      <template #meta>
        <span class="plot-threads-list__thread-name">{{ threadName(thread) }}</span>
      </template>

      <p v-if="thread.description" class="plot-threads-list__thread-description">{{ thread.description }}</p>

      <template #footer>
        <ProductChipList aria-label="线索状态" :items="threadChips(thread)" />
      </template>
    </ProductRecordCard>
  </div>
</template>

<style scoped>
.plot-threads-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.plot-threads-list__card {
  --product-record-card-background: var(--insight-surface-secondary);
  --product-record-card-accent: var(--color-status-warning);
  --product-record-card-radius: 10px;
  --product-record-card-padding: 14px;
}

.plot-threads-list__card--resolved {
  --product-record-card-accent: var(--color-status-success);

  opacity: 0.8;
}

.plot-threads-list__thread-name {
  color: var(--insight-text-primary);
  font-size: 14px;
  font-weight: 600;
}

.plot-threads-list__thread-description {
  margin: 0 0 8px;
  color: var(--insight-text-secondary);
  font-size: 13px;
  line-height: 1.5;
}
</style>
