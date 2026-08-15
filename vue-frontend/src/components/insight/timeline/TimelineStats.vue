<script setup lang="ts">
import { computed } from 'vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import type { TimelineStats } from '@/types/insight'

const props = defineProps<{
  stats: TimelineStats
}>()

const statItems = computed<ProductChipItem[]>(() => {
  const items: ProductChipItem[] = []

  if (props.stats.total_arcs) {
    items.push({
      id: 'arcs',
      iconName: 'book-marked',
      label: `${props.stats.total_arcs} 个剧情弧`,
      tone: 'neutral',
    })
  }

  items.push({
    id: 'events',
    iconName: 'bar-chart',
    label: `${props.stats.total_events} 个事件`,
    tone: 'neutral',
  })

  if (props.stats.total_characters) {
    items.push({
      id: 'characters',
      iconName: 'users',
      label: `${props.stats.total_characters} 个角色`,
      tone: 'neutral',
    })
  }

  if (props.stats.total_threads) {
    items.push({
      id: 'threads',
      iconName: 'link',
      label: `${props.stats.total_threads} 条线索`,
      tone: 'neutral',
    })
  }

  items.push({
    id: 'pages',
    iconName: 'file-text',
    label: `${props.stats.total_pages} 页`,
    tone: 'neutral',
  })

  return items
})
</script>

<template>
  <div class="timeline-stats">
    <ProductChipList aria-label="时间线统计" :items="statItems" />
  </div>
</template>

<style scoped>
.timeline-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--color-border-muted);
}
</style>
