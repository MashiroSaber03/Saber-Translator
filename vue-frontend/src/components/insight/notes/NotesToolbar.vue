<script setup lang="ts">
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { NoteType } from '@/stores/insightStore'

defineProps<{
  filter: NoteType | 'all'
  filterOptions: Array<{ label: string; value: NoteType | 'all' }>
}>()

defineEmits<{
  (event: 'update:filter', value: NoteType | 'all'): void
}>()
</script>

<template>
  <ProductSectionHeader title="笔记" icon-name="file-text" size="sm">
    <template #actions>
      <UiSelect
        :model-value="filter"
        aria-label="筛选笔记类型"
        :options="filterOptions"
        @change="$emit('update:filter', String($event) as NoteType | 'all')"
      />
    </template>
  </ProductSectionHeader>
</template>
