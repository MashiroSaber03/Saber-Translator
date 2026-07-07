<script setup lang="ts">
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

defineProps<{
  filter: string
  filterOptions: Array<{ label: string; value: string }>
}>()

defineEmits<{
  (event: 'update:filter', value: string): void
}>()
</script>

<template>
  <ProductSectionHeader title="笔记" icon-name="file-text" size="sm">
    <template #actions>
      <UiSelect
        :model-value="filter"
        aria-label="筛选笔记类型"
        :options="filterOptions"
        @change="$emit('update:filter', String($event))"
      />
    </template>
  </ProductSectionHeader>
</template>
