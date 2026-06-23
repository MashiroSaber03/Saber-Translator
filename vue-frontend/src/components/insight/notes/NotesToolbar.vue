<script setup lang="ts">
import CustomSelect from '@/components/common/CustomSelect.vue'

defineProps<{
  filter: string
  filterOptions: Array<{ label: string; value: string }>
}>()

defineEmits<{
  (event: 'update:filter', value: string): void
}>()
</script>

<template>
  <div class="section-header-with-actions">
    <h3 class="section-title">📝 笔记</h3>
    <div class="notes-filter">
      <CustomSelect
        :model-value="filter"
        :options="filterOptions"
        @update:model-value="$emit('update:filter', String($event))"
      />
    </div>
  </div>
</template>

<style scoped>
.section-header-with-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  color: var(--insight-text-secondary);
  font-weight: 600;
  font-size: 14px;
}
</style>
