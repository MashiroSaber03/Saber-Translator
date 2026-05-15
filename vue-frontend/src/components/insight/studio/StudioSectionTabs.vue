<template>
  <div class="studio-tabs" role="tablist">
    <button
      v-for="item in items"
      :key="item.value"
      class="tab-btn"
      :class="{ active: modelValue === item.value }"
      :data-tab="item.value"
      @click="$emit('update:modelValue', item.value)"
    >
      <span class="tab-icon">{{ item.icon }}</span>
      <span class="tab-label">{{ item.label }}</span>
    </button>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  modelValue: string
  items: Array<{ value: string; label: string; icon: string }>
}>()

defineEmits<{
  (e: 'update:modelValue', value: string): void
}>()
</script>

<style scoped>
.studio-tabs {
  display: flex;
  gap: 8px;
  padding: 8px;
  border-radius: 18px;
  background: rgba(16, 39, 65, 0.04);
  border: 1px solid rgba(28, 55, 94, 0.08);
  overflow-x: auto;
}

.tab-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border: none;
  border-radius: 14px;
  padding: 10px 14px;
  background: transparent;
  color: #55708f;
  cursor: pointer;
  white-space: nowrap;
  transition: background 0.2s ease, color 0.2s ease, transform 0.2s ease;
}

.tab-btn:hover {
  transform: translateY(-1px);
  background: rgba(37, 99, 199, 0.08);
  color: #234977;
}

.tab-btn.active {
  background: linear-gradient(135deg, rgba(37, 99, 199, 0.14), rgba(77, 134, 238, 0.1));
  color: #16365b;
  box-shadow: inset 0 0 0 1px rgba(37, 99, 199, 0.16);
}

.tab-icon {
  font-size: 15px;
}

.tab-label {
  font-size: 13px;
  font-weight: 600;
}
</style>
