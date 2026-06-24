<template>
  <div class="studio-tabs" role="tablist">
    <UiButton
      variant="toolbar"
      v-for="item in items"
      :key="item.value"
      class="tab-btn"
      :class="{ active: modelValue === item.value }"
      :data-tab="item.value"
      @click="$emit('update:modelValue', item.value)"
    >
      <span class="tab-icon">{{ item.icon }}</span>
      <span class="tab-label">{{ item.label }}</span>
    </UiButton>
  </div>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
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
  --studio-section-tabs-shadow-default: rgba(37, 99, 199, .16);
  --studio-section-tabs-surface-base: rgba(16, 39, 65, .04);
  --studio-section-tabs-surface-raised: rgba(37, 99, 199, .08);
  --studio-section-tabs-surface-muted: rgba(77, 134, 238, .1);
  --studio-section-tabs-text-primary: #55708f;
  --studio-section-tabs-text-secondary: #16365b;

  display: flex;
  gap: 8px;
  padding: 8px;
  border-radius: 18px;
  background: var(--studio-section-tabs-surface-base);
  border: 1px solid var(--studio-border-default);
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
  color: var(--studio-section-tabs-text-primary);
  cursor: pointer;
  white-space: nowrap;
  transition: background 0.2s ease, color 0.2s ease, transform 0.2s ease;
}

.tab-btn:hover {
  transform: translateY(-1px);
  background: var(--studio-section-tabs-surface-raised);
  color: var(--studio-text-default);
}

.tab-btn.active {
  background: linear-gradient(135deg, var(--studio-surface-tint-strong), var(--studio-section-tabs-surface-muted));
  color: var(--studio-section-tabs-text-secondary);
  box-shadow: inset 0 0 0 1px var(--studio-section-tabs-shadow-default);
}

.tab-icon {
  font-size: 15px;
}

.tab-label {
  font-size: 13px;
  font-weight: 600;
}
</style>
