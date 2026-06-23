<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'

type PreviewTab = {
  value: 'chat' | 'assistant' | 'runtime'
  icon: string
  label: string
}

defineProps<{
  activeTab: PreviewTab['value']
  tabs: PreviewTab[]
}>()

defineEmits<{
  (event: 'update:activeTab', value: PreviewTab['value']): void
}>()
</script>

<template>
  <div class="workspace-tabs" role="tablist">
    <UiButton
      v-for="item in tabs"
      :key="item.value"
      variant="toolbar"
      class="tab-btn"
      :class="{ active: activeTab === item.value }"
      @click="$emit('update:activeTab', item.value)"
    >
      <span>{{ item.icon }}</span>
      <strong>{{ item.label }}</strong>
    </UiButton>
  </div>
</template>

<style scoped>
.workspace-tabs {
  display: flex;
  gap: 8px;
  width: 100%;
  padding: 6px;
  border: 1px solid var(--studio-border-default);
  border-radius: 20px;
  background: var(--color-surface-raised);
  box-shadow: 0 18px 32px var(--character-studio-preview-shell-shadow-default);
}

.tab-btn {
  display: inline-flex;
  flex: 1 1 0;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 14px;
  border: none;
  border-radius: 14px;
  background: transparent;
  color: var(--character-studio-preview-shell-text-secondary);
  cursor: pointer;
}

.tab-btn.active {
  background: linear-gradient(135deg, var(--studio-surface-tint-strong), var(--character-studio-preview-shell-surface-base));
  box-shadow: inset 0 0 0 1px var(--character-studio-preview-shell-shadow-raised);
  color: var(--character-studio-preview-shell-text-muted);
}

@media (--breakpoint-preview-down) {
  .tab-btn {
    flex: initial;
    justify-content: flex-start;
  }

  .workspace-tabs {
    overflow-x: auto;
  }
}
</style>
