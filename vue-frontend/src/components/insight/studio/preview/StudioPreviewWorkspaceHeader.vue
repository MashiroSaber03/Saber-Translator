<script setup lang="ts">
defineProps<{
  title: string
  description?: string
}>()
</script>

<template>
  <header class="studio-preview-workspace-header">
    <div class="studio-preview-workspace-header__copy">
      <h4 class="studio-preview-workspace-header__title">{{ title }}</h4>
      <p v-if="description" class="studio-preview-workspace-header__description">{{ description }}</p>
    </div>
    <div v-if="$slots.actions" class="studio-preview-workspace-header__actions">
      <slot name="actions" />
    </div>
  </header>
</template>

<style scoped>
.studio-preview-workspace-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.studio-preview-workspace-header__copy {
  min-width: 0;
}

.studio-preview-workspace-header__title {
  margin: 8px 0 0;
  color: var(--color-text-heading);
}

.studio-preview-workspace-header__description {
  margin: 8px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.7;
}

.studio-preview-workspace-header__actions {
  flex: 0 0 auto;
  min-width: 0;
}
</style>
