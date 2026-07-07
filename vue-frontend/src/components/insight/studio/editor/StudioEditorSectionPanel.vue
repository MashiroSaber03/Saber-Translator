<script setup lang="ts">
defineProps<{
  title: string
  description?: string
  ariaLabel?: string
}>()
</script>

<template>
  <section class="studio-editor-section-panel" :aria-label="ariaLabel || title">
    <div class="studio-editor-section-panel__head">
      <div>
        <h3 class="studio-editor-section-panel__title">{{ title }}</h3>
        <p v-if="description" class="studio-editor-section-panel__description">{{ description }}</p>
      </div>
      <slot name="actions" />
    </div>

    <slot />
  </section>
</template>

<style scoped>
.studio-editor-section-panel {
  border-radius: 22px;
  border: 1px solid var(--studio-border-default);
  background: color-mix(in srgb, var(--color-surface-card) 82%, transparent);
  padding: 18px;
}

.studio-editor-section-panel__head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.studio-editor-section-panel__title {
  margin: 0;
  font-size: 18px;
  color: var(--color-text-heading);
}

.studio-editor-section-panel__description {
  margin: 6px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.7;
}

@media (--breakpoint-lg-down) {
  .studio-editor-section-panel__head {
    flex-direction: column;
  }
}
</style>
