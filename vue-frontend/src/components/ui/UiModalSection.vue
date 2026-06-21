<script setup lang="ts">
withDefaults(defineProps<{
  title?: string
  description?: string
}>(), {
  title: '',
  description: '',
})
</script>

<template>
  <section class="ui-modal-section">
    <header v-if="title || description" class="ui-modal-section__header">
      <h4 v-if="title" class="ui-modal-section__title">{{ title }}</h4>
      <p v-if="description" class="ui-modal-section__description">{{ description }}</p>
    </header>
    <slot />
  </section>
</template>

<style scoped>
.ui-modal-section {
  padding: 18px 0;
  border-bottom: 1px solid var(--color-border-muted);
}

.ui-modal-section:last-child {
  border-bottom: 0;
}

.ui-modal-section__header {
  margin-bottom: 12px;
}

.ui-modal-section__title {
  margin: 0;
  color: var(--color-text-default);
  font-size: 1rem;
  font-weight: 700;
}

.ui-modal-section__description {
  margin: 4px 0 0;
  color: var(--color-text-supporting);
  font-size: 0.86rem;
  line-height: 1.45;
}
</style>
