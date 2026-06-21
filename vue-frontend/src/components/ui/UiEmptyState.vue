<script setup lang="ts">
defineProps<{
  icon: string
  title: string
  description?: string
}>()
</script>

<template>
  <div class="ui-empty-state">
    <div class="ui-empty-state__icon">{{ icon }}</div>
    <h2>{{ title }}</h2>
    <p v-if="description">{{ description }}</p>
    <slot />
  </div>
</template>

<style scoped>
.ui-empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 20px;
  text-align: center;
}

.ui-empty-state__icon {
  font-size: 4rem;
  margin-bottom: 16px;
}

.ui-empty-state h2 {
  font-size: 1.5rem;
  color: var(--color-text-default);
  margin: 0 0 8px;
}

.ui-empty-state p {
  color: var(--color-text-supporting);
  margin: 0 0 24px;
}
</style>
