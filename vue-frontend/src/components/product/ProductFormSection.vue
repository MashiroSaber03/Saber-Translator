<script setup lang="ts">
withDefaults(defineProps<{
  padded?: boolean
}>(), {
  padded: true,
})
</script>

<template>
  <section class="product-form-section" :class="{ 'product-form-section--padded': padded }">
    <div v-if="$slots.title" class="product-form-section__title">
      <slot name="title" />
    </div>
    <slot />
  </section>
</template>

<style scoped>
.product-form-section {
  border: 1px solid var(--product-form-section-border, var(--color-border-muted));
  border-radius: 10px;
  background-color: var(--product-form-section-background, var(--color-surface-input));
  box-shadow: none;
  margin-bottom: var(--product-form-section-margin-bottom, 25px);
}

.product-form-section:last-child {
  margin-bottom: 0;
}

.product-form-section--padded {
  padding: var(--product-form-section-padding, 20px);
}

.product-form-section__title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: var(--product-form-section-title-margin-bottom, 15px);
  padding-bottom: var(--product-form-section-title-padding-bottom, 10px);
  border-bottom: var(--product-form-section-title-border-bottom, 1px solid var(--product-form-section-divider, var(--color-border-muted)));
  color: var(--product-form-section-title-text, var(--color-action-primary));
  font-size: var(--product-form-section-title-font-size, 1.1em);
  font-weight: var(--product-form-section-title-font-weight, 600);
}

.product-form-section :slotted(.ui-form-hint) {
  margin-top: 6px;
  color: var(--color-text-supporting);
  font-size: 0.85em;
  line-height: 1.45;
}
</style>
