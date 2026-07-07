<script setup lang="ts">
withDefaults(defineProps<{
  framed?: boolean
  label: string
  scroll?: boolean
}>(), {
  framed: true,
  scroll: false,
})
</script>

<template>
  <section class="product-detail-section">
    <div class="product-detail-section__header">
      <h4 class="product-detail-section__label">{{ label }}</h4>
      <div v-if="$slots['label-actions']" class="product-detail-section__label-actions">
        <slot name="label-actions" />
      </div>
    </div>
    <div
      class="product-detail-section__content"
      :class="{
        'product-detail-section__content--framed': framed,
        'product-detail-section__content--scroll': scroll,
      }"
    >
      <slot />
    </div>
  </section>
</template>

<style scoped>
.product-detail-section {
  min-width: 0;
}

.product-detail-section__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
  margin-bottom: 8px;
}

.product-detail-section__label {
  display: block;
  min-width: 0;
  margin: 0;
  color: var(--color-text-supporting);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0;
  line-height: 1.2;
  text-transform: uppercase;
}

.product-detail-section__label-actions {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

.product-detail-section__content {
  min-width: 0;
  color: var(--color-text-default);
  font-size: 14px;
  line-height: 1.6;
}

.product-detail-section__content--framed {
  padding: 12px;
  border-radius: 8px;
  background: var(--color-surface-card);
}

.product-detail-section__content--scroll {
  max-height: 180px;
  overflow-y: auto;
  scrollbar-gutter: stable;
}
</style>
