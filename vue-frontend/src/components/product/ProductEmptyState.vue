<script setup lang="ts">
import UiIcon from '@/components/ui/UiIcon.vue'
import type { UiIconName } from '@/components/ui/iconRegistry'

withDefaults(defineProps<{
  ariaLive?: 'polite' | 'assertive' | 'off'
  description?: string
  eyebrow?: string
  iconName: UiIconName
  role?: 'status' | 'note'
  size?: 'default' | 'compact'
  title: string
  variant?: 'default' | 'inverse'
}>(), {
  ariaLive: undefined,
  description: '',
  eyebrow: '',
  role: undefined,
  size: 'default',
  variant: 'default',
})
</script>

<template>
  <section
    class="product-empty-state"
    :class="[
      `product-empty-state--${variant}`,
      size === 'compact' ? 'product-empty-state--compact' : '',
    ]"
    :role="role"
    :aria-live="ariaLive"
  >
    <div class="product-empty-state__icon" aria-hidden="true">
      <UiIcon :name="iconName" size="40" />
    </div>
    <p v-if="eyebrow" class="product-empty-state__eyebrow">
      {{ eyebrow }}
    </p>
    <h2 class="product-empty-state__title">{{ title }}</h2>
    <p v-if="description" class="product-empty-state__description">
      {{ description }}
    </p>
    <div v-if="$slots.actions" class="product-empty-state__actions">
      <slot name="actions" />
    </div>
  </section>
</template>

<style scoped>
.product-empty-state {
  --internal-product-empty-state-description: var(--color-text-supporting);
  --internal-product-empty-state-icon-background: var(--color-surface-quiet);
  --internal-product-empty-state-icon-border: var(--color-border-muted);
  --internal-product-empty-state-icon-color: var(--color-action-brand);
  --internal-product-empty-state-min-height: 360px;
  --internal-product-empty-state-text: var(--color-text-default);
  --internal-product-empty-state-eyebrow-background: var(--color-surface-muted);
  --internal-product-empty-state-eyebrow-text: var(--color-text-supporting);
  --internal-product-empty-state-title: var(--color-text-strong);

  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  max-width: 520px;
  min-height: var(--product-empty-state-min-height, var(--internal-product-empty-state-min-height));
  margin-inline: auto;
  padding: 64px 20px;
  color: var(--product-empty-state-text, var(--internal-product-empty-state-text));
  text-align: center;
}

.product-empty-state--inverse {
  --internal-product-empty-state-description: color-mix(in srgb, var(--color-text-inverse) 72%, transparent);
  --internal-product-empty-state-icon-background: var(--color-overlay-inverse-soft);
  --internal-product-empty-state-icon-border: var(--color-overlay-inverse-emphasis);
  --internal-product-empty-state-icon-color: var(--color-text-inverse);
  --internal-product-empty-state-text: var(--color-text-inverse);
  --internal-product-empty-state-eyebrow-background: var(--color-overlay-inverse-soft);
  --internal-product-empty-state-eyebrow-text: var(--color-text-inverse);
  --internal-product-empty-state-title: var(--color-text-inverse);
}

.product-empty-state--compact {
  --internal-product-empty-state-min-height: 100%;

  max-width: none;
  padding: 16px 8px;
}

.product-empty-state__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 72px;
  height: 72px;
  margin-bottom: 18px;
  border: 1px solid var(--product-empty-state-icon-border, var(--internal-product-empty-state-icon-border));
  border-radius: 18px;
  background: var(--product-empty-state-icon-background, var(--internal-product-empty-state-icon-background));
  color: var(--product-empty-state-icon-color, var(--internal-product-empty-state-icon-color));
}

.product-empty-state--compact .product-empty-state__icon {
  width: 48px;
  height: 48px;
  margin-bottom: 10px;
  border-radius: 12px;
}

.product-empty-state__eyebrow {
  display: inline-flex;
  margin: 0 0 12px;
  padding: 5px 12px;
  border-radius: 999px;
  background: var(--product-empty-state-eyebrow-background, var(--internal-product-empty-state-eyebrow-background));
  color: var(--product-empty-state-eyebrow-text, var(--internal-product-empty-state-eyebrow-text));
  font-weight: 600;
  font-size: 0.78rem;
  line-height: 1.4;
}

.product-empty-state__title {
  margin: 0;
  color: var(--product-empty-state-title, var(--internal-product-empty-state-title));
  font-weight: 700;
  font-size: 1.45rem;
  line-height: 1.25;
}

.product-empty-state--compact .product-empty-state__title {
  font-size: 0.82rem;
  line-height: 1.3;
}

.product-empty-state__description {
  margin: 10px 0 0;
  color: var(--product-empty-state-description, var(--internal-product-empty-state-description));
  font-size: 0.95rem;
  line-height: 1.6;
}

.product-empty-state--compact .product-empty-state__description {
  margin-top: 6px;
  font-size: 0.78rem;
  line-height: 1.4;
}

.product-empty-state__actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
  margin-top: 24px;
}
</style>
