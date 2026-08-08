<script setup lang="ts">
import { computed } from 'vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { UiIconName } from '@/components/ui/iconRegistry'

type ProductStatusTone = 'neutral' | 'info' | 'success' | 'warning' | 'danger'

const props = withDefaults(defineProps<{
  ariaLive?: 'polite' | 'assertive' | 'off'
  iconName?: UiIconName
  role?: 'status' | 'alert' | 'note'
  title?: string
  tone?: ProductStatusTone
}>(), {
  ariaLive: undefined,
  iconName: undefined,
  role: undefined,
  title: '',
  tone: 'info',
})

const defaultIconByTone: Record<ProductStatusTone, UiIconName> = {
  neutral: 'message',
  info: 'message',
  success: 'check',
  warning: 'alert-triangle',
  danger: 'alert-triangle',
}

const resolvedIconName = computed(() => props.iconName ?? defaultIconByTone[props.tone])
const resolvedRole = computed(() => props.role ?? (props.tone === 'danger' ? 'alert' : undefined))
</script>

<template>
  <section
    class="product-status-banner"
    :class="`product-status-banner--${tone}`"
    :role="resolvedRole"
    :aria-live="ariaLive"
  >
    <span
      v-if="$slots.icon"
      class="product-status-banner__icon product-status-banner__icon-text"
      aria-hidden="true"
    ><slot name="icon" /></span>
    <UiIcon v-else class="product-status-banner__icon" :name="resolvedIconName" size="18" aria-hidden="true" />
    <div class="product-status-banner__content">
      <strong v-if="title" class="product-status-banner__title">{{ title }}</strong>
      <div class="product-status-banner__body">
        <slot />
      </div>
    </div>
    <div v-if="$slots.actions" class="product-status-banner__actions">
      <slot name="actions" />
    </div>
  </section>
</template>

<style scoped>
.product-status-banner {
  --internal-product-status-banner-accent: var(--color-status-info);
  --internal-product-status-banner-background: var(--color-surface-quiet);

  display: flex;
  flex-direction: var(--product-status-banner-flex-direction, row);
  align-items: var(--product-status-banner-align-items, flex-start);
  justify-content: var(--product-status-banner-justify-content, flex-start);
  gap: var(--product-status-banner-gap, 10px);
  width: var(--product-status-banner-width, auto);
  min-height: var(--product-status-banner-min-height, 0);
  margin-inline: var(--product-status-banner-margin-inline, 0);
  padding: var(--product-status-banner-padding, 12px 14px);
  border: var(--product-status-banner-border, 1px solid var(--product-status-banner-accent, var(--internal-product-status-banner-accent)));
  border-radius: var(--product-status-banner-radius, 8px);
  background: var(--product-status-banner-background, var(--internal-product-status-banner-background));
  color: var(--color-text-default);
  line-height: var(--product-status-banner-line-height, 1.5);
  text-align: var(--product-status-banner-text-align, left);
}

.product-status-banner--neutral {
  --internal-product-status-banner-accent: var(--color-border-muted);
  --internal-product-status-banner-background: var(--color-surface-card);
}

.product-status-banner--info {
  --internal-product-status-banner-accent: var(--color-status-info);
  --internal-product-status-banner-background: var(--color-surface-quiet);
}

.product-status-banner--success {
  --internal-product-status-banner-accent: var(--color-status-success);
  --internal-product-status-banner-background: var(--color-surface-quiet);
}

.product-status-banner--warning {
  --internal-product-status-banner-accent: var(--color-status-warning);
  --internal-product-status-banner-background: var(--color-status-warning-surface-soft);
}

.product-status-banner--danger {
  --internal-product-status-banner-accent: var(--color-status-error);
  --internal-product-status-banner-background: var(--color-surface-danger-soft);
}

.product-status-banner__icon {
  display: var(--product-status-banner-icon-display, inline-flex);
  flex: 0 0 auto;
  margin: var(--product-status-banner-icon-margin, 2px 0 0);
  color: var(--product-status-banner-accent, var(--internal-product-status-banner-accent));
  transform: var(--product-status-banner-icon-transform, none);
}

.product-status-banner__icon-text {
  font-size: var(--product-status-banner-icon-font-size, 18px);
  line-height: 1;
}

.product-status-banner__content {
  display: var(--product-status-banner-content-display, block);
  flex: 1 1 auto;
  align-items: var(--product-status-banner-content-align-items, initial);
  gap: var(--product-status-banner-content-gap, 0);
  min-width: 0;
}

.product-status-banner__title {
  display: var(--product-status-banner-title-display, block);
  margin-bottom: var(--product-status-banner-title-margin-bottom, 2px);
  color: var(--product-status-banner-title-color, var(--color-text-strong));
  font-size: var(--product-status-banner-title-font-size, 0.92rem);
}

.product-status-banner__body {
  display: var(--product-status-banner-body-display, block);
  color: var(--product-status-banner-body-color, var(--color-text-default));
  font-size: var(--product-status-banner-body-font-size, 0.9rem);
  font-weight: var(--product-status-banner-body-font-weight, inherit);
}

.product-status-banner__actions {
  display: flex;
  flex: var(--product-status-banner-actions-flex, 0 0 auto);
  align-items: center;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  width: var(--product-status-banner-actions-width, auto);
  margin-left: var(--product-status-banner-actions-margin-left, 0);
}
</style>
