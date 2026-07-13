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
    <UiIcon class="product-status-banner__icon" :name="resolvedIconName" size="18" aria-hidden="true" />
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
  align-items: flex-start;
  gap: 10px;
  padding: 12px 14px;
  border: 1px solid var(--product-status-banner-accent, var(--internal-product-status-banner-accent));
  border-radius: 8px;
  background: var(--product-status-banner-background, var(--internal-product-status-banner-background));
  color: var(--color-text-default);
  line-height: 1.5;
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
  flex: 0 0 auto;
  margin-top: 2px;
  color: var(--product-status-banner-accent, var(--internal-product-status-banner-accent));
}

.product-status-banner__content {
  flex: 1 1 auto;
  min-width: 0;
}

.product-status-banner__title {
  display: block;
  margin-bottom: 2px;
  color: var(--color-text-strong);
  font-size: 0.92rem;
}

.product-status-banner__body {
  color: var(--color-text-default);
  font-size: 0.9rem;
}

.product-status-banner__actions {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}
</style>
