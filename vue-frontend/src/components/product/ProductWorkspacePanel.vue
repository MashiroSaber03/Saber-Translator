<script setup lang="ts">
import type { ProductClassValue } from '@/components/product/productClassTypes'

type ProductWorkspacePanelVariant = 'default' | 'tab' | 'wizard' | 'split'

withDefaults(defineProps<{
  variant?: ProductWorkspacePanelVariant
  ariaLabel?: string
  contentClass?: ProductClassValue
  flush?: boolean
}>(), {
  variant: 'default',
  ariaLabel: undefined,
  contentClass: '',
  flush: false,
})
</script>

<template>
  <section
    class="product-workspace-panel"
    :class="[
      `product-workspace-panel--${variant}`,
      { 'product-workspace-panel--flush': flush },
    ]"
    :aria-label="ariaLabel"
  >
    <div v-if="$slots.header" class="product-workspace-panel__header">
      <slot name="header" />
    </div>

    <div class="product-workspace-panel__scroll" :class="contentClass">
      <slot />
    </div>

    <div v-if="$slots.footer" class="product-workspace-panel__footer">
      <slot name="footer" />
    </div>
  </section>
</template>

<style scoped>
.product-workspace-panel {
  --internal-product-workspace-panel-radius: 12px;
  --internal-product-workspace-panel-shadow: var(--card-shadow);
  --internal-product-workspace-panel-padding: 20px;

  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: hidden;
  background: var(--product-workspace-panel-background, var(--color-surface-base));
  border: 1px solid var(--product-workspace-panel-border, var(--color-border-muted, var(--color-border-default)));
  border-radius: var(--product-workspace-panel-radius, var(--internal-product-workspace-panel-radius));
  box-shadow: var(--product-workspace-panel-shadow, var(--internal-product-workspace-panel-shadow));
}

.product-workspace-panel--tab {
  --internal-product-workspace-panel-radius: 0 0 12px 12px;
  --internal-product-workspace-panel-shadow: none;
}

.product-workspace-panel--wizard {
  --internal-product-workspace-panel-padding: 20px;
}

.product-workspace-panel--split {
  --internal-product-workspace-panel-radius: 0;
  --internal-product-workspace-panel-shadow: none;
}

.product-workspace-panel__header,
.product-workspace-panel__footer {
  flex: 0 0 auto;
  min-width: 0;
}

.product-workspace-panel__header {
  padding: var(--product-workspace-panel-header-padding, 16px 20px);
  border-bottom: 1px solid var(--product-workspace-panel-border, var(--color-border-muted, var(--color-border-default)));
}

.product-workspace-panel__scroll {
  flex: 1 1 auto;
  min-width: 0;
  min-height: 0;
  overflow: auto;
  padding: var(--product-workspace-panel-padding, var(--internal-product-workspace-panel-padding));
  scrollbar-gutter: stable;
}

.product-workspace-panel--flush > .product-workspace-panel__scroll {
  padding: 0;
}

.product-workspace-panel__footer {
  padding: var(--product-workspace-panel-footer-padding, 16px 20px);
  border-top: 1px solid var(--product-workspace-panel-border, var(--color-border-muted, var(--color-border-default)));
}
</style>
