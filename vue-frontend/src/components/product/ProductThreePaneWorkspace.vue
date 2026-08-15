<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  as?: string
  ariaLabel?: string
  leftWidth?: string
  rightWidth?: string
  leftMobileVisible?: boolean
  rightMobileVisible?: boolean
  showRight?: boolean
  mobileMode?: 'flow' | 'drawer'
}>(), {
  as: 'div',
  ariaLabel: undefined,
  leftWidth: '280px',
  rightWidth: '320px',
  leftMobileVisible: false,
  rightMobileVisible: false,
  showRight: true,
  mobileMode: 'flow',
})

const workspaceStyle = computed(() => ({
  '--product-three-pane-left-width': props.leftWidth,
  '--product-three-pane-right-width': props.rightWidth,
}))
</script>

<template>
  <component
    :is="as"
    class="product-three-pane-workspace"
    :class="`product-three-pane-workspace--mobile-${mobileMode}`"
    :aria-label="ariaLabel"
    :style="workspaceStyle"
  >
    <aside
      v-if="$slots.left"
      class="product-three-pane-workspace__pane product-three-pane-workspace__pane--left"
      :class="{ 'product-three-pane-workspace__pane--mobile-visible': leftMobileVisible }"
    >
      <slot name="left" />
    </aside>

    <section class="product-three-pane-workspace__main">
      <slot />
    </section>

    <aside
      v-if="showRight && $slots.right"
      class="product-three-pane-workspace__pane product-three-pane-workspace__pane--right"
      :class="{ 'product-three-pane-workspace__pane--mobile-visible': rightMobileVisible }"
    >
      <slot name="right" />
    </aside>
  </component>
</template>

<style scoped>
.product-three-pane-workspace {
  --product-three-pane-background: var(--color-surface-page);
  --product-three-pane-pane-background: var(--color-surface-muted);
  --product-three-pane-border: var(--color-border-muted);
  --product-three-pane-drawer-z-index: var(--z-dropdown);

  display: flex;
  flex: 1 1 auto;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  background: var(--product-three-pane-background);
}

.product-three-pane-workspace__pane {
  display: flex;
  flex: 0 0 auto;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  overflow-y: auto;
  background: var(--product-three-pane-pane-background);
}

.product-three-pane-workspace__pane--left {
  width: var(--product-three-pane-left-width);
  min-width: var(--product-three-pane-left-width);
  border-right: 1px solid var(--product-three-pane-border);
}

.product-three-pane-workspace__pane--right {
  width: var(--product-three-pane-right-width);
  min-width: var(--product-three-pane-right-width);
  border-left: 1px solid var(--product-three-pane-border);
}

.product-three-pane-workspace__main {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}

@media (--breakpoint-lg-down) {
  .product-three-pane-workspace--mobile-drawer {
    position: relative;
  }

  .product-three-pane-workspace--mobile-drawer .product-three-pane-workspace__pane {
    display: none;
    position: absolute;
    top: 0;
    bottom: 0;
    z-index: var(--product-three-pane-drawer-z-index);
    max-width: 85%;
    box-shadow: var(--card-shadow);
  }

  .product-three-pane-workspace--mobile-drawer .product-three-pane-workspace__pane--left {
    left: 0;
  }

  .product-three-pane-workspace--mobile-drawer .product-three-pane-workspace__pane--right {
    right: 0;
  }

  .product-three-pane-workspace--mobile-drawer .product-three-pane-workspace__pane--mobile-visible {
    display: flex;
  }
}
</style>
