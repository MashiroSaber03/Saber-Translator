<script setup lang="ts">
withDefaults(defineProps<{
  level?: 'overlay' | 'popover' | 'mobile-overlay'
  passthrough?: boolean
}>(), {
  level: 'overlay',
  passthrough: false,
})

const emit = defineEmits<{
  backdrop: []
}>()
</script>

<template>
  <div
    class="ui-overlay-layer"
    :class="[
      `ui-overlay-layer--${level}`,
      { 'ui-overlay-layer--passthrough': passthrough },
    ]"
    @click.self="emit('backdrop')"
  >
    <slot />
  </div>
</template>

<style scoped>
.ui-overlay-layer {
  position: fixed;
  inset: 0;
}

.ui-overlay-layer--overlay {
  z-index: var(--z-overlay);
}

.ui-overlay-layer--popover {
  z-index: var(--z-popover);
}

.ui-overlay-layer--mobile-overlay {
  z-index: var(--z-mobile-overlay);
}

.ui-overlay-layer--passthrough {
  pointer-events: none;
}

.ui-overlay-layer--passthrough > * {
  pointer-events: auto;
}
</style>
