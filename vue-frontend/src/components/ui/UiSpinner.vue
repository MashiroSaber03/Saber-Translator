<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  size?: number | string
  label?: string
  decorative?: boolean
}>(), {
  size: 14,
  label: '',
  decorative: true,
})

const cssSize = computed(() => typeof props.size === 'number' ? `${props.size}px` : props.size)
const isDecorative = computed(() => props.decorative || props.label.length === 0)
</script>

<template>
  <span
    class="ui-spinner"
    :style="{ '--ui-spinner-size': cssSize }"
    :aria-hidden="isDecorative ? 'true' : undefined"
    :aria-label="isDecorative ? undefined : label"
    :role="isDecorative ? undefined : 'status'"
  ></span>
</template>

<style scoped>
.ui-spinner {
  display: inline-block;
  flex: 0 0 auto;
  width: var(--ui-spinner-size);
  height: var(--ui-spinner-size);
  border: var(--ui-spinner-border-width, 2px) solid var(--ui-spinner-track-color, transparent);
  border-top-color: var(--ui-spinner-color, currentcolor);
  border-radius: 50%;
  animation: spin var(--ui-spinner-duration, 0.8s) linear infinite;
}
</style>
