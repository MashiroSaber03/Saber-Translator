<script setup lang="ts">
import { computed } from 'vue'
import { uiIconRegistry, type UiIconName } from './iconRegistry'

const props = withDefaults(defineProps<{
  name: UiIconName
  size?: number | string
  strokeWidth?: number | string
  label?: string
  decorative?: boolean
}>(), {
  size: 18,
  strokeWidth: 2,
  label: '',
  decorative: true,
})

const iconComponent = computed(() => uiIconRegistry[props.name])
const isDecorative = computed(() => props.decorative || props.label.length === 0)
const numericSize = computed(() => {
  const value = Number(props.size)
  return Number.isFinite(value) && value > 0 ? value : undefined
})
const cssSize = computed(() => numericSize.value === undefined ? String(props.size) : undefined)
const numericStrokeWidth = computed(() => {
  const value = Number(props.strokeWidth)
  return Number.isFinite(value) && value >= 0 ? value : 2
})
</script>

<template>
  <component
    :is="iconComponent"
    class="ui-icon"
    :size="numericSize"
    :stroke-width="numericStrokeWidth"
    :style="cssSize ? { width: cssSize, height: cssSize } : undefined"
    :aria-hidden="isDecorative ? 'true' : undefined"
    :aria-label="isDecorative ? undefined : label"
    :role="isDecorative ? undefined : 'img'"
    focusable="false"
  />
</template>

<style scoped>
.ui-icon {
  display: inline-block;
  flex: 0 0 auto;
  color: currentcolor;
  stroke: currentcolor;
  vertical-align: -0.125em;
}
</style>
