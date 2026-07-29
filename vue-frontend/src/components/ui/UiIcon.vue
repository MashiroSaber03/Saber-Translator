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
const normalizedSize = computed(() => Number(props.size))
const normalizedStrokeWidth = computed(() => Number(props.strokeWidth))
</script>

<template>
  <component
    :is="iconComponent"
    class="ui-icon"
    :size="normalizedSize"
    :stroke-width="normalizedStrokeWidth"
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
