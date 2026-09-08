<script setup lang="ts">
defineProps<{ modelValue: string; label: string; disabled?: boolean }>()
defineEmits<{ 'update:modelValue': [value: string] }>()
</script>
<template>
  <span class="color-control" :class="{ disabled }"
    ><span class="color-swatch" :style="{ background: modelValue }" /><span>{{
      modelValue.toUpperCase()
    }}</span
    ><input
      type="color"
      :aria-label="label"
      :value="modelValue"
      :disabled="disabled"
      @input="$emit('update:modelValue', ($event.target as HTMLInputElement).value)"
  /></span>
</template>
