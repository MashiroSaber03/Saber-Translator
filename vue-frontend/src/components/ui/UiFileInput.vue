<script setup lang="ts">
import { ref, useAttrs } from 'vue'

defineOptions({ inheritAttrs: false })

withDefaults(defineProps<{
  accept?: string
  multiple?: boolean
  hidden?: boolean
  disabled?: boolean
}>(), {
  accept: '',
  multiple: false,
  hidden: false,
  disabled: false,
})

const attrs = useAttrs()
const inputRef = ref<HTMLInputElement | null>(null)

defineExpose({
  click: () => inputRef.value?.click(),
  get value() {
    return inputRef.value?.value ?? ''
  },
  set value(nextValue: string) {
    if (inputRef.value) {
      inputRef.value.value = nextValue
    }
  },
  get files() {
    return inputRef.value?.files ?? null
  },
})
</script>

<template>
  <input
    ref="inputRef"
    v-bind="attrs"
    class="ui-file-input"
    type="file"
    :accept="accept || undefined"
    :multiple="multiple"
    :hidden="hidden"
    :disabled="disabled"
  >
</template>

<style scoped>
.ui-file-input {
  color: var(--color-text-default);
  font: inherit;
}
</style>
