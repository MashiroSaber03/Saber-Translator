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

const emit = defineEmits<{
  (event: 'files-change', files: File[]): void
}>()

function handleChange(event: Event) {
  const target = event.target
  if (!(target instanceof HTMLInputElement)) {
    emit('files-change', [])
    return
  }
  emit('files-change', Array.from(target.files ?? []))
}

defineExpose({
  click: () => inputRef.value?.click(),
  clear: () => {
    if (inputRef.value) {
      inputRef.value.value = ''
    }
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
    @change="handleChange"
  >
</template>

<style scoped>
.ui-file-input {
  color: var(--color-text-default);
  font: inherit;
}
</style>
