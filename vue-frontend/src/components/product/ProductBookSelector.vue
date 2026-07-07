<script setup lang="ts">
import { computed, useId } from 'vue'
import UiCombobox from '@/components/ui/UiCombobox.vue'
import type { UiSelectValue } from '@/components/ui/selectTypes'

export interface ProductBookSelectorBook {
  id: string
  title?: string | null
}

const props = withDefaults(defineProps<{
  modelValue: string
  books: ProductBookSelectorBook[]
  placeholder?: string
  disabled?: boolean
}>(), {
  placeholder: '选择书籍',
  disabled: false,
})

const emit = defineEmits<{
  (event: 'update:modelValue', value: string): void
  (event: 'select', bookId: string): void
}>()

const selectorId = useId()

const options = computed(() => [
  { label: props.placeholder, value: '' },
  ...props.books.map(book => ({
    label: book.title || book.id,
    value: book.id,
  })),
])

function handleChange(value: UiSelectValue): void {
  const bookId = String(value)
  emit('update:modelValue', bookId)
  if (bookId) {
    emit('select', bookId)
  }
}
</script>

<template>
  <div class="product-book-selector">
    <UiCombobox
      :input-id="selectorId"
      :aria-label="placeholder"
      :model-value="modelValue"
      :options="options"
      :placeholder="placeholder"
      :disabled="disabled"
      fit
      @change="handleChange"
    />
  </div>
</template>

<style scoped>
.product-book-selector {
  width: min(100%, 320px);
  min-width: 220px;
}

@media (--breakpoint-sm-down) {
  .product-book-selector {
    width: 100%;
    min-width: 0;
  }
}
</style>
