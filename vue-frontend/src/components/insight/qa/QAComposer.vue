<script setup lang="ts">
import { computed } from 'vue'
import ProductComposer from '@/components/product/ProductComposer.vue'

const props = defineProps<{
  isStreaming: boolean
  question: string
}>()

const emit = defineEmits<{
  (event: 'submit'): void
  (event: 'update:question', value: string): void
}>()

const questionModel = computed({
  get: () => props.question,
  set: value => emit('update:question', value),
})
</script>

<template>
  <ProductComposer
    v-model="questionModel"
    placeholder="输入你的问题..."
    :disabled="isStreaming"
    submit-label="发送"
    @submit="emit('submit')"
  />
</template>
