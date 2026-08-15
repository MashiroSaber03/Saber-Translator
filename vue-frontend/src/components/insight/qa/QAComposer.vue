<script setup lang="ts">
import { computed } from 'vue'
import ProductComposer from '@/components/product/ProductComposer.vue'

const props = defineProps<{
  disabled: boolean
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
    class="qa-composer"
    v-model="questionModel"
    placeholder="输入你的问题..."
    :disabled="disabled"
    :show-submit-icon="false"
    submit-label="发送"
    @submit="emit('submit')"
  />
</template>

<style scoped>
.qa-composer {
  --ui-button-primary-disabled-background: var(--insight-text-muted);
  --ui-button-primary-disabled-color: var(--color-text-inverse);
  --ui-button-primary-disabled-opacity: 1;
}
</style>
