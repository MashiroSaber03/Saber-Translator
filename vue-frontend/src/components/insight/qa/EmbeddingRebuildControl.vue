<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'

defineProps<{
  isRebuilding: boolean
  progressLabel: string
}>()

defineEmits<{
  (event: 'rebuild'): void
}>()
</script>

<template>
  <UiButton
    variant="secondary"
    type="button"
    size="sm"
    title="重建向量索引"
    :disabled="isRebuilding"
    @click="$emit('rebuild')"
  >
    {{ isRebuilding ? `⏳ ${progressLabel || '重建中...'}` : '🔄 重建向量' }}
  </UiButton>
</template>
