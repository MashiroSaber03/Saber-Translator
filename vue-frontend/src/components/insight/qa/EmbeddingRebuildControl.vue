<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'

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
    <template v-if="isRebuilding">
      <UiSpinner :decorative="false" label="向量索引重建中" />
      <span>{{ progressLabel || '重建中...' }}</span>
    </template>
    <template v-else>
      <UiIcon name="refresh" />
      <span>重建向量</span>
    </template>
  </UiButton>
</template>
