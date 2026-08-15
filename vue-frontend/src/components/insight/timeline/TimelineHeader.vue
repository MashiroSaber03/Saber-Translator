<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'

defineProps<{
  isLoading: boolean
  isPending: boolean
  isRegenerating: boolean
  showRegenerate: boolean
}>()

defineEmits<{
  (event: 'regenerate'): void
}>()
</script>

<template>
  <div class="timeline-header">
    <h3 class="timeline-header__title">
      <span aria-hidden="true">📈</span>
      <span>剧情时间线</span>
    </h3>
    <UiButton
      v-if="showRegenerate"
      variant="secondary"
      size="sm"
      class="timeline-header__regenerate-action"
      :disabled="isLoading || isRegenerating || isPending"
      :loading="isRegenerating || isPending"
      @click="$emit('regenerate')"
    >
      <UiSpinner v-if="isRegenerating || isPending" :size="14" />
      <span v-else aria-hidden="true">🔄</span>
      <span>{{ isRegenerating || isPending ? '生成中...' : '重新生成' }}</span>
    </UiButton>
  </div>
</template>

<style scoped>
.timeline-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}

.timeline-header__title {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  font-size: 18px;
}
</style>
