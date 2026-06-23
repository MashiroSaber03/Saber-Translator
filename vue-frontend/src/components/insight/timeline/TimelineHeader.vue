<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'

defineProps<{
  isLoading: boolean
  isRegenerating: boolean
}>()

defineEmits<{
  (event: 'regenerate'): void
}>()
</script>

<template>
  <div class="timeline-header">
    <h3>📈 剧情时间线</h3>
    <UiButton
      variant="secondary"
      size="sm"
      :disabled="isLoading || isRegenerating"
      :loading="isRegenerating"
      @click="$emit('regenerate')"
    >
      <span v-if="isRegenerating" class="btn-spinner"></span>
      {{ isRegenerating ? '生成中...' : '🔄 重新生成' }}
    </UiButton>
  </div>
</template>

<style scoped>
.timeline-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.timeline-header h3 {
  margin: 0;
  font-size: 18px;
}

.btn-spinner {
  display: inline-block;
  width: 12px;
  height: 12px;
  border: 2px solid currentcolor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-right: 6px;
}
</style>
