<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'

defineProps<{
  canGoNext: boolean
  canGoPrevious: boolean
}>()

defineEmits<{
  (event: 'next'): void
  (event: 'previous'): void
}>()
</script>

<template>
  <div class="navigation-buttons">
    <UiButton
      id="prevImageButton"
      class="navigation-button"
      variant="toolbar"
      :disabled="!canGoPrevious"
      @click="$emit('previous')"
    >
      上一张
    </UiButton>
    <UiButton
      id="nextImageButton"
      class="navigation-button"
      variant="toolbar"
      :disabled="!canGoNext"
      @click="$emit('next')"
    >
      下一张
    </UiButton>
  </div>
</template>

<style scoped>
.navigation-buttons {
  --settings-sidebar-navigation-button-background: #6c7784;
  --settings-sidebar-navigation-button-disabled-background: #c2c9d4;
  --settings-sidebar-navigation-button-hover-background: #5a6572;

  display: flex;
  gap: 10px;
  margin-top: 16px;
}

.navigation-button {
  flex: 1;
  min-height: 38px;
  border: none;
  border-radius: 8px;
  background: var(--settings-sidebar-navigation-button-background);
  color: var(--color-text-inverse);
  font-weight: 600;
  font-size: 13px;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.navigation-button:disabled {
  background: var(--settings-sidebar-navigation-button-disabled-background);
  cursor: not-allowed;
}

.navigation-button:hover:not(:disabled) {
  background: var(--settings-sidebar-navigation-button-hover-background);
}
</style>
