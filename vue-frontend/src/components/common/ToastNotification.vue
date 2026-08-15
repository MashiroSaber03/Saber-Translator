<template>
  <Teleport to="body">
    <div class="vue-toast-container">
      <TransitionGroup name="toast-slide">
        <div
          v-for="toast in toasts"
          :key="toast.id"
          class="vue-toast-message"
          :class="'vue-toast-' + toast.type"
        >
          <span>{{ toast.message }}</span>
          <UiIconButton
            class="vue-toast-close"
            label="关闭通知"
            title="关闭通知"
            variant="plain"
            size="xs"
            shape="circle"
            @click.stop="removeToast(toast.id)"
          >
            <UiIcon name="x" size="14" />
          </UiIconButton>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import { onUnmounted } from 'vue'
import { toastService } from '@/utils/toast'

const toasts = toastService.toasts

const removeToast = (id: number): void => {
  toastService.removeToast(id)
}

onUnmounted(() => {
  toastService.clearAll()
})
</script>

<style scoped>
.vue-toast-container {
  position: fixed;
  bottom: 80px;
  left: 50%;
  transform: translateX(-50%);
  z-index: var(--z-toast);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  max-width: 80%;
  pointer-events: none;
}

.vue-toast-message {
  background-color: var(--toast-notification-default-background);
  border-radius: 8px;
  padding: 12px 24px;
  padding-right: 36px;
  margin-bottom: 0;
  box-shadow: 0 4px 12px var(--toast-notification-shadow-color);
  position: relative;
  max-width: 100%;
  pointer-events: auto;
  word-break: break-word;
  color: white;
  text-align: center;
  font-size: 14px;
}

.vue-toast-info {
  background-color: var(--toast-notification-info-background);
}

.vue-toast-success {
  background-color: var(--toast-notification-success-background);
}

.vue-toast-warning {
  background-color: var(--toast-notification-warning-background);
}

.vue-toast-error {
  background-color: var(--toast-notification-error-background);
}

.vue-toast-close {
  position: absolute;
  top: calc(50% - 12px);
  right: 10px;
  color: var(--toast-notification-close-text);
}

.vue-toast-close:hover {
  color: white;
}

.toast-slide-enter-active {
  animation: vueToastSlideUp 0.3s ease-out forwards;
}

.toast-slide-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.toast-slide-leave-to {
  opacity: 0;
  transform: translateY(20px);
}

.toast-slide-move {
  transition: transform 0.3s ease;
}

@keyframes vueToastSlideUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (--breakpoint-md-down) {
  .vue-toast-container {
    bottom: 60px;
    max-width: 90%;
  }

  .vue-toast-message {
    padding: 10px 20px;
    padding-right: 32px;
    font-size: 13px;
  }
}
</style>
