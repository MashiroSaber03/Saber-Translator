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
          <span v-if="toast.isHTML" v-html="toast.message"></span>
          <span v-else>{{ toast.message }}</span>
          <UiButton variant="toolbar" class="vue-toast-close" @click.stop="removeToast(toast.id)">×</UiButton>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
/**
 * Toast 通知组件
 * 使用全局 toastService 显示消息提示
 * 样式与业务契约完全一致：底部居中、彩色背景、白色文字
 */
import { onUnmounted } from 'vue'
import { toastService } from '@/utils/toast'

// 使用全局 toast 服务的消息队列
const toasts = toastService.toasts

/**
 * 移除指定 ID 的 Toast
 * @param id - Toast ID
 */
const removeToast = (id: number): void => {
  toastService.removeToast(id)
}

// 组件卸载时清除所有定时器
onUnmounted(() => {
  toastService.clearAll()
})

// 暴露方法供外部使用（保留外部调用 API）
defineExpose({
  toasts,
  addToast: toastService.addToast,
  removeToast: toastService.removeToast,
  clearAll: toastService.clearAll,
  success: toastService.success,
  error: toastService.error,
  info: toastService.info,
  warning: toastService.warning,
  showGeneralMessage: toastService.showGeneralMessage,
  clearGeneralMessageById: toastService.clearGeneralMessageById,
  clearAllGeneralMessages: toastService.clearAllGeneralMessages
})
</script>

<style scoped>
/* ============ Vue Toast 组件样式 ============ */

/* 使用独特前缀避免与全局 CSS 冲突 */

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
  background-color: var(--toast-notification-surface-base);
  border-radius: 8px;
  padding: 12px 24px;
  padding-right: 36px;
  margin-bottom: 0;
  box-shadow: 0 4px 12px var(--toast-notification-shadow-default);
  position: relative;
  max-width: 100%;
  pointer-events: auto;
  word-break: break-word;
  color: white;
  text-align: center;
  font-size: 14px;
}

.vue-toast-info {
  background-color: var(--toast-notification-surface-raised);
}

.vue-toast-success {
  background-color: var(--toast-notification-surface-muted);
}

.vue-toast-warning {
  background-color: var(--toast-notification-surface-subtle);
}

.vue-toast-error {
  background-color: var(--toast-notification-surface-hover);
}

.vue-toast-close {
  position: absolute;
  top: 50%;
  right: 10px;
  transform: translateY(-50%);
  background: none;
  border: none;
  font-size: 18px;
  cursor: pointer;
  color: var(--toast-notification-text-primary);
  line-height: 1;
  padding: 0;
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.vue-toast-close:hover {
  color: white;
}

/* Toast 从下往上滑入动画 */
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

/* 响应式适配 */
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
