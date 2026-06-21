<script setup lang="ts">
import './ConfirmModal.global.styles.css'
/**
 * 确认对话框组件
 * 用于需要用户确认的操作，如删除、批量操作等
 * 基于 BaseModal 实现
 */

import BaseModal from './BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'

// ============================================================
// Props 和 Emits 定义
// ============================================================

interface Props {
  /** 确认消息内容 */
  message: string
  /** 标题（可选） */
  title?: string
  /** 确认按钮文字 */
  confirmText?: string
  /** 取消按钮文字 */
  cancelText?: string
  /** 确认按钮类型（danger 为红色警告样式） */
  confirmType?: 'primary' | 'danger'
}

withDefaults(defineProps<Props>(), {
  title: '确认操作',
  confirmText: '确定',
  cancelText: '取消',
  confirmType: 'primary'
})

const emit = defineEmits<{
  /** 用户点击确认 */
  confirm: []
  /** 用户点击取消或关闭 */
  cancel: []
}>()

// ============================================================
// 方法
// ============================================================

/**
 * 处理确认按钮点击
 */
function handleConfirm(): void {
  emit('confirm')
}

/**
 * 处理取消按钮点击
 */
function handleCancel(): void {
  emit('cancel')
}
</script>

<template>
  <BaseModal
    :title="title"
    size="small"
    custom-class="confirm-modal"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="handleCancel"
  >
    <!-- 消息内容 -->
    <div class="confirm-modal-body">
      <p class="confirm-message">{{ message }}</p>
    </div>

    <!-- 按钮区域 -->
    <template #footer>
      <UiButton
        variant="secondary"
        @click="handleCancel"
      >
        {{ cancelText }}
      </UiButton>
      <UiButton
        :variant="confirmType"
        @click="handleConfirm"
      >
        {{ confirmText }}
      </UiButton>
    </template>
  </BaseModal>
</template>

<style scoped>
.confirm-message {
  margin: 0;
  font-size: 14px;
  line-height: 1.6;
  color: var(--color-text-strong);
}
</style>
