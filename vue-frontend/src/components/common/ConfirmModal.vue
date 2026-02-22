<script setup lang="ts">
/**
 * 确认对话框组件
 * 用于需要用户确认的操作，如删除、批量操作等
 * 基于 BaseModal 实现
 */

import BaseModal from './BaseModal.vue'

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

const props = withDefaults(defineProps<Props>(), {
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
      <button 
        class="btn btn-secondary" 
        @click="handleCancel"
      >
        {{ cancelText }}
      </button>
      <button 
        :class="['btn', confirmType === 'danger' ? 'btn-danger' : 'btn-primary']"
        @click="handleConfirm"
      >
        {{ confirmText }}
      </button>
    </template>
  </BaseModal>
</template>

<style>
/* 不使用 scoped，因为 BaseModal 使用 Teleport 将内容传送到 body */

/* 确认模态框特定样式 */
.confirm-modal .modal-body {
  padding: 20px;
  text-align: center;
}

.confirm-modal .confirm-message {
  margin: 0;
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-color);
}

.confirm-modal .modal-footer .btn-danger {
  background-color: #dc3545;
  color: white;
  border: none;
}

.confirm-modal .modal-footer .btn-danger:hover {
  background-color: #c82333;
}
</style>
