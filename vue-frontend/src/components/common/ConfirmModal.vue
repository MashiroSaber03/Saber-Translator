<script setup lang="ts">
/**
 * 确认对话框组件
 * 用于需要用户确认的操作，如删除、批量操作等
 */

import { ref, onMounted, onUnmounted } from 'vue'

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
// 状态
// ============================================================

/** 模态框容器引用 */
const modalRef = ref<HTMLElement | null>(null)

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

/**
 * 处理点击遮罩层（关闭模态框）
 */
function handleOverlayClick(event: MouseEvent): void {
  // 只有点击遮罩层本身才关闭，点击内容区域不关闭
  if (event.target === event.currentTarget) {
    emit('cancel')
  }
}

/**
 * 处理键盘事件
 */
function handleKeydown(event: KeyboardEvent): void {
  if (event.key === 'Escape') {
    emit('cancel')
  } else if (event.key === 'Enter') {
    emit('confirm')
  }
}

// ============================================================
// 生命周期
// ============================================================

onMounted(() => {
  // 添加键盘事件监听
  document.addEventListener('keydown', handleKeydown)
  // 聚焦到模态框以便接收键盘事件
  modalRef.value?.focus()
})

onUnmounted(() => {
  // 移除键盘事件监听
  document.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <div 
    ref="modalRef"
    class="modal-overlay confirm-modal-overlay"
    tabindex="-1"
    @click="handleOverlayClick"
  >
    <div class="modal-content confirm-modal-content">
      <!-- 标题 -->
      <div class="modal-header">
        <h3 class="modal-title">{{ title }}</h3>
        <button class="modal-close-btn" @click="handleCancel" title="关闭">×</button>
      </div>

      <!-- 消息内容 -->
      <div class="modal-body confirm-modal-body">
        <p class="confirm-message">{{ message }}</p>
      </div>

      <!-- 按钮区域 -->
      <div class="modal-footer confirm-modal-footer">
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
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 确认模态框特定样式 */
.confirm-modal-content {
  max-width: 400px;
  width: 90%;
}

.confirm-modal-body {
  padding: 20px;
  text-align: center;
}

.confirm-message {
  margin: 0;
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-color);
}

.confirm-modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 15px 20px;
  border-top: 1px solid var(--border-color);
}

.btn-danger {
  background-color: #dc3545;
  color: white;
  border: none;
}

.btn-danger:hover {
  background-color: #c82333;
}
</style>
