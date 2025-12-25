<script setup lang="ts">
/**
 * 书籍右键上下文菜单组件
 * 提供快捷操作：打开详情、编辑、删除、管理标签
 */

import { ref, onMounted, onUnmounted } from 'vue'

// ============================================================
// Props 和 Emits 定义
// ============================================================

interface Props {
  /** 菜单显示位置 X */
  x: number
  /** 菜单显示位置 Y */
  y: number
  /** 书籍ID */
  bookId: string
}

const props = defineProps<Props>()

const emit = defineEmits<{
  /** 关闭菜单 */
  close: []
  /** 打开详情 */
  openDetail: [bookId: string]
  /** 编辑书籍 */
  edit: [bookId: string]
  /** 删除书籍 */
  delete: [bookId: string]
  /** 管理标签 */
  manageTags: [bookId: string]
  /** 进入批量模式 */
  enterBatchMode: []
}>()

// ============================================================
// 状态
// ============================================================

const menuRef = ref<HTMLElement | null>(null)

// ============================================================
// 方法
// ============================================================

/**
 * 处理点击外部关闭菜单
 */
function handleClickOutside(event: MouseEvent): void {
  if (menuRef.value && !menuRef.value.contains(event.target as Node)) {
    emit('close')
  }
}

/**
 * 处理菜单项点击
 */
function handleAction(action: string): void {
  switch (action) {
    case 'detail':
      emit('openDetail', props.bookId)
      break
    case 'edit':
      emit('edit', props.bookId)
      break
    case 'delete':
      emit('delete', props.bookId)
      break
    case 'tags':
      emit('manageTags', props.bookId)
      break
    case 'batch':
      emit('enterBatchMode')
      break
  }
  emit('close')
}

// ============================================================
// 生命周期
// ============================================================

onMounted(() => {
  // 延迟添加点击监听，避免立即触发关闭
  setTimeout(() => {
    document.addEventListener('click', handleClickOutside)
  }, 0)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<template>
  <div
    ref="menuRef"
    class="context-menu"
    :style="{ left: `${x}px`, top: `${y}px` }"
  >
    <button class="menu-item" @click="handleAction('detail')">
      <span class="menu-icon">📖</span>
      <span>打开详情</span>
    </button>
    <button class="menu-item" @click="handleAction('edit')">
      <span class="menu-icon">✏️</span>
      <span>编辑书籍</span>
    </button>
    <button class="menu-item" @click="handleAction('tags')">
      <span class="menu-icon">🏷️</span>
      <span>管理标签</span>
    </button>
    <div class="menu-divider"></div>
    <button class="menu-item" @click="handleAction('batch')">
      <span class="menu-icon">☑️</span>
      <span>批量操作</span>
    </button>
    <div class="menu-divider"></div>
    <button class="menu-item menu-item-danger" @click="handleAction('delete')">
      <span class="menu-icon">🗑️</span>
      <span>删除书籍</span>
    </button>
  </div>
</template>

<style scoped>
.context-menu {
  position: fixed;
  z-index: 1000;
  min-width: 160px;
  background: var(--card-bg, #fff);
  border: 1px solid var(--border-color, #ddd);
  border-radius: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
  padding: 6px 0;
  animation: fadeIn 0.15s ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.menu-item {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 10px 16px;
  background: none;
  border: none;
  cursor: pointer;
  font-size: 14px;
  color: var(--text-primary, #333);
  text-align: left;
  transition: background-color 0.15s;
}

.menu-item:hover {
  background: var(--bg-secondary, #f5f5f5);
}

.menu-item-danger {
  color: #dc3545;
}

.menu-item-danger:hover {
  background: #fee;
}

.menu-icon {
  font-size: 16px;
  width: 20px;
  text-align: center;
}

.menu-divider {
  height: 1px;
  background: var(--border-color, #eee);
  margin: 6px 0;
}
</style>
