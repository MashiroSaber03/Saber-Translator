<script setup lang="ts">
/**
 * 可折叠面板组件
 * 提供可展开/折叠的面板容器
 * 
 * 功能：
 * - 点击标题展开/折叠
 * - 折叠状态图标切换（▶/▼）
 * - 支持默认展开状态
 * - 支持状态持久化（可选）
 * 
 * Requirements: UI一致性
 */

import { ref, watch, onMounted } from 'vue'

// ============================================================
// Props 和 Emits
// ============================================================

interface Props {
  /** 面板标题 */
  title: string
  /** 是否默认展开 */
  defaultExpanded?: boolean
  /** 持久化存储键（如果提供，则保存展开状态到 localStorage） */
  storageKey?: string
  /** 是否禁用折叠功能 */
  disabled?: boolean
  /** 自定义图标（展开状态） */
  expandedIcon?: string
  /** 自定义图标（折叠状态） */
  collapsedIcon?: string
}

const props = withDefaults(defineProps<Props>(), {
  defaultExpanded: true,
  storageKey: undefined,
  disabled: false,
  expandedIcon: '▼',
  collapsedIcon: '▶'
})

const emit = defineEmits<{
  /** 展开/折叠状态变化 */
  (e: 'toggle', expanded: boolean): void
}>()

// ============================================================
// 状态定义
// ============================================================

/** 是否展开 */
const isExpanded = ref(props.defaultExpanded)

// ============================================================
// 生命周期
// ============================================================

onMounted(() => {
  // 如果提供了存储键，从 localStorage 恢复状态
  if (props.storageKey) {
    loadExpandedState()
  }
})

// ============================================================
// 方法
// ============================================================

/**
 * 切换展开/折叠状态
 */
function toggle() {
  if (props.disabled) return
  
  isExpanded.value = !isExpanded.value
  emit('toggle', isExpanded.value)
  
  // 如果提供了存储键，保存状态到 localStorage
  if (props.storageKey) {
    saveExpandedState()
  }
}

/**
 * 从 localStorage 加载展开状态
 */
function loadExpandedState() {
  if (!props.storageKey) return
  
  try {
    const saved = localStorage.getItem(`collapsible_${props.storageKey}`)
    if (saved !== null) {
      isExpanded.value = saved === 'true'
    }
  } catch (error) {
    console.warn('[CollapsiblePanel] 加载展开状态失败:', error)
  }
}

/**
 * 保存展开状态到 localStorage
 */
function saveExpandedState() {
  if (!props.storageKey) return
  
  try {
    localStorage.setItem(`collapsible_${props.storageKey}`, String(isExpanded.value))
  } catch (error) {
    console.warn('[CollapsiblePanel] 保存展开状态失败:', error)
  }
}

/**
 * 展开面板
 */
function expand() {
  if (!isExpanded.value) {
    isExpanded.value = true
    emit('toggle', true)
    if (props.storageKey) {
      saveExpandedState()
    }
  }
}

/**
 * 折叠面板
 */
function collapse() {
  if (isExpanded.value) {
    isExpanded.value = false
    emit('toggle', false)
    if (props.storageKey) {
      saveExpandedState()
    }
  }
}

// 监听 defaultExpanded 变化
watch(() => props.defaultExpanded, (newValue) => {
  // 只有在没有存储键的情况下才响应 defaultExpanded 变化
  if (!props.storageKey) {
    isExpanded.value = newValue
  }
})

// ============================================================
// 暴露方法
// ============================================================

defineExpose({
  isExpanded,
  toggle,
  expand,
  collapse
})
</script>

<template>
  <div 
    class="collapsible-panel"
    :class="{ 
      'is-expanded': isExpanded,
      'is-collapsed': !isExpanded,
      'is-disabled': disabled
    }"
  >
    <!-- 面板头部 -->
    <div 
      class="collapsible-header"
      :class="{ 'clickable': !disabled }"
      @click="toggle"
    >
      <span class="collapsible-title">
        <slot name="title">{{ title }}</slot>
      </span>
      <span class="toggle-icon" :class="{ 'expanded': isExpanded }">
        {{ isExpanded ? expandedIcon : collapsedIcon }}
      </span>
    </div>
    
    <!-- 面板内容 -->
    <div 
      v-show="isExpanded" 
      class="collapsible-content"
    >
      <slot></slot>
    </div>
  </div>
</template>

<style scoped>
/* 可折叠面板容器 */
.collapsible-panel {
  margin-bottom: 16px;
}

/* 面板头部 */
.collapsible-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  user-select: none;
}

.collapsible-header.clickable {
  cursor: pointer;
}

.collapsible-header.clickable:hover {
  color: var(--primary-color, #4a90d9);
}

/* 禁用状态 */
.is-disabled .collapsible-header {
  cursor: not-allowed;
  opacity: 0.6;
}

.is-disabled .collapsible-header:hover {
  color: inherit;
}

/* 标题 */
.collapsible-title {
  font-weight: 500;
  font-size: 14px;
}

/* 切换图标 */
.toggle-icon {
  font-size: 12px;
  transition: transform 0.2s ease;
  color: var(--text-secondary, #666);
}

/* 面板内容 */
.collapsible-content {
  padding-top: 8px;
}

/* 动画效果（可选） */
.collapsible-content-enter-active,
.collapsible-content-leave-active {
  transition: all 0.2s ease;
  overflow: hidden;
}

.collapsible-content-enter-from,
.collapsible-content-leave-to {
  opacity: 0;
  max-height: 0;
}

/* ===================================
   可折叠面板样式 - 完整迁移自 components.css
   =================================== */

.collapsible-header {
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  margin: 0;
  user-select: none;
}

.collapsible-header:hover {
  color: #3498db;
}

.collapsible-header .toggle-icon {
  transition: transform 0.3s ease;
}

.collapsible-header.collapsed .toggle-icon {
  transform: rotate(-90deg);
}

.collapsible-content {
  overflow: visible;
  max-height: none;
  transition: max-height 0.3s ease;
}

.collapsible-content.collapsed {
  max-height: 0;
  overflow: hidden;
  padding-top: 0;
  padding-bottom: 0;
  margin-top: 0;
  margin-bottom: 0;
}
</style>
