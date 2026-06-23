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
 */

import UiButton from '@/components/ui/UiButton.vue'
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
  /** 视觉场景变体 */
  variant?: 'default' | 'settings'
}

const props = withDefaults(defineProps<Props>(), {
  defaultExpanded: true,
  storageKey: undefined,
  disabled: false,
  expandedIcon: '▼',
  collapsedIcon: '▶',
  variant: 'default'
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
      'is-disabled': disabled,
      [`collapsible-panel--${variant}`]: true
    }"
  >
    <UiButton
      type="button"
      variant="toolbar"
      class="collapsible-header"
      :class="{ 'clickable': !disabled }"
      :disabled="disabled"
      :aria-expanded="isExpanded ? 'true' : 'false'"
      @click="toggle"
    >
      <span class="collapsible-title">
        <slot name="title">{{ title }}</slot>
      </span>
      <span class="toggle-icon" :class="{ 'expanded': isExpanded }">
        {{ isExpanded ? expandedIcon : collapsedIcon }}
      </span>
    </UiButton>
    
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
.collapsible-panel--settings {
  --collapsible-panel-border-default: #dde7f4;
  --collapsible-panel-text-primary: #304464;
  --collapsible-panel-text-secondary: #6e81a2;
}

/* 可折叠面板容器 */
.collapsible-panel {
  margin-bottom: 16px;
}

/* 标题 */
.collapsible-title {
  font-weight: 600;
  font-size: 20px;
}

/* 切换图标 */
.toggle-icon {
  font-size: 12px;
  transition: transform 0.2s ease;
  color: var(--color-text-supporting, var(--color-text-secondary));
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

.collapsible-header {
  cursor: pointer;
  display: flex;
  width: 100%;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  margin: 0;
  border: 0;
  user-select: none;
  background: transparent;
  color: inherit;
  font: inherit;
  text-align: left;
}

.collapsible-header:hover {
  color: var(--color-text-link);
}

.collapsible-header .toggle-icon {
  transition: transform 0.3s ease;
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

.collapsible-panel--settings .collapsible-header {
  margin: 0 0 10px;
  padding: 0 0 8px;
  color: var(--collapsible-panel-text-primary);
  border-bottom: 1px solid var(--collapsible-panel-border-default);
}

.collapsible-panel--settings .collapsible-title {
  font-size: 17px;
  font-weight: 700;
}

.collapsible-panel--settings .toggle-icon {
  color: var(--collapsible-panel-text-secondary);
  font-size: 12px;
}

.collapsible-header.collapsed .toggle-icon {
  transform: rotate(-90deg);
}

.is-disabled .collapsible-header {
  cursor: not-allowed;
  opacity: 0.6;
}

.is-disabled .collapsible-header:hover {
  color: inherit;
}

.collapsible-panel--settings .collapsible-content {
  padding-top: 2px;
}
</style>
