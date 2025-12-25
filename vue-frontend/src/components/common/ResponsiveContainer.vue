<script setup lang="ts">
/**
 * 响应式容器组件
 * 提供自适应布局的容器，根据屏幕尺寸自动调整布局
 * 
 * 功能：
 * - 自动检测屏幕尺寸
 * - 提供不同断点的插槽
 * - 支持侧边栏布局
 */

import { computed, provide, ref } from 'vue'
import { useResponsive, type LayoutMode } from '@/composables/useResponsive'

// Props
const props = withDefaults(defineProps<{
  /** 是否包含侧边栏 */
  hasSidebar?: boolean
  /** 侧边栏位置 */
  sidebarPosition?: 'left' | 'right'
  /** 侧边栏是否可见 */
  sidebarVisible?: boolean
  /** 是否使用全宽布局 */
  fullWidth?: boolean
  /** 自定义类名 */
  containerClass?: string
}>(), {
  hasSidebar: false,
  sidebarPosition: 'left',
  sidebarVisible: true,
  fullWidth: false,
  containerClass: '',
})

// Emits
const emit = defineEmits<{
  /** 侧边栏可见性变化 */
  (e: 'update:sidebarVisible', value: boolean): void
}>()

// 响应式布局
const {
  isMobile,
  isTablet,
  isDesktop,
  deviceType,
  layoutMode,
  sidebarWidth,
  shouldShowSidebar,
  toggleSidebar,
  showSidebar,
  hideSidebar,
} = useResponsive()

// 本地侧边栏状态
const localSidebarVisible = ref(props.sidebarVisible)

// 计算侧边栏是否显示
const isSidebarVisible = computed(() => {
  if (!props.hasSidebar) return false
  if (isDesktop.value) return true
  return localSidebarVisible.value
})

// 切换侧边栏
function handleToggleSidebar() {
  localSidebarVisible.value = !localSidebarVisible.value
  emit('update:sidebarVisible', localSidebarVisible.value)
}

// 关闭侧边栏
function handleCloseSidebar() {
  localSidebarVisible.value = false
  emit('update:sidebarVisible', false)
}

// 容器类名
const containerClasses = computed(() => [
  'responsive-container',
  `device-${deviceType.value}`,
  `layout-${layoutMode.value}`,
  {
    'has-sidebar': props.hasSidebar,
    'sidebar-visible': isSidebarVisible.value,
    'sidebar-left': props.sidebarPosition === 'left',
    'sidebar-right': props.sidebarPosition === 'right',
    'full-width': props.fullWidth,
  },
  props.containerClass,
])

// 主内容区样式
const mainContentStyle = computed(() => {
  if (!props.hasSidebar || !isSidebarVisible.value || isMobile.value) {
    return {}
  }
  
  const margin = props.sidebarPosition === 'left' ? 'marginLeft' : 'marginRight'
  return {
    [margin]: sidebarWidth.value,
  }
})

// 侧边栏样式
const sidebarStyle = computed(() => ({
  width: isMobile.value ? '100%' : sidebarWidth.value,
}))

// 提供响应式上下文给子组件
provide('responsive', {
  isMobile,
  isTablet,
  isDesktop,
  deviceType,
  layoutMode,
  sidebarVisible: isSidebarVisible,
  toggleSidebar: handleToggleSidebar,
  closeSidebar: handleCloseSidebar,
})
</script>

<template>
  <div :class="containerClasses">
    <!-- 移动端导航按钮 -->
    <slot 
      name="mobile-nav"
      :is-mobile="isMobile"
      :is-tablet="isTablet"
      :sidebar-visible="isSidebarVisible"
      :toggle-sidebar="handleToggleSidebar"
    >
      <!-- 默认移动端导航按钮 -->
      <button
        v-if="hasSidebar && (isMobile || isTablet)"
        class="mobile-nav-toggle"
        :class="{ 'is-open': isSidebarVisible }"
        aria-label="切换侧边栏"
        @click="handleToggleSidebar"
      >
        <span class="nav-icon">☰</span>
      </button>
    </slot>
    
    <!-- 侧边栏 -->
    <aside
      v-if="hasSidebar"
      class="responsive-sidebar"
      :class="{ 'is-visible': isSidebarVisible }"
      :style="sidebarStyle"
    >
      <slot 
        name="sidebar"
        :is-mobile="isMobile"
        :is-tablet="isTablet"
        :is-desktop="isDesktop"
        :close-sidebar="handleCloseSidebar"
      />
    </aside>
    
    <!-- 遮罩层（移动端侧边栏打开时） -->
    <Transition name="fade">
      <div
        v-if="hasSidebar && isSidebarVisible && (isMobile || isTablet)"
        class="sidebar-backdrop"
        @click="handleCloseSidebar"
      />
    </Transition>
    
    <!-- 主内容区 -->
    <main
      class="responsive-main"
      :style="mainContentStyle"
    >
      <slot 
        :is-mobile="isMobile"
        :is-tablet="isTablet"
        :is-desktop="isDesktop"
        :layout-mode="layoutMode"
        :device-type="deviceType"
      />
    </main>
  </div>
</template>

<style scoped>
/* 响应式容器 */
.responsive-container {
  position: relative;
  width: 100%;
  min-height: 100vh;
}

/* 侧边栏 */
.responsive-sidebar {
  position: fixed;
  top: var(--header-height, 60px);
  bottom: 0;
  z-index: 100;
  background: var(--sidebar-bg, #f8f9fa);
  border-right: 1px solid var(--border-color, #e0e0e0);
  overflow-y: auto;
  transition: transform 0.3s ease, width 0.3s ease;
}

.sidebar-left .responsive-sidebar {
  left: 0;
}

.sidebar-right .responsive-sidebar {
  right: 0;
  border-right: none;
  border-left: 1px solid var(--border-color, #e0e0e0);
}

/* 主内容区 */
.responsive-main {
  min-height: calc(100vh - var(--header-height, 60px));
  padding: var(--content-padding, 16px);
  transition: margin 0.3s ease;
}

/* 移动端导航按钮 */
.mobile-nav-toggle {
  display: none;
  position: fixed;
  top: calc(var(--header-height, 60px) + 12px);
  left: 12px;
  z-index: 101;
  width: 40px;
  height: 40px;
  padding: 0;
  background: var(--card-bg, #fff);
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 8px;
  cursor: pointer;
  font-size: 20px;
  line-height: 1;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.mobile-nav-toggle:hover {
  background: var(--hover-bg, #f5f5f5);
}

.mobile-nav-toggle.is-open .nav-icon {
  transform: rotate(90deg);
}

.nav-icon {
  display: inline-block;
  transition: transform 0.3s ease;
}

/* 遮罩层 */
.sidebar-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 99;
  background: rgba(0, 0, 0, 0.5);
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* ==================== 响应式布局 ==================== */

/* 平板端（768px - 1024px） */
@media (max-width: 1024px) {
  .mobile-nav-toggle {
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .responsive-sidebar {
    transform: translateX(-100%);
  }
  
  .sidebar-right .responsive-sidebar {
    transform: translateX(100%);
  }
  
  .responsive-sidebar.is-visible {
    transform: translateX(0);
  }
  
  .responsive-main {
    margin-left: 0 !important;
    margin-right: 0 !important;
  }
}

/* 移动端（< 768px） */
@media (max-width: 768px) {
  .responsive-sidebar {
    width: 100% !important;
    max-width: 320px;
  }
  
  .responsive-main {
    padding: var(--content-padding-mobile, 12px);
  }
  
  .mobile-nav-toggle {
    top: calc(var(--header-height-mobile, 50px) + 8px);
    left: 8px;
    width: 36px;
    height: 36px;
    font-size: 18px;
  }
}

/* 超小屏幕（< 480px） */
@media (max-width: 480px) {
  .responsive-sidebar {
    max-width: 280px;
  }
  
  .responsive-main {
    padding: var(--content-padding-xs, 8px);
  }
}

/* 全宽布局 */
.full-width .responsive-main {
  max-width: none;
  padding-left: 0;
  padding-right: 0;
}

/* 紧凑布局模式 */
.layout-compact .responsive-main {
  padding: 8px;
}

/* 宽屏布局模式 */
.layout-wide .responsive-main {
  max-width: 1600px;
  margin-left: auto;
  margin-right: auto;
}

.layout-wide.has-sidebar .responsive-main {
  margin-left: auto;
}
</style>
