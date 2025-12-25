<script setup lang="ts">
/**
 * 移动端导航组件
 * 提供移动端的侧边栏切换按钮和导航功能
 * 
 * 功能：
 * - 汉堡菜单按钮
 * - 侧边栏遮罩层
 * - 滑动手势支持
 */

import { ref, watch, onMounted, onUnmounted } from 'vue'
import { useResponsive } from '@/composables/useResponsive'

// Props
const props = defineProps<{
  /** 侧边栏是否可见 */
  sidebarVisible?: boolean
}>()

// Emits
const emit = defineEmits<{
  /** 切换侧边栏 */
  (e: 'toggle'): void
  /** 关闭侧边栏 */
  (e: 'close'): void
}>()

// 响应式布局
const { isMobile, isTablet } = useResponsive()

// 本地状态
const isOpen = ref(props.sidebarVisible ?? false)

// 监听 props 变化
watch(() => props.sidebarVisible, (newVal) => {
  isOpen.value = newVal ?? false
})

// 切换侧边栏
function toggleSidebar() {
  emit('toggle')
}

// 关闭侧边栏
function closeSidebar() {
  emit('close')
}

// 处理遮罩层点击
function handleOverlayClick() {
  closeSidebar()
}

// 处理键盘事件（Escape 关闭）
function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape' && isOpen.value) {
    closeSidebar()
  }
}

// 生命周期
onMounted(() => {
  window.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <!-- 移动端导航按钮（仅在移动端/平板显示） -->
  <button
    v-if="isMobile || isTablet"
    class="mobile-nav-btn"
    :class="{ 'is-open': isOpen }"
    aria-label="切换导航菜单"
    @click="toggleSidebar"
  >
    <span class="hamburger-icon">
      <span class="hamburger-line"></span>
      <span class="hamburger-line"></span>
      <span class="hamburger-line"></span>
    </span>
  </button>
  
  <!-- 遮罩层（侧边栏打开时显示） -->
  <Teleport to="body">
    <Transition name="fade">
      <div
        v-if="(isMobile || isTablet) && isOpen"
        class="sidebar-overlay"
        @click="handleOverlayClick"
      />
    </Transition>
  </Teleport>
</template>

<style scoped>
/* 移动端导航按钮 */
.mobile-nav-btn {
  display: none;
  position: fixed;
  top: 12px;
  left: 12px;
  z-index: 1001;
  width: 40px;
  height: 40px;
  padding: 8px;
  background: var(--card-bg, #fff);
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.mobile-nav-btn:hover {
  background: var(--hover-bg, #f5f5f5);
}

.mobile-nav-btn:active {
  transform: scale(0.95);
}

/* 汉堡图标 */
.hamburger-icon {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  width: 100%;
  height: 100%;
}

.hamburger-line {
  display: block;
  width: 100%;
  height: 2px;
  background: var(--text-color, #333);
  border-radius: 1px;
  transition: all 0.3s ease;
}

/* 打开状态的汉堡图标（变成 X） */
.mobile-nav-btn.is-open .hamburger-line:nth-child(1) {
  transform: translateY(9px) rotate(45deg);
}

.mobile-nav-btn.is-open .hamburger-line:nth-child(2) {
  opacity: 0;
}

.mobile-nav-btn.is-open .hamburger-line:nth-child(3) {
  transform: translateY(-9px) rotate(-45deg);
}

/* 遮罩层 */
.sidebar-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 999;
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(2px);
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

/* 响应式显示 */
@media (max-width: 1024px) {
  .mobile-nav-btn {
    display: flex;
  }
}

@media (max-width: 768px) {
  .mobile-nav-btn {
    top: 8px;
    left: 8px;
    width: 36px;
    height: 36px;
  }
}
</style>
