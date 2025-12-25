<script setup lang="ts">
/**
 * 应用根组件
 * 提供路由视图容器和全局布局
 */
import { onMounted } from 'vue'
import { RouterView } from 'vue-router'
import { useSettingsStore } from '@/stores/settingsStore'
import ToastNotification from '@/components/common/ToastNotification.vue'

// 初始化设置 Store 并应用主题
const settingsStore = useSettingsStore()

onMounted(() => {
  // 初始化所有设置（包括主题、服务商配置等）
  settingsStore.initSettings()
})
</script>

<template>
  <RouterView />
  <!-- 全局 Toast 通知（只挂载一次） -->
  <ToastNotification />
</template>

<style>
/* 全局样式通过 CSS 入口文件引入 */
</style>
