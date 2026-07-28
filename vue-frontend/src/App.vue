<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue'
import { RouterView } from 'vue-router'
import { useSettingsStore } from '@/stores/settings'
import ToastNotification from '@/components/common/ToastNotification.vue'
import ProductConfirmProvider from '@/components/product/ProductConfirmProvider.vue'
import ProductTextInputProvider from '@/components/product/ProductTextInputProvider.vue'
import TaskCenterDrawer from '@/components/task-center/TaskCenterDrawer.vue'
import TaskCenterLauncher from '@/components/task-center/TaskCenterLauncher.vue'
import { useTaskCenterStore } from '@/stores/taskCenterStore'

const settingsStore = useSettingsStore()
const taskCenterStore = useTaskCenterStore()

onMounted(() => {
  settingsStore.initSettings()
  void taskCenterStore.initialize()
})

onBeforeUnmount(() => taskCenterStore.disconnect())
</script>

<template>
  <RouterView />
  <TaskCenterLauncher />
  <TaskCenterDrawer />
  <ProductConfirmProvider />
  <ProductTextInputProvider />
  <ToastNotification />
</template>
