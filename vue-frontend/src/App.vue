<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, watch } from 'vue'
import { RouterView, useRoute } from 'vue-router'
import { useSettingsStore } from '@/stores/settings'
import ToastNotification from '@/components/common/ToastNotification.vue'
import ProductConfirmProvider from '@/components/product/ProductConfirmProvider.vue'
import ProductTextInputProvider from '@/components/product/ProductTextInputProvider.vue'
import TaskCenterDrawer from '@/components/task-center/TaskCenterDrawer.vue'
import TaskCenterLauncher from '@/components/task-center/TaskCenterLauncher.vue'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { useTaskCenterStore } from '@/stores/taskCenterStore'

const settingsStore = useSettingsStore()
const taskCenterStore = useTaskCenterStore()
const route = useRoute()
const readerImmersiveMode = computed(() => route.name === 'reader')
let appMounted = false

settingsStore.initSettings()

async function syncTaskCenterLifecycle() {
  if (!appMounted) return
  if (readerImmersiveMode.value) {
    taskCenterStore.close()
    taskCenterStore.disconnect()
    return
  }

  await taskCenterStore.initialize()
  if (!appMounted || readerImmersiveMode.value) {
    taskCenterStore.disconnect()
  }
}

onMounted(() => {
  appMounted = true
  void settingsStore.loadFromBackend()
  void syncTaskCenterLifecycle()
})

watch(readerImmersiveMode, () => {
  void syncTaskCenterLifecycle()
})

onBeforeUnmount(() => {
  appMounted = false
  taskCenterStore.disconnect()
})
</script>

<template>
  <OverlayLayer
    v-if="!settingsStore.isBackendReady && settingsStore.backendError"
    level="mobile-overlay"
    passthrough
  >
    <ProductStatusBanner
      class="backend-restricted-banner"
      tone="danger"
      role="alert"
      title="设置加载失败"
    >
      部分需要全局设置的功能暂不可用：{{ settingsStore.backendError }}
      <template #actions>
        <UiButton variant="secondary" size="sm" @click="settingsStore.loadFromBackend()">
          重试
        </UiButton>
      </template>
    </ProductStatusBanner>
  </OverlayLayer>
  <RouterView />
  <TaskCenterLauncher v-if="!readerImmersiveMode" />
  <TaskCenterDrawer v-if="!readerImmersiveMode" />
  <ProductConfirmProvider />
  <ProductTextInputProvider />
  <ToastNotification />
</template>

<style scoped>
.backend-restricted-banner {
  --product-status-banner-background: var(--color-surface-card);

  position: absolute;
  bottom: 16px;
  left: 16px;
  width: min(640px, calc(100% - 32px));
  margin: 0;
  box-shadow: var(--shadow-medium);
}

@media (--breakpoint-md-down) {
  .backend-restricted-banner {
    bottom: 12px;
    left: 12px;
    width: calc(100% - 24px);
  }
}
</style>
