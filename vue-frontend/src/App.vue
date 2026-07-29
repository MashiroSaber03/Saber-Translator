<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue'
import { RouterView } from 'vue-router'
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

settingsStore.initSettings()

onMounted(() => {
  void settingsStore.loadFromBackend()
  void taskCenterStore.initialize()
})

onBeforeUnmount(() => taskCenterStore.disconnect())
</script>

<template>
  <OverlayLayer
    v-if="settingsStore.backendError"
    level="popover"
    passthrough
  >
    <ProductStatusBanner
      class="backend-restricted-banner"
      tone="danger"
      role="alert"
      title="设置受限模式"
    >
      设置加载失败，当前为受限模式：{{ settingsStore.backendError }}
      <template #actions>
        <UiButton
          variant="secondary"
          size="sm"
          @click="settingsStore.loadFromBackend()"
        >
          重试
        </UiButton>
      </template>
    </ProductStatusBanner>
  </OverlayLayer>
  <RouterView />
  <TaskCenterLauncher />
  <TaskCenterDrawer />
  <ProductConfirmProvider />
  <ProductTextInputProvider />
  <ToastNotification />
</template>

<style scoped>
.backend-restricted-banner {
  width: min(760px, calc(100% - 24px));
  margin: 12px auto 0;
}
</style>
