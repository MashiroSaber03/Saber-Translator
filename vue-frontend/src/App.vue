<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { RouterView, useRoute, useRouter } from 'vue-router'
import { useSettingsStore } from '@/stores/settings'
import ToastNotification from '@/components/common/ToastNotification.vue'
import ProductConfirmProvider from '@/components/product/ProductConfirmProvider.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductTextInputProvider from '@/components/product/ProductTextInputProvider.vue'
import TaskCenterDrawer from '@/components/task-center/TaskCenterDrawer.vue'
import TaskCenterLauncher from '@/components/task-center/TaskCenterLauncher.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import { useRuntimeStore } from '@/stores/runtimeStore'
import { useAuthStore } from '@/stores/authStore'
import AccountDock from '@/components/common/AccountDock.vue'

const settingsStore = useSettingsStore()
const taskCenterStore = useTaskCenterStore()
const runtimeStore = useRuntimeStore()
const authStore = useAuthStore()
const route = useRoute()
const router = useRouter()
const routerReady = ref(false)
const settingsLoading = ref(false)
const loadedSettingsContext = ref<string | null>(null)
const readerImmersiveMode = computed(() => route.name === 'reader')
const applicationRoute = computed(
  () => routerReady.value && !route.meta.guestOnly && !route.meta.standalone
)
const settingsContext = computed(() => {
  const capabilities = runtimeStore.capabilities
  if (!capabilities) return null
  if (!capabilities.requiresAuth) return 'local'
  return authStore.user ? `user:${authStore.user.id}` : null
})
const applicationContextReady = computed(
  () =>
    applicationRoute.value &&
    runtimeStore.capabilities !== null &&
    (!runtimeStore.capabilities?.requiresAuth || authStore.authenticated)
)
const applicationSettingsReady = computed(
  () =>
    settingsStore.isBackendReady &&
    settingsContext.value !== null &&
    loadedSettingsContext.value === settingsContext.value
)
const applicationReady = computed(
  () => applicationContextReady.value && applicationSettingsReady.value
)
const routeContentReady = computed(
  () => routerReady.value && (!applicationRoute.value || applicationReady.value)
)
const bootstrapError = computed(() => {
  if (!routerReady.value || !applicationRoute.value) return ''
  if (!runtimeStore.capabilities) return '无法读取运行配置，请确认后端服务可以访问。'
  if (runtimeStore.capabilities.requiresAuth && !authStore.authenticated) {
    return '登录状态已失效，请重新登录。'
  }
  if (!settingsLoading.value && !settingsStore.isBackendReady && settingsStore.backendError) {
    return `应用设置加载失败：${settingsStore.backendError}`
  }
  return ''
})
const bootstrapTitle = computed(() => (bootstrapError.value ? '应用暂时无法启动' : '正在准备应用'))
const bootstrapDescription = computed(() => {
  if (bootstrapError.value) return bootstrapError.value
  if (!routerReady.value || !applicationContextReady.value) return '正在确认运行环境与登录状态。'
  return '正在读取已保存的应用设置。'
})
let appMounted = false

settingsStore.initSettings()

async function syncTaskCenterLifecycle() {
  if (!appMounted || !applicationReady.value || readerImmersiveMode.value) {
    taskCenterStore.close()
    taskCenterStore.disconnect()
    return
  }

  await taskCenterStore.initialize()
  if (!appMounted || !applicationReady.value || readerImmersiveMode.value) {
    taskCenterStore.disconnect()
  }
}

async function activateApplication(): Promise<void> {
  if (!appMounted || !routerReady.value || !applicationContextReady.value) {
    taskCenterStore.disconnect()
    return
  }

  const context = settingsContext.value
  if (!context) return
  if (!applicationSettingsReady.value) {
    if (settingsLoading.value) return
    settingsLoading.value = true
    const loaded = await settingsStore.loadFromBackend()
    settingsLoading.value = false
    if (!appMounted) return
    if (settingsContext.value !== context) {
      void activateApplication()
      return
    }
    if (!loaded) {
      taskCenterStore.disconnect()
      return
    }
    loadedSettingsContext.value = context
  }

  await syncTaskCenterLifecycle()
}

function retryBootstrap(): void {
  if (!runtimeStore.capabilities) {
    window.location.reload()
    return
  }
  if (runtimeStore.capabilities.requiresAuth && !authStore.authenticated) {
    void router.replace({ name: 'login', query: { redirect: route.fullPath } })
    return
  }
  void activateApplication()
}

function handleAuthenticationRequired(): void {
  authStore.markUnauthenticated()
  if (runtimeStore.capabilities?.requiresAuth && !route.meta.guestOnly) {
    void router.replace({ name: 'login', query: { redirect: route.fullPath } })
  }
}

onMounted(async () => {
  appMounted = true
  window.addEventListener('saber:authentication-required', handleAuthenticationRequired)
  await router.isReady()
  if (appMounted) routerReady.value = true
})

watch([routerReady, readerImmersiveMode, applicationContextReady, settingsContext], () => {
  void activateApplication()
})

onBeforeUnmount(() => {
  appMounted = false
  window.removeEventListener('saber:authentication-required', handleAuthenticationRequired)
  taskCenterStore.disconnect()
})
</script>

<template>
  <main v-if="!routeContentReady" class="app-bootstrap">
    <ProductEmptyState
      :title="bootstrapTitle"
      :description="bootstrapDescription"
      :role="bootstrapError ? 'note' : 'status'"
      :aria-live="bootstrapError ? 'assertive' : 'polite'"
    >
      <template v-if="!bootstrapError" #icon>
        <UiSpinner :decorative="false" label="正在准备应用" size="28" />
      </template>
      <template v-if="bootstrapError" #actions>
        <UiButton variant="primary" @click="retryBootstrap">重试</UiButton>
      </template>
    </ProductEmptyState>
  </main>
  <RouterView v-else />
  <TaskCenterLauncher v-if="applicationReady && !readerImmersiveMode" />
  <TaskCenterDrawer v-if="applicationReady && !readerImmersiveMode" />
  <AccountDock
    v-if="runtimeStore.capabilities?.requiresAuth && applicationReady && !readerImmersiveMode"
  />
  <ProductConfirmProvider />
  <ProductTextInputProvider />
  <ToastNotification />
</template>

<style scoped>
.app-bootstrap {
  display: grid;
  min-height: 100dvh;
  place-items: center;
  padding: 24px;
  background: var(--color-surface-base);
}

.app-bootstrap .product-empty-state {
  --product-empty-state-min-height: auto;

  width: min(100%, 520px);
  padding: 48px 24px;
}
</style>
