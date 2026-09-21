<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { STORAGE_KEY } from '../storage'
import { isStoreInstallation, UPDATE_KEY } from '../updates'
import type { BackgroundResponse, ExtensionSettings } from '../types'

const props = defineProps<{ active: boolean }>()
defineEmits<{ configure: [] }>()
const connection = ref<'checking' | 'online' | 'offline'>('checking')
const connectionError = ref('')
const version = chrome.runtime.getManifest().version
const store = ref(false)
const pendingVersion = ref('')
const expanded = ref(false)
const checkingUpdate = ref(false)
const updateMessage = ref('')
let connectionGeneration = 0
async function checkConnection() {
  const generation = ++connectionGeneration
  try {
    const response: BackgroundResponse<unknown> = await chrome.runtime.sendMessage({ type: 'status' })
    if (!response?.ok) throw new Error(response?.error?.message ?? '扩展后台没有响应')
    if (generation === connectionGeneration) {
      connection.value = 'online'
      connectionError.value = ''
    }
  } catch (error) {
    if (generation !== connectionGeneration) return
    connection.value = 'offline'
    connectionError.value = (error as Error).message
  }
}
async function checkUpdate() {
  if (!store.value || checkingUpdate.value) return
  checkingUpdate.value = true
  updateMessage.value = ''
  try {
    const result = await chrome.runtime.requestUpdateCheck()
    if (result.status === 'update_available') {
      pendingVersion.value = result.version ?? ''
      await chrome.storage.session.set({ [UPDATE_KEY]: pendingVersion.value })
    } else {
      updateMessage.value = result.status === 'throttled'
        ? '检查过于频繁，请稍后再试。' : '暂未发现可用更新。'
    }
  } catch {
    updateMessage.value = '检查更新失败，请稍后重试。'
  } finally {
    checkingUpdate.value = false
  }
}
function storageChanged(changes: Record<string, chrome.storage.StorageChange>, area: string) {
  if (area === 'session' && changes[UPDATE_KEY]) pendingVersion.value = String(changes[UPDATE_KEY].newValue ?? '')
  const change = changes[STORAGE_KEY]
  const oldValue = change?.oldValue as ExtensionSettings | undefined
  const newValue = change?.newValue as ExtensionSettings | undefined
  if (area === 'local' && change && props.active
    && (oldValue?.token !== newValue?.token || oldValue?.serverPort !== newValue?.serverPort)) void checkConnection()
}
function focus() { if (props.active) void checkConnection() }
defineExpose({ checkConnection })
watch(() => props.active, active => { if (active) void checkConnection() }, { immediate: true })
onMounted(async () => {
  chrome.storage.onChanged.addListener(storageChanged)
  window.addEventListener('focus', focus)
  try {
    store.value = await isStoreInstallation()
    if (store.value) pendingVersion.value = String((await chrome.storage.session.get(UPDATE_KEY))[UPDATE_KEY] ?? '')
  } catch { /* 无法确认安装来源时不提供商店更新入口。 */ }
})
onBeforeUnmount(() => {
  connectionGeneration++
  chrome.storage.onChanged.removeListener(storageChanged)
  window.removeEventListener('focus', focus)
})
</script>
<template>
  <section class="studio-status" aria-label="插件状态">
    <div class="status-line">
      <span role="status"><i class="status-dot" :class="{ offline: connection !== 'online' }" aria-hidden="true" />
        {{ connection === 'online' ? '已连接 Saber' : connection === 'checking' ? '正在连接…' : '未连接 Saber' }}</span>
      <button v-if="connection === 'offline'" class="text-button" @click="$emit('configure')">设置连接</button>
      <button class="text-button version-button" :aria-expanded="expanded" @click="expanded = !expanded">
        {{ pendingVersion ? '有新版可用' : `v${version}` }}
      </button>
    </div>
    <div v-if="connection === 'offline'" class="connection-detail">
      <p>{{ connectionError }}</p><button class="text-button" @click="checkConnection">重试连接</button>
    </div>
    <div v-if="expanded" class="version-detail">
      <p>当前版本 {{ version }} · {{ store ? '由浏览器自动更新' : '非商店安装' }}</p>
      <p v-if="pendingVersion" role="status">新版本 {{ pendingVersion }} 已就绪，等待浏览器安装。完成当前阅读后，可重启浏览器使更新生效。</p>
      <p v-else-if="updateMessage" role="status">{{ updateMessage }}</p>
      <button v-if="store" class="text-button" :disabled="checkingUpdate || !!pendingVersion" @click="checkUpdate">{{ checkingUpdate ? '正在检查…' : '检查更新' }}</button>
    </div>
  </section>
</template>
