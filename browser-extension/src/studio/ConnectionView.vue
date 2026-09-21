<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'
import { STORAGE_KEY } from '../storage'
import type { BackgroundRequest, BackgroundResponse, ExtensionSettings } from '../types'
import type { StudioAction, StudioState } from './protocol'

const props = defineProps<{
  active: boolean
  state: StudioState | null
  request: (action: StudioAction) => Promise<unknown>
}>()
const emit = defineEmits<{ saved: [] }>()
const token = ref('')
const port = ref(5000)
const hostname = ref('')
const busy = ref(false)
const message = ref('')
const failed = ref(false)
let saved: { token: string; port: number } | undefined

async function send<T>(request: BackgroundRequest): Promise<T> {
  const response: BackgroundResponse<T> = await chrome.runtime.sendMessage(request)
  if (!response?.ok) throw new Error(response?.error?.message ?? '扩展后台没有响应')
  return response.data
}
async function checkConnection(save = false) {
  busy.value = true
  failed.value = false
  message.value = ''
  try {
    if (save) {
      await send({ type: 'save-connection', token: token.value, serverPort: port.value })
      token.value = token.value.trim()
      saved = { token: token.value, port: port.value }
      emit('saved')
    } else {
      const connection = await send<{ token: string; serverPort: number; hostname: string }>({ type: 'get-connection-state' })
      hostname.value = connection.hostname
      if (saved && (token.value !== saved.token || port.value !== saved.port)) {
        message.value = '连接信息尚未保存，点击「保存并连接」后生效。'
        return
      }
      token.value = connection.token
      port.value = connection.serverPort
      saved = { token: token.value, port: port.value }
    }
    if (save) message.value = '连接信息已保存，连接状态见顶部。'
  } catch (error) {
    failed.value = true
    message.value = (error as Error).message
  } finally {
    busy.value = false
  }
}
watch(() => props.active, active => {
  if (active && !busy.value) void checkConnection()
}, { immediate: true })

function connectionChanged(changes: Record<string, chrome.storage.StorageChange>, area: string) {
  const change = changes[STORAGE_KEY]
  const previous = change?.oldValue as ExtensionSettings | undefined
  const current = change?.newValue as ExtensionSettings | undefined
  if (area === 'local' && change && props.active && !busy.value
    && (current?.token !== previous?.token || current?.serverPort !== previous?.serverPort)) {
    void checkConnection()
  }
}
chrome.storage.onChanged.addListener(connectionChanged)
onBeforeUnmount(() => chrome.storage.onChanged.removeListener(connectionChanged))

async function toggleSite() {
  busy.value = true
  try {
    await props.request(props.state?.preference.disabled ? 'enable' : 'disable')
  } catch (error) {
    failed.value = true
    message.value = (error as Error).message
  } finally {
    busy.value = false
  }
}
</script>
<template>
  <div class="view-heading"><h2>连接与站点</h2></div>
  <div v-if="message" class="notice" :class="{ error: failed }" role="status">{{ message }}</div>
  <form class="setting-group" @submit.prevent="checkConnection(true)">
    <p class="muted">先在 GUI「概览」启动后端，再在「设置」中允许扩展连接并复制配对令牌。</p>
    <label class="field">配对令牌
      <input v-model="token" type="password" autocomplete="off" required minlength="32" maxlength="200" placeholder="从 Saber GUI 复制令牌" :disabled="busy" />
    </label>
    <label class="field">本机端口
      <input v-model.number="port" type="number" min="1" max="65535" step="1" required :disabled="busy" />
    </label>
    <button class="button primary grow" type="submit" :disabled="busy">{{ busy ? '正在连接…' : '保存并连接' }}</button>
  </form>
  <section v-if="state && hostname" class="setting-group">
    <h3>{{ hostname }}</h3>
    <p class="muted">{{ state.preference.disabled ? '该站点已停用，重新启用后可翻译图片。' : '该站点已启用。停用后隐藏悬浮按钮，可通过工具栏图标重新开启。' }}</p>
    <button class="button" :disabled="busy" @click="toggleSite">{{ state.preference.disabled ? '重新启用当前网站' : '停用当前网站' }}</button>
  </section>
</template>
