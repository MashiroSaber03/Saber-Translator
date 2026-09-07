<template>
  <div>
    <p v-if="notice" role="status" :class="{ 'browser-settings-error': hasError }">{{ notice }}</p>
    <div ref="container" />
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { apiClient } from '@/api/client'
import {
  createPluginSettingsEditor,
  type PluginSettingsApi,
} from '../../../../src/shared/browserExtensionSettings'

const emit = defineEmits<{ (event: 'saving', value: boolean): void }>()
const container = ref<HTMLElement>()
const notice = ref('正在读取插件配置…')
const hasError = ref(false)
let editor: ReturnType<typeof createPluginSettingsEditor> | undefined

const api: PluginSettingsApi = async <T,>(path: string, method = 'GET', body?: unknown) => {
  const url = `/api/v2${path}`
  const config = {
    params: path === '/fonts' ? {} : { scope: 'browser_extension' },
    headers: method === 'GET' ? {} : { 'Idempotency-Key': crypto.randomUUID() },
  }
  try {
    if (method === 'GET') return await apiClient.get<T>(url, config)
    if (method === 'PUT') return await apiClient.put<T>(url, body, config)
    return await apiClient.post<T>(url, body, config)
  } catch (error) {
    if ((error as { status?: number }).status === 409)
      throw new Error('插件配置已在其他窗口变化，请重新读取后再修改。')
    throw error
  }
}

onMounted(async () => {
  editor = createPluginSettingsEditor(
    container.value!,
    api,
    (text, error = false) => {
      notice.value = text
      hasError.value = error
    },
    saving => emit('saving', saving)
  )
  try {
    await editor.load()
    notice.value = ''
  } catch (error) {
    notice.value = error instanceof Error ? error.message : '读取插件配置失败'
    hasError.value = true
  }
})

onBeforeUnmount(() => editor?.dispose())
defineExpose({ save: () => editor?.save() ?? Promise.resolve(true) })
</script>

<style scoped>
.browser-settings-error {
  color: var(--color-text-danger, #ad2542);
}
</style>
