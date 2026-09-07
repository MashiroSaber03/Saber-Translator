<template>
  <BrowserExtensionSettingsForm ref="editor" :api="api" @saving="emit('saving', $event)" />
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { apiClient } from '@/api/client'
import BrowserExtensionSettingsForm from './BrowserExtensionSettingsForm.vue'
import type { PluginSettingsApi } from '@/types/browserExtensionSettings'
const emit = defineEmits<{ (event: 'saving', value: boolean): void }>()
const editor = ref<InstanceType<typeof BrowserExtensionSettingsForm>>()
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

defineExpose({ save: () => editor.value?.save() ?? Promise.resolve(true) })
</script>
