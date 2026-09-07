import { createApp } from 'vue'
import PanelApp from './PanelApp.vue'
import { saberRequest } from './api'
import type { PluginSettingsApi } from '../../vue-frontend/src/types/browserExtensionSettings'
import './panel.css'

const systemTheme = window.matchMedia('(prefers-color-scheme: dark)')
const applyTheme = () => {
  document.documentElement.dataset.theme = systemTheme.matches ? 'dark' : 'light'
}
applyTheme()
systemTheme.addEventListener('change', applyTheme)

const api: PluginSettingsApi = (path, method = 'GET', body) =>
  saberRequest(
    `/manage${path}`,
    {
      method,
      headers: method === 'GET' ? {} : { 'Idempotency-Key': crypto.randomUUID() },
      ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
    },
    undefined,
    120_000
  )
createApp(PanelApp, { api }).mount('#app')
