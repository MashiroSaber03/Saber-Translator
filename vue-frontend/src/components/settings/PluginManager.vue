<template>
  <div class="plugin-manager">
    <!-- 插件列表 -->
    <UiPanel variant="settings">
      <template #title>
        <span>已安装插件</span>
        <div class="plugin-header-actions">
          <UiButton variant="secondary" :disabled="isImporting" @click="triggerImport" size="sm">
            {{ isImporting ? '导入中...' : '导入插件' }}
          </UiButton>
          <UiButton variant="secondary" @click="showAgentModal = true" size="sm">
            自动生成插件
          </UiButton>
          <UiButton variant="secondary" :disabled="isRefreshing" @click="refreshPluginList" size="sm">
            {{ isRefreshing ? '刷新中...' : '刷新插件' }}
          </UiButton>
        </div>
      </template>
      <UiFileInput
        ref="pluginImportInputRef"
        accept=".zip,application/zip"
        class="sr-only"
        @change="handleImportFileChange"
      />
      <div v-if="isLoading" class="loading-hint">加载中...</div>
      <div v-else-if="plugins.length === 0" class="empty-hint">暂无已安装的插件</div>
      <div v-else class="plugin-list">
        <div v-for="plugin in plugins" :key="plugin.id" class="plugin-item">
          <div class="plugin-info">
            <div class="plugin-header">
              <span class="plugin-name">{{ plugin.display_name }}</span>
              <span class="plugin-version">v{{ plugin.version || '1.0.0' }}</span>
            </div>
            <p class="plugin-description">{{ plugin.description || '暂无描述' }}</p>
            <p class="plugin-meta">步骤: {{ (plugin.supported_steps || []).join(', ') || '无' }}</p>
            <p class="plugin-meta">模式: {{ (plugin.supported_modes || []).join(', ') || '无' }}</p>
          </div>
          <div class="plugin-controls">
            <label class="switch">
              <UiInput type="checkbox" :checked="plugin.enabled" @change="togglePlugin(plugin)" />
              <span class="slider"></span>
            </label>
            <UiButton variant="secondary" @click="downloadPlugin(plugin)" title="导出" size="sm">导出</UiButton>
            <UiButton variant="secondary" @click="openPluginConfig(plugin)" v-if="plugin.has_config" title="配置" size="sm">⚙️</UiButton>
            <UiButton variant="danger" @click="deletePlugin(plugin)" title="删除" size="sm">🗑️</UiButton>
          </div>
        </div>
      </div>
    </UiPanel>

    <!-- 默认启用状态设置 -->
    <UiPanel variant="settings">
      <template #title>默认启用状态</template>
      <p class="settings-hint">设置插件在新会话中的默认启用状态</p>
      <div v-for="plugin in plugins" :key="'default-' + plugin.id" class="default-state-item">
        <span class="plugin-name">{{ plugin.display_name }}</span>
        <label class="switch">
          <UiInput type="checkbox" :checked="defaultStates[plugin.id]" @change="updateDefaultState(plugin.id, $event)" />
          <span class="slider"></span>
        </label>
      </div>
    </UiPanel>

    <!-- 插件配置模态框 -->
    <div
      v-if="showConfigModal"
      ref="configModalOverlayRef"
      class="plugin-config-modal"
      @mousedown.self="handleConfigModalOverlayMouseDown"
    >
      <div class="plugin-config-content">
        <div class="plugin-config-header">
          <h4>{{ configPlugin?.display_name }} 配置</h4>
          <span class="close-btn" @click="closeConfigModal">&times;</span>
        </div>
        <div class="plugin-config-body">
          <div v-for="(field, key) in configSchema" :key="key" class="config-field" :class="`field-${field.type}`">
            <div class="config-field-head">
              <label :for="'config-' + key" class="config-field-label">{{ field.label || key }}</label>
              <span class="config-field-key">{{ key }}</span>
            </div>
            <div class="config-field-control">
              <template v-if="field.type === 'boolean'">
                <label class="config-switch">
                  <UiInput type="checkbox" :id="'config-' + key" v-model="configValues[key]" />
                  <span class="config-switch-track"></span>
                  <span class="config-switch-text">{{ configValues[key] ? '启用' : '禁用' }}</span>
                </label>
              </template>
              <template v-else-if="field.type === 'select'">
                <div class="config-select-wrap">
                  <CustomSelect
                    :model-value="String(configValues[key] ?? '')"
                    :options="field.options || []"
                    @change="(v: string | number) => { configValues[key] = v }"
                  />
                </div>
              </template>
              <template v-else-if="field.type === 'number'">
                <UiInput
                  type="number"
                  class="config-input"
                  :id="'config-' + key"
                  v-model.number="configValues[key]"
                  :min="field.min"
                  :max="field.max"
                />
              </template>
              <template v-else>
                <UiInput
                  type="text"
                  class="config-input"
                  :id="'config-' + key"
                  v-model="configValues[key]"
                  :placeholder="field.placeholder"
                />
              </template>
            </div>
            <p v-if="field.description" class="field-description">{{ field.description }}</p>
          </div>
        </div>
        <div class="plugin-config-footer">
          <UiButton variant="secondary" @click="closeConfigModal">取消</UiButton>
          <UiButton variant="primary" @click="savePluginConfig">保存</UiButton>
        </div>
      </div>
    </div>

    <PluginAgentModal
      v-model="showAgentModal"
      @plugins-changed="handlePluginAgentRefresh"
    />
  </div>
</template>

<script setup lang="ts">
import UiPanel from '@/components/ui/UiPanel.vue'

import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiInput from '@/components/ui/UiInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
/**
 * 插件管理组件
 * 管理插件的刷新、启用/禁用、配置和删除
 */
import { ref, onMounted } from 'vue'
import * as pluginApi from '@/api/plugin'
import type { PluginData } from '@/types'
import { useToast } from '@/utils/toast'
import CustomSelect from '@/components/common/CustomSelect.vue'
import PluginAgentModal from '@/components/settings/PluginAgentModal.vue'
import { useOverlayDismiss } from '@/composables/useOverlayDismiss'

type Plugin = PluginData

// 配置字段接口
interface ConfigField {
  type: string
  label?: string
  description?: string
  placeholder?: string
  options?: { value: string; label: string }[]
  min?: number
  max?: number
}

// Toast
const toast = useToast()

// 状态
const plugins = ref<Plugin[]>([])
const defaultStates = ref<Record<string, boolean>>({})
const isLoading = ref(false)
const isRefreshing = ref(false)
const isImporting = ref(false)
const pluginImportInputRef = ref<HTMLInputElement | null>(null)

// 配置模态框状态
const showConfigModal = ref(false)
const configPlugin = ref<Plugin | null>(null)
const configSchema = ref<Record<string, ConfigField>>({})
const configValues = ref<Record<string, unknown>>({})
const showAgentModal = ref(false)
const {
  overlayRef: configModalOverlayRef,
  handleOverlayMouseDown: handleConfigModalOverlayMouseDown,
} = useOverlayDismiss(closeConfigModal, {
  enabled: showConfigModal,
})

// 加载插件列表
async function loadPlugins() {
  isLoading.value = true
  try {
    const result = await pluginApi.getPlugins()
    plugins.value = result.plugins || []
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '加载插件列表失败'
    toast.error(errorMessage)
  } finally {
    isLoading.value = false
  }
}

// 加载默认状态
async function loadDefaultStates() {
  try {
    const result = await pluginApi.getPluginDefaultStates()
    defaultStates.value = result.default_states || {}
  } catch (error: unknown) {
    console.error('加载默认状态失败:', error)
  }
}

// 刷新插件列表并触发后端热重载
async function refreshPluginList() {
  await refreshPluginListCore({ showToast: true })
}

async function refreshPluginListCore(options: { showToast: boolean }) {
  isRefreshing.value = true
  closeConfigModal()
  try {
    const result = await pluginApi.refreshPlugins()
    plugins.value = result.plugins || []
    defaultStates.value = result.default_states || {}

    if (options.showToast) {
      if (result.partial_success) {
        const failedCount = result.summary?.failed ?? result.failures?.length ?? 0
        toast.warning(
          failedCount > 0
            ? `部分插件刷新失败（${failedCount} 个）`
            : '部分插件刷新失败'
        )
      } else {
        toast.success('插件列表已刷新')
      }
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '刷新插件失败'
    toast.error(errorMessage)
  } finally {
    isRefreshing.value = false
  }
}

// 切换插件启用状态
async function togglePlugin(plugin: Plugin) {
  try {
    if (plugin.enabled) {
      await pluginApi.disablePlugin(plugin.id)
      plugin.enabled = false
      toast.success(`已禁用 ${plugin.display_name}`)
    } else {
      await pluginApi.enablePlugin(plugin.id)
      plugin.enabled = true
      toast.success(`已启用 ${plugin.display_name}`)
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '操作失败'
    toast.error(errorMessage)
  }
}

// 设置默认启用状态
async function updateDefaultState(pluginName: string, event: Event) {
  const target = event.target as HTMLInputElement
  const enabled = target.checked
  try {
    await pluginApi.setPluginDefaultState(pluginName, enabled)
    defaultStates.value[pluginName] = enabled
    toast.success('默认状态已更新')
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '设置失败'
    toast.error(errorMessage)
    // 恢复原状态
    target.checked = !enabled
  }
}

// 打开插件配置
async function openPluginConfig(plugin: Plugin) {
  configPlugin.value = plugin
  try {
    // 获取配置规范
    const schemaResult = await pluginApi.getPluginConfigSchema(plugin.id)
    configSchema.value = (schemaResult.schema || {}) as Record<string, ConfigField>

    // 获取当前配置
    const configResult = await pluginApi.getPluginConfig(plugin.id)
    configValues.value = configResult.config || {}

    showConfigModal.value = true
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '加载配置失败'
    toast.error(errorMessage)
  }
}

// 关闭配置模态框
function closeConfigModal() {
  showConfigModal.value = false
  configPlugin.value = null
  configSchema.value = {}
  configValues.value = {}
}

// 保存插件配置
async function savePluginConfig() {
  if (!configPlugin.value) return
  try {
    await pluginApi.savePluginConfig(configPlugin.value.id, configValues.value)
    toast.success('配置保存成功')
    closeConfigModal()
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '保存配置失败'
    toast.error(errorMessage)
  }
}

// 删除插件
async function deletePlugin(plugin: Plugin) {
  if (!confirm(`确定要删除插件 "${plugin.display_name}" 吗？`)) {
    return
  }
  try {
    await pluginApi.deletePlugin(plugin.id)
    toast.success('插件删除成功')
    await loadPlugins()
    await loadDefaultStates()
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '删除插件失败'
    toast.error(errorMessage)
  }
}

function triggerImport() {
  pluginImportInputRef.value?.click()
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  URL.revokeObjectURL(url)
}

async function downloadPlugin(plugin: Plugin) {
  try {
    const result = await pluginApi.exportPlugin(plugin.id)
    downloadBlob(result.blob, result.filename)
    toast.success(`已导出 ${plugin.display_name}`)
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '导出插件失败'
    toast.error(errorMessage)
  }
}

async function importPluginFile(file: File, replace = false) {
  return pluginApi.importPlugin(file, replace)
}

async function handleImportFileChange(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return

  isImporting.value = true
  try {
    await importPluginFile(file, false)
    await refreshPluginListCore({ showToast: false })
    toast.success('插件导入成功')
  } catch (error: unknown) {
    const conflictError = error as { status?: number; details?: Record<string, unknown>; message?: string }
    if (conflictError?.status === 409) {
      const pluginId = String(conflictError.details?.plugin_id || '')
      const confirmed = confirm(`插件 "${pluginId || file.name}" 已存在，是否替换？`)
      if (confirmed) {
        await importPluginFile(file, true)
        await refreshPluginListCore({ showToast: false })
        toast.success('插件导入成功')
      }
    } else {
      const errorMessage = error instanceof Error ? error.message : '导入插件失败'
      toast.error(errorMessage)
    }
  } finally {
    target.value = ''
    isImporting.value = false
  }
}

async function handlePluginAgentRefresh() {
  await loadPlugins()
  await loadDefaultStates()
}

// 初始化
onMounted(() => {
  loadPlugins()
  loadDefaultStates()
})
</script>

<style scoped>.plugin-manager .plugin-header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.plugin-manager .plugin-list {
  border: 1px solid var(--color-border-muted);
  border-radius: 4px;
}

.plugin-manager .plugin-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 15px;
  border-bottom: 1px solid var(--color-border-muted);
}

.plugin-manager .plugin-item:last-child {
  border-bottom: none;
}

.plugin-manager .plugin-info {
  flex: 1;
}

.plugin-manager .plugin-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 4px;
}

.plugin-manager .plugin-name {
  font-weight: 500;
}

.plugin-manager .plugin-version {
  font-size: 12px;
  color: var(--color-text-supporting);
}

.plugin-manager .plugin-description {
  font-size: 13px;
  color: var(--color-text-supporting);
  margin: 0;
}

.plugin-manager .plugin-meta {
  margin: 4px 0 0;
  font-size: 12px;
  color: var(--color-text-supporting);
}

.plugin-manager .plugin-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

/* 开关样式 */
.plugin-manager .switch {
  position: relative;
  display: inline-block;
  width: 40px;
  height: 22px;
}

.plugin-manager .switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.plugin-manager .slider {
  position: absolute;
  cursor: pointer;
  inset: 0;
  background-color: var(--color-surface-muted);
  transition: 0.3s;
  border-radius: 22px;
}

.plugin-manager .slider::before {
  position: absolute;
  content: '';
  height: 16px;
  width: 16px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  transition: 0.3s;
  border-radius: 50%;
}

.plugin-manager input:checked + .slider {
  background-color: var(--color-action-primary);
}

.plugin-manager input:checked + .slider::before {
  transform: translateX(18px);
}

/* 默认状态设置 */
.plugin-manager .default-state-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid var(--color-border-muted);
}

.plugin-manager .default-state-item:last-child {
  border-bottom: none;
}

/* 配置模态框 */
.plugin-manager .plugin-config-modal {
  position: fixed;
  inset: 0;
  background: var(--plugin-manager-surface-base);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: var(--z-popover);
  padding: 20px;
}

.plugin-manager .plugin-config-content {
  background: var(--color-surface-card, var(--color-surface-base));
  border: 1px solid var(--color-border-muted, var(--color-border-muted));
  border-radius: 16px;
  box-shadow: 0 24px 60px var(--plugin-manager-shadow-default);
  width: 90%;
  max-width: 620px;
  max-height: 80vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.plugin-manager .plugin-config-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  background: linear-gradient(180deg, var(--plugin-manager-surface-raised) 0%, var(--plugin-manager-surface-muted) 100%);
  border-bottom: 1px solid var(--color-border-muted);
}

.plugin-manager .plugin-config-header h4 {
  margin: 0;
  font-size: 1.18rem;
  font-weight: 700;
  color: var(--color-text-strong, var(--plugin-manager-text-primary));
}

.plugin-manager .close-btn {
  width: 36px;
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  font-size: 24px;
  cursor: pointer;
  color: var(--color-text-supporting);
  transition: background-color var(--transition-fast), color var(--transition-fast);
}

.plugin-manager .close-btn:hover {
  background: var(--plugin-manager-surface-subtle);
  color: var(--color-text-strong, var(--plugin-manager-text-primary));
}

.plugin-manager .plugin-config-body {
  background:
    linear-gradient(180deg, var(--plugin-manager-accent-primary) 0%, var(--plugin-manager-accent-secondary) 100%),
    var(--color-surface-card, var(--plugin-manager-accent-muted));
  padding: 20px 24px 24px;
  overflow-y: auto;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.plugin-manager .config-field {
  padding: 16px 18px;
  border: 1px solid var(--color-border-muted, var(--color-border-muted));
  border-radius: 14px;
  background: var(--color-surface-input, var(--plugin-manager-surface-hover));
  box-shadow: 0 8px 24px var(--plugin-manager-shadow-raised);
}

.plugin-manager .config-field-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.plugin-manager .config-field-label {
  display: block;
  font-size: 1rem;
  font-weight: 700;
  color: var(--color-text-strong, var(--plugin-manager-text-primary));
}

.plugin-manager .config-field-key {
  flex-shrink: 0;
  padding: 4px 8px;
  border-radius: 999px;
  background: var(--plugin-manager-surface-active);
  color: var(--plugin-manager-text-secondary);
  font-size: 12px;
  font-family: var(--font-mono, 'Consolas', monospace);
}

.plugin-manager .config-field-control {
  display: flex;
  align-items: center;
  min-height: 44px;
}

.plugin-manager .config-input {
  width: 100%;
  min-height: 44px;
  padding: 10px 12px;
  border: 1px solid var(--color-border-input, var(--plugin-manager-border-default));
  border-radius: 10px;
  background: var(--color-surface-card, var(--color-surface-base));
  color: var(--color-text-strong, var(--plugin-manager-text-primary));
  font-size: 14px;
  line-height: 1.4;
  transition: border-color var(--transition-fast), box-shadow var(--transition-fast), background-color var(--transition-fast);
}

.plugin-manager .config-input:hover {
  border-color: var(--plugin-manager-border-strong);
}

.plugin-manager .config-input:focus {
  outline: none;
  border-color: var(--plugin-manager-border-muted);
  box-shadow: 0 0 0 3px var(--plugin-manager-shadow-floating);
  background: var(--color-surface-base);
}

.plugin-manager .config-select-wrap {
  width: 100%;
}

.plugin-manager .config-switch {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  cursor: pointer;
  user-select: none;
}

.plugin-manager .config-switch input {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}

.plugin-manager .config-switch-track {
  position: relative;
  width: 46px;
  height: 26px;
  border-radius: 999px;
  background: var(--plugin-manager-surface-selected);
  transition: background-color var(--transition-fast);
}

.plugin-manager .config-switch-track::after {
  content: '';
  position: absolute;
  top: 3px;
  left: 3px;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--color-surface-base);
  box-shadow: 0 2px 6px var(--plugin-manager-shadow-strong);
  transition: transform var(--transition-fast);
}

.plugin-manager .config-switch input:checked + .config-switch-track {
  background: linear-gradient(135deg, var(--plugin-manager-surface-overlay) 0%, var(--plugin-manager-surface-inverse) 100%);
}

.plugin-manager .config-switch input:checked + .config-switch-track::after {
  transform: translateX(20px);
}

.plugin-manager .config-switch-text {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-strong, var(--plugin-manager-text-primary));
}

.plugin-manager .field-description {
  margin: 12px 0 0;
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-text-supporting);
}

.plugin-manager .plugin-config-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 18px 24px 22px;
  background: linear-gradient(0deg, var(--plugin-manager-surface-contrast) 0%, var(--plugin-manager-surface-muted) 100%);
  border-top: 1px solid var(--color-border-muted);
}

.plugin-manager .loading-hint,
.plugin-manager .empty-hint {
  padding: 20px;
  text-align: center;
  color: var(--color-text-supporting);
}

.plugin-manager .settings-hint {
  font-size: 13px;
  color: var(--color-text-supporting);
  margin-bottom: 10px;
}

.plugin-manager .ui-button--sm {
  padding: 4px 8px;
  font-size: 12px;
}

.plugin-manager .ui-button--danger {
  background: transparent;
  border: none;
}

.plugin-manager .sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
</style>
