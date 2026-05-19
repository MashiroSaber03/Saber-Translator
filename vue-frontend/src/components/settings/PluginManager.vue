<template>
  <div class="plugin-manager">
    <!-- 插件列表 -->
    <div class="settings-group">
      <div class="settings-group-header">
        <div class="settings-group-title">已安装插件</div>
        <div class="plugin-header-actions">
          <button class="btn btn-sm" :disabled="isImporting" @click="triggerImport">
            {{ isImporting ? '导入中...' : '导入插件' }}
          </button>
          <button class="btn btn-sm" @click="showAgentModal = true">
            自动生成插件
          </button>
          <button class="btn btn-sm" :disabled="isRefreshing" @click="refreshPluginList">
            {{ isRefreshing ? '刷新中...' : '刷新插件' }}
          </button>
        </div>
      </div>
      <input
        ref="pluginImportInputRef"
        type="file"
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
              <input type="checkbox" :checked="plugin.enabled" @change="togglePlugin(plugin)" />
              <span class="slider"></span>
            </label>
            <button class="btn btn-sm" @click="downloadPlugin(plugin)" title="导出">导出</button>
            <button class="btn btn-sm" @click="openPluginConfig(plugin)" v-if="plugin.has_config" title="配置">⚙️</button>
            <button class="btn btn-sm btn-danger" @click="deletePlugin(plugin)" title="删除">🗑️</button>
          </div>
        </div>
      </div>
    </div>

    <!-- 默认启用状态设置 -->
    <div class="settings-group">
      <div class="settings-group-title">默认启用状态</div>
      <p class="settings-hint">设置插件在新会话中的默认启用状态</p>
      <div v-for="plugin in plugins" :key="'default-' + plugin.id" class="default-state-item">
        <span class="plugin-name">{{ plugin.display_name }}</span>
        <label class="switch">
          <input type="checkbox" :checked="defaultStates[plugin.id]" @change="updateDefaultState(plugin.id, $event)" />
          <span class="slider"></span>
        </label>
      </div>
    </div>

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
                  <input type="checkbox" :id="'config-' + key" v-model="configValues[key]" />
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
                <input
                  type="number"
                  class="config-input"
                  :id="'config-' + key"
                  v-model.number="configValues[key]"
                  :min="field.min"
                  :max="field.max"
                />
              </template>
              <template v-else>
                <input
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
          <button class="btn btn-secondary" @click="closeConfigModal">取消</button>
          <button class="btn btn-primary" @click="savePluginConfig">保存</button>
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

<style scoped>
.settings-group-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}

.plugin-header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.plugin-list {
  border: 1px solid var(--border-color);
  border-radius: 4px;
}

.plugin-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 15px;
  border-bottom: 1px solid var(--border-color);
}

.plugin-item:last-child {
  border-bottom: none;
}

.plugin-info {
  flex: 1;
}

.plugin-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 4px;
}

.plugin-name {
  font-weight: 500;
}

.plugin-version {
  font-size: 12px;
  color: var(--text-secondary);
}

.plugin-description {
  font-size: 13px;
  color: var(--text-secondary);
  margin: 0;
}

.plugin-meta {
  margin: 4px 0 0;
  font-size: 12px;
  color: var(--text-secondary);
}

.plugin-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

/* 开关样式 */
.switch {
  position: relative;
  display: inline-block;
  width: 40px;
  height: 22px;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  cursor: pointer;
  inset: 0;
  background-color: var(--bg-tertiary);
  transition: 0.3s;
  border-radius: 22px;
}

.slider::before {
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

input:checked + .slider {
  background-color: var(--color-primary);
}

input:checked + .slider::before {
  transform: translateX(18px);
}

/* 默认状态设置 */
.default-state-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid var(--border-color);
}

.default-state-item:last-child {
  border-bottom: none;
}

/* 配置模态框 */
.plugin-config-modal {
  position: fixed;
  inset: 0;
  background: rgb(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: var(--z-popover);
  padding: 20px;
}

.plugin-config-content {
  background: var(--card-bg-color, #fff);
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 16px;
  box-shadow: 0 24px 60px rgb(15, 23, 42, 0.26);
  width: 90%;
  max-width: 620px;
  max-height: 80vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.plugin-config-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  background: linear-gradient(180deg, rgb(248, 250, 252, 0.96) 0%, rgb(255, 255, 255, 0.98) 100%);
  border-bottom: 1px solid var(--border-color);
}

.plugin-config-header h4 {
  margin: 0;
  font-size: 1.18rem;
  font-weight: 700;
  color: var(--text-color, #1a202c);
}

.close-btn {
  width: 36px;
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  font-size: 24px;
  cursor: pointer;
  color: var(--text-secondary);
  transition: background-color var(--transition-fast), color var(--transition-fast);
}

.close-btn:hover {
  background: rgb(15, 23, 42, 0.06);
  color: var(--text-color, #1a202c);
}

.plugin-config-body {
  background:
    linear-gradient(180deg, rgb(248, 250, 252, 0.52) 0%, rgb(255, 255, 255, 0.92) 100%),
    var(--card-bg-color, #fff);
  padding: 20px 24px 24px;
  overflow-y: auto;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.config-field {
  padding: 16px 18px;
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 14px;
  background: var(--input-bg-color, #f9fafc);
  box-shadow: 0 8px 24px rgb(15, 23, 42, 0.04);
}

.config-field-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.config-field-label {
  display: block;
  font-size: 1rem;
  font-weight: 700;
  color: var(--text-color, #1a202c);
}

.config-field-key {
  flex-shrink: 0;
  padding: 4px 8px;
  border-radius: 999px;
  background: rgb(37, 99, 235, 0.08);
  color: rgb(37, 99, 235);
  font-size: 12px;
  font-family: var(--font-mono, 'Consolas', monospace);
}

.config-field-control {
  display: flex;
  align-items: center;
  min-height: 44px;
}

.config-input {
  width: 100%;
  min-height: 44px;
  padding: 10px 12px;
  border: 1px solid var(--input-border-color, #d5deea);
  border-radius: 10px;
  background: var(--card-bg-color, #fff);
  color: var(--text-color, #1a202c);
  font-size: 14px;
  line-height: 1.4;
  transition: border-color var(--transition-fast), box-shadow var(--transition-fast), background-color var(--transition-fast);
}

.config-input:hover {
  border-color: rgb(140, 158, 255);
}

.config-input:focus {
  outline: none;
  border-color: rgb(91, 115, 242);
  box-shadow: 0 0 0 3px rgb(91, 115, 242, 0.14);
  background: #fff;
}

.config-select-wrap {
  width: 100%;
}

.config-switch {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  cursor: pointer;
  user-select: none;
}

.config-switch input {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}

.config-switch-track {
  position: relative;
  width: 46px;
  height: 26px;
  border-radius: 999px;
  background: #cbd5e1;
  transition: background-color var(--transition-fast);
}

.config-switch-track::after {
  content: '';
  position: absolute;
  top: 3px;
  left: 3px;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #fff;
  box-shadow: 0 2px 6px rgb(15, 23, 42, 0.18);
  transition: transform var(--transition-fast);
}

.config-switch input:checked + .config-switch-track {
  background: linear-gradient(135deg, rgb(37, 99, 235) 0%, rgb(99, 102, 241) 100%);
}

.config-switch input:checked + .config-switch-track::after {
  transform: translateX(20px);
}

.config-switch-text {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-color, #1a202c);
}

.field-description {
  margin: 12px 0 0;
  font-size: 13px;
  line-height: 1.6;
  color: var(--text-secondary);
}

.plugin-config-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 18px 24px 22px;
  background: linear-gradient(0deg, rgb(248, 250, 252, 0.9) 0%, rgb(255, 255, 255, 0.98) 100%);
  border-top: 1px solid var(--border-color);
}

.loading-hint,
.empty-hint {
  padding: 20px;
  text-align: center;
  color: var(--text-secondary);
}

.settings-hint {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 10px;
}

.btn-sm {
  padding: 4px 8px;
  font-size: 12px;
}

.btn-danger {
  background: transparent;
  border: none;
}

.sr-only {
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
