<template>
  <div class="plugin-manager">
    <ProductFormSection>
      <template #title>
        <span>已安装插件</span>
        <div class="plugin-manager__header-actions">
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
        class="plugin-manager__import-input"
        @files-change="handleImportFiles"
      />
      <ProductStatusBanner
        v-if="isLoading"
        class="plugin-manager__status"
        tone="info"
        role="status"
        icon-name="loading"
        title="正在加载插件"
      >
        正在读取已安装插件列表...
      </ProductStatusBanner>
      <ProductStatusBanner
        v-else-if="plugins.length === 0"
        class="plugin-manager__status"
        tone="neutral"
        role="note"
        icon-name="settings"
        title="暂无已安装的插件"
      >
        导入插件或使用自动生成插件开始。
      </ProductStatusBanner>
      <div v-else class="plugin-manager__list">
        <ProductRecordCard v-for="plugin in plugins" :key="plugin.id" class="plugin-manager__plugin-card">
          <template #meta>
            <div class="plugin-manager__plugin-header">
              <span class="plugin-manager__plugin-name">{{ plugin.display_name }}</span>
              <span class="plugin-manager__plugin-version">v{{ plugin.version || '1.0.0' }}</span>
            </div>
          </template>

          <template #actions>
            <div class="plugin-manager__plugin-controls">
              <UiSwitch
                :model-value="plugin.enabled"
                :aria-label="`${plugin.enabled ? '禁用' : '启用'}插件 ${plugin.display_name}`"
                @change="setPluginEnabled(plugin, $event)"
              />
              <UiButton variant="secondary" @click="downloadPlugin(plugin)" title="导出" size="sm">导出</UiButton>
              <UiIconButton
                v-if="plugin.has_config"
                variant="soft"
                size="sm"
                :label="`配置插件 ${plugin.display_name}`"
                @click="openPluginConfig(plugin)"
              >
                <UiIcon name="settings" />
              </UiIconButton>
              <UiIconButton
                variant="danger"
                size="sm"
                :label="`删除插件 ${plugin.display_name}`"
                @click="deletePlugin(plugin)"
              >
                <UiIcon name="trash" />
              </UiIconButton>
            </div>
          </template>

          <p class="plugin-manager__plugin-description">{{ plugin.description || '暂无描述' }}</p>
          <p class="plugin-manager__plugin-meta">步骤: {{ (plugin.supported_steps || []).join(', ') || '无' }}</p>
          <p class="plugin-manager__plugin-meta">模式: {{ (plugin.supported_modes || []).join(', ') || '无' }}</p>
        </ProductRecordCard>
      </div>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>默认启用状态</template>
      <p class="plugin-manager__settings-hint">设置插件在新会话中的默认启用状态</p>
      <div v-for="plugin in plugins" :key="'default-' + plugin.id" class="plugin-manager__default-state-item">
        <span class="plugin-manager__plugin-name">{{ plugin.display_name }}</span>
        <UiSwitch
          :model-value="Boolean(defaultStates[plugin.id])"
          :aria-label="`${defaultStates[plugin.id] ? '关闭' : '开启'} ${plugin.display_name} 默认启用状态`"
          @change="updateDefaultState(plugin.id, $event)"
        />
      </div>
    </ProductFormSection>

    <BaseModal
      :model-value="showConfigModal"
      :title="`${configPlugin?.display_name || '插件'} 配置`"
      custom-class="plugin-config-modal"
      frame-variant="outlined"
      divider-variant="soft"
      footer-tone="muted"
      width="90%"
      max-width="620px"
      max-height="80vh"
      body-padding="none"
      scroll-mode="contained"
      body-display="flex"
      body-direction="column"
      body-min-height="0"
      footer-padding="18px 24px 22px"
      @update:model-value="value => { if (!value) closeConfigModal() }"
      @close="closeConfigModal"
    >
      <div class="plugin-manager__config-body">
        <ProductRecordCard
          v-for="(field, key) in configSchema"
          :key="key"
          class="plugin-manager__config-field-card"
        >
          <template #meta>
            <span class="plugin-manager__config-field-key">{{ key }}</span>
          </template>

          <UiField
            variant="settings"
            :label="field.label || key"
            :description="field.description"
            :control-id="'config-' + key"
          >
            <template v-if="field.type === 'boolean'">
              <div class="plugin-manager__config-switch-row">
                <UiSwitch
                  :id="'config-' + key"
                  :model-value="Boolean(configValues[key])"
                  :aria-label="`${field.label || key}：${configValues[key] ? '禁用' : '启用'}`"
                  @change="(value) => { configValues[key] = value }"
                />
                <span class="plugin-manager__config-switch-text">{{ configValues[key] ? '启用' : '禁用' }}</span>
              </div>
            </template>
            <template v-else-if="field.type === 'select'">
              <UiSelect
                :id="'config-' + key"
                :model-value="String(configValues[key] ?? '')"
                :options="field.options || []"
                @change="(v: string | number) => { configValues[key] = v }"
              />
            </template>
            <template v-else-if="field.type === 'number'">
              <UiNumberField
                :input-id="'config-' + key"
                :model-value="getNumberConfigValue(key)"
                nullable
                :min="field.min"
                :max="field.max"
                @update:model-value="value => setConfigValue(key, value)"
              />
            </template>
            <template v-else>
              <UiInput
                type="text"
                :id="'config-' + key"
                v-model="configValues[key]"
                :placeholder="field.placeholder"
              />
            </template>
          </UiField>
        </ProductRecordCard>
      </div>

      <template #footer>
        <ProductActionRow variant="dialog" aria-label="插件配置操作">
          <UiButton variant="secondary" @click="closeConfigModal">取消</UiButton>
          <UiButton variant="primary" @click="savePluginConfig">保存</UiButton>
        </ProductActionRow>
      </template>
    </BaseModal>

    <PluginAgentModal
      v-model="showAgentModal"
      @plugins-changed="handlePluginAgentRefresh"
    />
  </div>
</template>

<script setup lang="ts">
import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import UiField from '@/components/ui/UiField.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'

import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSwitch from '@/components/ui/UiSwitch.vue'

import UiButton from '@/components/ui/UiButton.vue'
import { ref, onMounted } from 'vue'
import * as pluginApi from '@/api/plugin'
import type { PluginData } from '@/types'
import { useToast } from '@/utils/toast'
import { confirmProductAction } from '@/composables/useProductConfirm'
import { triggerBlobDownload } from '@/utils/browserDownload'
import PluginAgentModal from '@/components/settings/PluginAgentModal.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

type Plugin = PluginData

interface ConfigField {
  type: string
  label?: string
  description?: string
  placeholder?: string
  options?: { value: string; label: string }[]
  min?: number
  max?: number
}

const toast = useToast()

const plugins = ref<Plugin[]>([])
const defaultStates = ref<Record<string, boolean>>({})
const isLoading = ref(false)
const isRefreshing = ref(false)
const isImporting = ref(false)
const pluginImportInputRef = ref<InstanceType<typeof UiFileInput> | null>(null)

const showConfigModal = ref(false)
const configPlugin = ref<Plugin | null>(null)
const configSchema = ref<Record<string, ConfigField>>({})
const configValues = ref<Record<string, unknown>>({})
const showAgentModal = ref(false)

function getNumberConfigValue(key: string | number): number | null {
  const value = configValues.value[String(key)]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function setConfigValue(key: string | number, value: unknown): void {
  configValues.value[String(key)] = value
}

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

async function loadDefaultStates() {
  try {
    const result = await pluginApi.getPluginDefaultStates()
    defaultStates.value = result.default_states || {}
  } catch {
    defaultStates.value = {}
  }
}

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

async function setPluginEnabled(plugin: Plugin, enabled: boolean) {
  if (plugin.enabled === enabled) return

  try {
    if (enabled) {
      await pluginApi.enablePlugin(plugin.id)
      plugin.enabled = true
      toast.success(`已启用 ${plugin.display_name}`)
    } else {
      await pluginApi.disablePlugin(plugin.id)
      plugin.enabled = false
      toast.success(`已禁用 ${plugin.display_name}`)
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '操作失败'
    toast.error(errorMessage)
  }
}

async function updateDefaultState(pluginName: string, enabled: boolean) {
  try {
    await pluginApi.setPluginDefaultState(pluginName, enabled)
    defaultStates.value[pluginName] = enabled
    toast.success('默认状态已更新')
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '设置失败'
    toast.error(errorMessage)
    defaultStates.value[pluginName] = !enabled
  }
}

async function openPluginConfig(plugin: Plugin) {
  configPlugin.value = plugin
  try {
    const schemaResult = await pluginApi.getPluginConfigSchema(plugin.id)
    configSchema.value = (schemaResult.schema || {}) as Record<string, ConfigField>

    const configResult = await pluginApi.getPluginConfig(plugin.id)
    configValues.value = configResult.config || {}

    showConfigModal.value = true
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '加载配置失败'
    toast.error(errorMessage)
  }
}

function closeConfigModal() {
  showConfigModal.value = false
  configPlugin.value = null
  configSchema.value = {}
  configValues.value = {}
}

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

async function deletePlugin(plugin: Plugin) {
  const confirmed = await confirmProductAction({
    title: '删除插件',
    message: `确定要删除插件 "${plugin.display_name}" 吗？`,
    confirmText: '删除',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed) return
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

async function downloadPlugin(plugin: Plugin) {
  try {
    const result = await pluginApi.exportPlugin(plugin.id)
    triggerBlobDownload(result.blob, result.filename)
    toast.success(`已导出 ${plugin.display_name}`)
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '导出插件失败'
    toast.error(errorMessage)
  }
}

async function importPluginFile(file: File, replace = false) {
  return pluginApi.importPlugin(file, replace)
}

async function handleImportFiles(files: File[]) {
  const file = files[0]
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
      const confirmed = await confirmProductAction({
        title: '替换插件',
        message: `插件 "${pluginId || file.name}" 已存在，是否替换？`,
        confirmText: '替换',
        cancelText: '取消',
        tone: 'danger',
      })
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
    pluginImportInputRef.value?.clear()
    isImporting.value = false
  }
}

async function handlePluginAgentRefresh() {
  await loadPlugins()
  await loadDefaultStates()
}

onMounted(() => {
  loadPlugins()
  loadDefaultStates()
})
</script>

<style scoped>
.plugin-manager__header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.plugin-manager {
  --plugin-manager-config-body-background-start: var(--color-overlay-inverse-muted);
  --plugin-manager-config-body-background-end: var(--color-overlay-inverse-muted);
  --plugin-manager-config-key-background: color-mix(in srgb, var(--color-action-brand) 8%, transparent);
  --plugin-manager-config-key-text: var(--color-text-brand);
}

.plugin-manager__list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.plugin-manager__plugin-card {
  --product-record-card-background: var(--color-surface-base);
  --product-record-card-border: var(--color-border-muted);
  --product-record-card-padding: 14px 15px;
}

.plugin-manager__plugin-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 4px;
}

.plugin-manager__plugin-name {
  font-weight: 500;
}

.plugin-manager__plugin-version {
  font-size: 12px;
  color: var(--color-text-supporting);
}

.plugin-manager__plugin-description {
  font-size: 13px;
  color: var(--color-text-supporting);
  margin: 0;
}

.plugin-manager__plugin-meta {
  margin: 4px 0 0;
  font-size: 12px;
  color: var(--color-text-supporting);
}

.plugin-manager__plugin-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

.plugin-manager__default-state-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid var(--color-border-muted);
}

.plugin-manager__default-state-item:last-child {
  border-bottom: none;
}

.plugin-manager__config-body {
  background:
    linear-gradient(180deg, var(--plugin-manager-config-body-background-start) 0%, var(--plugin-manager-config-body-background-end) 100%),
    var(--color-surface-card);
  padding: 20px 24px 24px;
  overflow-y: auto;
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.plugin-manager__config-field-card {
  --product-record-card-background: var(--color-surface-input);
  --product-record-card-border: var(--color-border-muted);
  --product-record-card-radius: 14px;
  --product-record-card-padding: 16px 18px;
  --product-record-card-shadow: 0 8px 24px var(--shadow-soft);
  --product-record-card-gap: 10px;
}

.plugin-manager__config-field-key {
  flex-shrink: 0;
  padding: 4px 8px;
  border-radius: 999px;
  background: var(--plugin-manager-config-key-background);
  color: var(--plugin-manager-config-key-text);
  font-size: 12px;
  font-family: var(--font-mono, 'Consolas', monospace);
}

.plugin-manager__config-switch-row {
  display: inline-flex;
  align-items: center;
  gap: 12px;
}

.plugin-manager__config-switch-text {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-strong);
}

.plugin-manager__status {
  margin-top: 12px;
}

.plugin-manager__settings-hint {
  font-size: 13px;
  color: var(--color-text-supporting);
  margin-bottom: 10px;
}

.plugin-manager__import-input {
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
