<template>
  <div class="plugin-manager">
    <!-- 插件列表 -->
    <div class="settings-group">
      <div class="settings-group-title">已安装插件</div>
      <div v-if="isLoading" class="loading-hint">加载中...</div>
      <div v-else-if="plugins.length === 0" class="empty-hint">暂无已安装的插件</div>
      <div v-else class="plugin-list">
        <div v-for="plugin in plugins" :key="plugin.name" class="plugin-item">
          <div class="plugin-info">
            <div class="plugin-header">
              <span class="plugin-name">{{ plugin.display_name || plugin.name }}</span>
              <span class="plugin-version">v{{ plugin.version || '1.0.0' }}</span>
            </div>
            <p class="plugin-description">{{ plugin.description || '暂无描述' }}</p>
          </div>
          <div class="plugin-controls">
            <label class="switch">
              <input type="checkbox" :checked="plugin.enabled" @change="togglePlugin(plugin)" />
              <span class="slider"></span>
            </label>
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
      <div v-for="plugin in plugins" :key="'default-' + plugin.name" class="default-state-item">
        <span class="plugin-name">{{ plugin.display_name || plugin.name }}</span>
        <label class="switch">
          <input type="checkbox" :checked="defaultStates[plugin.name]" @change="setDefaultState(plugin.name, $event)" />
          <span class="slider"></span>
        </label>
      </div>
    </div>

    <!-- 插件配置模态框 -->
    <div v-if="showConfigModal" class="plugin-config-modal" @click.self="closeConfigModal">
      <div class="plugin-config-content">
        <div class="plugin-config-header">
          <h4>{{ configPlugin?.display_name || configPlugin?.name }} 配置</h4>
          <span class="close-btn" @click="closeConfigModal">&times;</span>
        </div>
        <div class="plugin-config-body">
          <div v-for="(field, key) in configSchema" :key="key" class="config-field">
            <label :for="'config-' + key">{{ field.label || key }}:</label>
            <template v-if="field.type === 'boolean'">
              <input type="checkbox" :id="'config-' + key" v-model="configValues[key]" />
            </template>
            <template v-else-if="field.type === 'select'">
              <CustomSelect
                :model-value="String(configValues[key] ?? '')"
                :options="field.options || []"
                @change="(v: string | number) => { configValues[key] = v }"
              />
            </template>
            <template v-else-if="field.type === 'number'">
              <input type="number" :id="'config-' + key" v-model.number="configValues[key]" :min="field.min" :max="field.max" />
            </template>
            <template v-else>
              <input type="text" :id="'config-' + key" v-model="configValues[key]" :placeholder="field.placeholder" />
            </template>
            <p v-if="field.description" class="field-description">{{ field.description }}</p>
          </div>
        </div>
        <div class="plugin-config-footer">
          <button class="btn btn-secondary" @click="closeConfigModal">取消</button>
          <button class="btn btn-primary" @click="savePluginConfig">保存</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 插件管理组件
 * 管理插件的启用/禁用、配置和删除
 */
import { ref, onMounted } from 'vue'
import * as pluginApi from '@/api/plugin'
import { useToast } from '@/utils/toast'
import CustomSelect from '@/components/common/CustomSelect.vue'

// 插件接口
interface Plugin {
  name: string
  display_name?: string
  version?: string
  description?: string
  enabled: boolean
  has_config?: boolean
}

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

// 配置模态框状态
const showConfigModal = ref(false)
const configPlugin = ref<Plugin | null>(null)
const configSchema = ref<Record<string, ConfigField>>({})
const configValues = ref<Record<string, unknown>>({})

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
    const result = await pluginApi.getDefaultStates()
    defaultStates.value = result.states || {}
  } catch (error: unknown) {
    console.error('加载默认状态失败:', error)
  }
}

// 切换插件启用状态
async function togglePlugin(plugin: Plugin) {
  try {
    if (plugin.enabled) {
      await pluginApi.disablePlugin(plugin.name)
      plugin.enabled = false
      toast.success(`已禁用 ${plugin.display_name || plugin.name}`)
    } else {
      await pluginApi.enablePlugin(plugin.name)
      plugin.enabled = true
      toast.success(`已启用 ${plugin.display_name || plugin.name}`)
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '操作失败'
    toast.error(errorMessage)
  }
}

// 设置默认启用状态
async function setDefaultState(pluginName: string, event: Event) {
  const target = event.target as HTMLInputElement
  const enabled = target.checked
  try {
    await pluginApi.setDefaultState(pluginName, enabled)
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
    const schemaResult = await pluginApi.getConfigSchema(plugin.name)
    configSchema.value = (schemaResult.schema || {}) as Record<string, ConfigField>

    // 获取当前配置
    const configResult = await pluginApi.getConfig(plugin.name)
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
    await pluginApi.saveConfig(configPlugin.value.name, configValues.value)
    toast.success('配置保存成功')
    closeConfigModal()
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '保存配置失败'
    toast.error(errorMessage)
  }
}

// 删除插件
async function deletePlugin(plugin: Plugin) {
  if (!confirm(`确定要删除插件 "${plugin.display_name || plugin.name}" 吗？`)) {
    return
  }
  try {
    await pluginApi.deletePlugin(plugin.name)
    toast.success('插件删除成功')
    await loadPlugins()
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '删除插件失败'
    toast.error(errorMessage)
  }
}

// 初始化
onMounted(() => {
  loadPlugins()
  loadDefaultStates()
})
</script>

<style scoped>
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
  z-index: var(--z-modal);
}

.plugin-config-content {
  background: var(--bg-primary);
  border-radius: 8px;
  width: 90%;
  max-width: 500px;
  max-height: 80vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.plugin-config-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  border-bottom: 1px solid var(--border-color);
}

.plugin-config-header h4 {
  margin: 0;
}

.close-btn {
  font-size: 24px;
  cursor: pointer;
  color: var(--text-secondary);
}

.plugin-config-body {
  padding: 20px;
  overflow-y: auto;
  flex: 1;
}

.config-field {
  margin-bottom: 15px;
}

.config-field label {
  display: block;
  margin-bottom: 5px;
  font-weight: 500;
}

.field-description {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 4px;
}

.plugin-config-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 15px 20px;
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
</style>
