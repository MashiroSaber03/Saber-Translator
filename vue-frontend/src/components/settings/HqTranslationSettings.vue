<template>
  <div class="hq-translation-settings">
    <!-- 高质量翻译服务配置 -->
    <UiPanel variant="settings">
      <template #title>高质量翻译服务配置</template>
      <div class="ui-settings-row">
        <UiField class="ui-settings-field">
          <label for="settingsHqTranslateProvider">服务商:</label>
          <CustomSelect
            :model-value="hqSettings.provider"
            :options="providerOptions"
            @change="(v: any) => handleProviderChange(v)"
          />
        </UiField>
        <UiField v-show="providerRequiresApiKey(hqSettings.provider)" class="ui-settings-field">
          <label for="settingsHqApiKey">API Key:</label>
          <div class="password-input-wrapper">
            <UiInput
              :type="showApiKey ? 'text' : 'password'"
              id="settingsHqApiKey"
              v-model="localHqSettings.apiKey"
              class="secure-input"
              placeholder="请输入API Key"
              autocomplete="off"
            />
            <UiButton variant="toolbar" type="button" class="password-toggle-btn" tabindex="-1" @click="showApiKey = !showApiKey">
              <span class="eye-icon" v-if="!showApiKey">👁</span>
              <span class="eye-off-icon" v-else>👁‍🗨</span>
            </UiButton>
          </div>
        </UiField>
      </div>

      <!-- 自定义Base URL -->
      <UiField v-show="providerRequiresBaseUrl(hqSettings.provider)" class="ui-settings-field">
        <label for="settingsHqCustomBaseUrl">Base URL:</label>
        <UiInput
          type="text"
          id="settingsHqCustomBaseUrl"
          v-model="localHqSettings.customBaseUrl"
          placeholder="例如: https://api.example.com/v1"
        />
      </UiField>

      <!-- 模型名称 -->
      <UiField class="ui-settings-field">
        <label for="settingsHqModelName">模型名称:</label>
        <div class="model-input-with-fetch">
          <UiInput
            type="text"
            id="settingsHqModelName"
            v-model="localHqSettings.modelName"
            class="hq-translation-settings__model-input"
            placeholder="请输入模型名称"
          />
          <UiButton
            variant="toolbar"
            type="button"
            class="fetch-models-btn"
            title="获取可用模型列表"
            @click="fetchModels"
            :disabled="isFetchingModels"
          >
            <span class="fetch-icon">🔍</span>
            <span class="fetch-text">{{ isFetchingModels ? '获取中...' : '获取模型' }}</span>
          </UiButton>
        </div>
        <!-- 模型选择下拉框 -->
        <div v-if="modelList.length > 0" class="model-select-container">
          <CustomSelect
            v-model="localHqSettings.modelName"
            :options="modelListOptions"
          />
          <span class="model-count">共 {{ modelList.length }} 个模型</span>
        </div>
      </UiField>

      <!-- 测试连接按钮 -->
      <UiField class="ui-settings-field">
        <UiButton variant="toolbar" class="settings-test-btn" @click="testConnection" :disabled="isTesting">
          {{ isTesting ? '测试中...' : '🔗 测试连接' }}
        </UiButton>
      </UiField>
    </UiPanel>

    <!-- 批处理设置 -->
    <UiPanel variant="settings">
      <template #title>批处理设置</template>
      <div class="ui-settings-row">
        <UiField class="ui-settings-field">
          <label for="settingsHqBatchSize">批次大小:</label>
          <UiInput type="number" id="settingsHqBatchSize" v-model.number="localHqSettings.batchSize" min="1" max="10" step="1" />
          <div class="ui-form-hint">每批处理的图片数量 (推荐3-5张)</div>
        </UiField>
      </div>
      <div class="ui-settings-row">
        <UiField class="ui-settings-field">
          <label for="settingsHqRpmLimit">RPM限制:</label>
          <UiInput type="number" id="settingsHqRpmLimit" v-model.number="localHqSettings.rpmLimit" min="0" step="1" />
          <div class="ui-form-hint">每分钟请求数，0表示无限制</div>
        </UiField>
        <UiField class="ui-settings-field">
          <label for="settingsHqMaxRetries">重试次数:</label>
          <UiInput type="number" id="settingsHqMaxRetries" v-model.number="localHqSettings.businessRetries" min="0" max="10" step="1" />
          <div class="ui-form-hint">业务重试：空结果/结构解析失败</div>
        </UiField>
        <UiField class="ui-settings-field">
          <label for="settingsHqTransportRetries">传输重试:</label>
          <UiInput type="number" id="settingsHqTransportRetries" v-model.number="localHqSettings.transportRetries" min="0" max="10" step="1" />
          <div class="ui-form-hint">网络超时/429/5xx</div>
        </UiField>
      </div>
    </UiPanel>

    <!-- 高级选项 -->
    <UiPanel variant="settings">
      <template #title>高级选项</template>
      <div class="ui-settings-row">
        <UiField class="ui-settings-field">
          <label class="ui-checkbox-label">
            <UiInput type="checkbox" v-model="localHqSettings.forceJsonOutput" />
            强制JSON输出
          </label>
          <div class="ui-form-hint">使用 response_format: json_object</div>
        </UiField>
        <UiField class="ui-settings-field">
          <label class="ui-checkbox-label">
            <UiInput type="checkbox" v-model="localHqSettings.useStream" />
            流式调用
          </label>
          <div class="ui-form-hint">使用流式API调用</div>
        </UiField>
      </div>
      <UiField class="ui-settings-field">
        <OpenAIExtraBodyEditor v-model="localHqSettings.extraBody" />
      </UiField>
    </UiPanel>

    <!-- 高质量翻译提示词 -->
    <UiPanel variant="settings">
      <template #title>高质量翻译提示词</template>
      <UiField class="ui-settings-field">
        <UiTextarea id="settingsHqPrompt" v-model="localHqSettings.prompt" rows="6" placeholder="高质量翻译提示词" />
        <!-- 快速选择提示词 -->
        <SavedPromptsPicker
          prompt-type="hq_translate"
          @select="handleHqPromptSelect"
        />
        <UiButton variant="secondary" class="hq-reset-prompt-btn" @click="resetHqPrompt" size="sm">重置为默认</UiButton>
      </UiField>
    </UiPanel>
  </div>
</template>

<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiPanel from '@/components/ui/UiPanel.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
/**
 * 高质量翻译设置组件
 * 管理高质量翻译服务配置
 */
import { ref, computed, watch } from 'vue'
import {
  getProviderDisplayName as getProviderDisplayNameFromManifest,
  getProviderOptionsForCapability,
  providerRequiresApiKey,
  providerRequiresBaseUrl,
  providerSupportsCapability
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settings'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
import { DEFAULT_HQ_TRANSLATE_PROMPT } from '@/constants'
import CustomSelect from '@/components/common/CustomSelect.vue'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'

/** 服务商选项 */
const providerOptions = getProviderOptionsForCapability('hqTranslation')

// Store
const settingsStore = useSettingsStore()
const toast = useToast()

// 获取高质量翻译设置的响应式引用（用于显示条件判断）
const hqSettings = computed(() => settingsStore.settings.hqTranslation)

// 本地设置状态（用于双向绑定，修改后自动同步到 store）
const localHqSettings = ref({
  apiKey: settingsStore.settings.hqTranslation.apiKey,
  modelName: settingsStore.settings.hqTranslation.modelName,
  customBaseUrl: settingsStore.settings.hqTranslation.customBaseUrl,
  batchSize: settingsStore.settings.hqTranslation.batchSize,
  rpmLimit: settingsStore.settings.hqTranslation.openaiOptions.execution.rpmLimit,
  transportRetries: settingsStore.settings.hqTranslation.openaiOptions.execution.transportRetries,
  businessRetries: settingsStore.settings.hqTranslation.openaiOptions.execution.businessRetries,
  forceJsonOutput: settingsStore.settings.hqTranslation.openaiOptions.request.forceJsonOutput,
  extraBody: settingsStore.settings.hqTranslation.openaiOptions.request.extraBody,
  useStream: settingsStore.settings.hqTranslation.openaiOptions.execution.useStream,
  prompt: settingsStore.settings.hqTranslation.prompt
})

// ============================================================
// Watch 同步：本地状态变化时自动保存到 store
// ============================================================
watch(() => localHqSettings.value.apiKey, (val) => {
  settingsStore.updateHqTranslation({ apiKey: val })
})
watch(() => localHqSettings.value.modelName, (val) => {
  settingsStore.updateHqTranslation({ modelName: val })
})
watch(() => localHqSettings.value.customBaseUrl, (val) => {
  settingsStore.updateHqTranslation({ customBaseUrl: val })
})
watch(() => localHqSettings.value.batchSize, (val) => {
  settingsStore.updateHqTranslation({ batchSize: val })
})
watch(() => localHqSettings.value.rpmLimit, (val) => {
  settingsStore.updateHqTranslation({ rpmLimit: val })
})
watch(() => localHqSettings.value.transportRetries, (val) => {
  settingsStore.updateHqTranslation({ transportRetries: val })
})
watch(() => localHqSettings.value.businessRetries, (val) => {
  settingsStore.updateHqTranslation({ businessRetries: val })
})
watch(() => localHqSettings.value.forceJsonOutput, (val) => {
  settingsStore.updateHqTranslation({ forceJsonOutput: val })
})
watch(() => localHqSettings.value.extraBody, (val) => {
  settingsStore.updateHqTranslation({ extraBody: val })
})
watch(() => localHqSettings.value.useStream, (val) => {
  settingsStore.updateHqTranslation({ useStream: val })
})
watch(() => localHqSettings.value.prompt, (val) => {
  settingsStore.updateHqTranslation({ prompt: val })
})

// 密码显示状态
const showApiKey = ref(false)

// 模型获取状态
const isFetchingModels = ref(false)
const modelList = ref<string[]>([])

// 测试状态
const isTesting = ref(false)

/** 模型列表选项（用于CustomSelect） */
const modelListOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  modelList.value.forEach(model => options.push({ label: model, value: model }))
  return options
})

// 处理服务商切换（业务逻辑：独立保存每个服务商的配置）
function handleProviderChange(newProvider: string) {
  // 切换服务商时保存当前配置并加载目标服务商配置
  settingsStore.setHqProvider(newProvider as import('@/types/settings').HqTranslationProvider)
  // 清空模型列表
  modelList.value = []
  // 同步目标服务商配置到本地表单
  syncLocalHqSettings()
}

// 同步本地高质量翻译状态
function syncLocalHqSettings() {
  const hq = settingsStore.settings.hqTranslation
  localHqSettings.value.apiKey = hq.apiKey
  localHqSettings.value.modelName = hq.modelName
  localHqSettings.value.customBaseUrl = hq.customBaseUrl
  localHqSettings.value.batchSize = hq.batchSize
  localHqSettings.value.rpmLimit = hq.openaiOptions.execution.rpmLimit
  localHqSettings.value.transportRetries = hq.openaiOptions.execution.transportRetries
  localHqSettings.value.businessRetries = hq.openaiOptions.execution.businessRetries
  localHqSettings.value.forceJsonOutput = hq.openaiOptions.request.forceJsonOutput
  localHqSettings.value.extraBody = hq.openaiOptions.request.extraBody
  localHqSettings.value.useStream = hq.openaiOptions.execution.useStream
  localHqSettings.value.prompt = hq.prompt
}

// 获取服务商显示名称（按业务契约）
function getProviderDisplayName(provider: string): string {
  return getProviderDisplayNameFromManifest(provider)
}

// 获取模型列表（模型列表获取流程）
async function fetchModels() {
  const provider = hqSettings.value.provider
  const apiKey = localHqSettings.value.apiKey?.trim()
  const baseUrl = localHqSettings.value.customBaseUrl?.trim()

  // 验证（按业务契约）
  if (providerRequiresApiKey(provider) && !apiKey) {
    toast.warning('请先填写 API Key')
    return
  }

  // 检查是否支持模型获取
  if (!providerSupportsCapability(provider, 'modelFetch')) {
    toast.warning(`${getProviderDisplayName(provider)} 不支持自动获取模型列表`)
    return
  }

  // 自定义服务需要 base_url
  if (providerRequiresBaseUrl(provider) && !baseUrl) {
    toast.warning('自定义服务需要先填写 Base URL')
    return
  }

  isFetchingModels.value = true
  try {
    const result = await configApi.fetchModels(provider, apiKey, baseUrl)
    if (result.success && result.models && result.models.length > 0) {
      // 后端返回的是 {id, name} 对象数组，提取 id 作为模型列表
      modelList.value = result.models.map(m => m.id)
      toast.success(`获取到 ${result.models.length} 个模型`)
    } else {
      toast.warning(result.message || '未获取到可用模型')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '获取模型列表失败'
    toast.error(errorMessage)
  } finally {
    isFetchingModels.value = false
  }
}

// 测试高质量翻译服务连接（业务逻辑）
async function testConnection() {
  const provider = hqSettings.value.provider
  const apiKey = localHqSettings.value.apiKey?.trim()
  const modelName = localHqSettings.value.modelName?.trim()
  const baseUrl = localHqSettings.value.customBaseUrl?.trim()

  // 验证必填字段
  if (providerRequiresApiKey(provider) && !apiKey) {
    toast.warning('请先填写 API Key')
    return
  }

  if (!modelName) {
    toast.warning('请填写模型名称')
    return
  }

  // 自定义服务需要 base_url
  if (providerRequiresBaseUrl(provider) && !baseUrl) {
    toast.warning('自定义服务需要填写 Base URL')
    return
  }

  isTesting.value = true
  toast.info('正在测试连接...')

  try {
    const result = await configApi.testAiTranslateConnection({
      provider,
      apiKey,
      modelName,
      baseUrl
    })

    if (result.success) {
      toast.success(result.message || `${getProviderDisplayName(provider)} 连接成功!`)
    } else {
      toast.error(result.message || result.error || '连接失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '连接测试失败'
    toast.error(errorMessage)
  } finally {
    isTesting.value = false
  }
}

// 重置高质量翻译提示词
function resetHqPrompt() {
  settingsStore.updateHqTranslation({ prompt: DEFAULT_HQ_TRANSLATE_PROMPT })
  // 同步本地状态
  localHqSettings.value.prompt = DEFAULT_HQ_TRANSLATE_PROMPT
  toast.success('已重置为默认提示词')
}

// 处理高质量翻译提示词选择
function handleHqPromptSelect(content: string, name: string) {
  settingsStore.updateHqTranslation({ prompt: content })
  // 同步本地状态
  localHqSettings.value.prompt = content
  toast.success(`已应用提示词: ${name}`)
}
</script>

<style scoped>
.hq-translation-settings {
  --ui-button-sm-padding: 4px 12px;
  --ui-button-sm-font-size: 12px;
}

.ui-checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.ui-checkbox-label input[type='checkbox'] {
  width: auto;
}

.hq-translation-settings .model-input-with-fetch {
  display: flex;
  gap: 8px;
  align-items: center;
}

.hq-translation-settings .model-input-with-fetch .hq-translation-settings__model-input {
  flex: 1;
  min-width: 0;
}

.hq-reset-prompt-btn {
  margin-top: 8px;
}

.hq-translation-settings .password-input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.hq-translation-settings .password-input-wrapper .secure-input {
  flex: 1;
  padding-right: 36px;
}

.hq-translation-settings .password-toggle-btn {
  position: absolute;
  right: 8px;
  top: 50%;
  padding: 4px;
  background: none;
  border: none;
  color: var(--color-text-supporting);
  font-size: 16px;
  line-height: 1;
  opacity: 0.6;
  transform: translateY(-50%);
  transition: opacity 0.2s ease;
}

.hq-translation-settings .password-toggle-btn:hover {
  opacity: 1;
}

.hq-translation-settings .fetch-models-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  height: 38px;
  padding: 8px 12px;
  border: none;
  border-radius: 6px;
  background: var(--color-action-primary, var(--translation-settings-surface-base));
  color: var(--color-text-inverse);
  font-size: 0.9em;
  font-weight: 500;
  line-height: 1;
  white-space: nowrap;
  cursor: pointer;
  transition: background 0.2s ease, opacity 0.2s ease;
}

.hq-translation-settings .fetch-models-btn:hover:not(:disabled) {
  background: var(--translation-settings-surface-raised);
}

.hq-translation-settings .fetch-models-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.hq-translation-settings .settings-test-btn {
  width: auto;
  padding: 10px 16px;
  border: none;
  border-radius: 6px;
  background-color: var(--color-status-info, var(--color-action-primary));
  color: var(--color-text-inverse);
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s ease, opacity 0.2s ease;
}

.hq-translation-settings .settings-test-btn:hover:not(:disabled) {
  background-color: var(--color-status-info-hover);
}

.hq-translation-settings .settings-test-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
