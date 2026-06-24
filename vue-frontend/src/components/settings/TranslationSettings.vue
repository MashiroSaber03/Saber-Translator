<template>
  <div class="translation-settings">
    <UiPanel variant="settings">
      <template #title>翻译服务配置</template>
      <UiFormGrid>
        <UiField class="ui-settings-field">
          <label for="settingsModelProvider">翻译服务商:</label>
          <CustomSelect
            :model-value="localSettings.modelProvider"
            :options="providerOptions"
            @change="handleProviderSelect"
          />
        </UiField>
        <UiField v-show="!isLocalProvider" class="ui-settings-field">
          <label for="settingsApiKey">{{ apiKeyLabel }}:</label>
          <div class="password-input-wrapper">
            <UiInput
              :type="showApiKey ? 'text' : 'password'"
              id="settingsApiKey"
              v-model="localSettings.apiKey"
              class="secure-input"
              :placeholder="apiKeyPlaceholder"
              autocomplete="off"
            />
            <UiButton variant="toolbar" type="button" class="password-toggle-btn" tabindex="-1" @click="showApiKey = !showApiKey">
              <span class="eye-icon" v-if="!showApiKey">👁</span>
              <span class="eye-off-icon" v-else>👁‍🗨</span>
            </UiButton>
          </div>
        </UiField>
      </UiFormGrid>
      <UiField v-show="providerRequiresBaseUrl(localSettings.modelProvider)" class="ui-settings-field">
        <label for="settingsCustomBaseUrl">Base URL:</label>
        <UiInput
          type="text"
          id="settingsCustomBaseUrl"
          v-model="localSettings.customBaseUrl"
          placeholder="例如: https://api.example.com/v1"
        />
      </UiField>
      <UiField v-show="!isLocalProvider" class="ui-settings-field">
        <label for="settingsModelName">{{ modelNameLabel }}:</label>
        <div class="model-input-with-fetch">
          <UiInput
            type="text"
            id="settingsModelName"
            v-model="localSettings.modelName"
            class="translation-settings__model-input"
            :placeholder="modelNamePlaceholder"
          />
          <UiButton
            variant="toolbar"
            v-show="supportsFetchModels"
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
        <div v-if="modelList.length > 0" class="model-select-container">
          <CustomSelect
            :model-value="localSettings.modelName"
            :options="modelListOptions"
            @change="handleModelSelect"
          />
          <span class="model-count">共 {{ modelList.length }} 个模型</span>
        </div>
      </UiField>
      <UiField v-show="isLocalProvider" class="ui-settings-field">
        <label for="settingsLocalModelName">模型名称:</label>
        <div class="model-input-with-fetch">
          <UiInput
            type="text"
            id="settingsLocalModelName"
            v-model="localSettings.modelName"
            class="translation-settings__model-input"
            :placeholder="localSettings.modelProvider === 'ollama' ? '例如: qwen2.5:7b' : '例如: sakura-14b-qwen2.5-v1.0'"
          />
          <UiButton
            variant="toolbar"
            type="button"
            class="fetch-models-btn"
            title="获取本地可用模型列表"
            @click="fetchLocalModels"
            :disabled="isFetchingModels"
          >
            <span class="fetch-icon">🔍</span>
            <span class="fetch-text">{{ isFetchingModels ? '获取中...' : '获取模型' }}</span>
          </UiButton>
        </div>
        <div v-if="localModelList.length > 0" class="model-select-container">
          <CustomSelect
            :model-value="localSettings.modelName"
            :options="localModelListOptions"
            @change="handleModelSelect"
          />
          <span class="model-count">共 {{ localModelList.length }} 个模型</span>
        </div>
      </UiField>
      <UiFormGrid>
        <UiField v-show="showRpmLimit" class="ui-settings-field">
          <label for="settingsRpmTranslation">RPM限制:</label>
          <UiInput type="number" id="settingsRpmTranslation" v-model.number="localSettings.rpmTranslation" min="0" step="1" />
          <div class="ui-form-hint">每分钟请求数，0表示无限制</div>
        </UiField>
        <UiField class="ui-settings-field">
          <label for="settingsTranslationMaxRetries">重试次数:</label>
          <UiInput
            type="number"
            id="settingsTranslationMaxRetries"
            v-model.number="localSettings.translationBusinessRetries"
            min="0"
            max="10"
            step="1"
          />
          <div class="ui-form-hint">业务重试：空结果/结构解析失败</div>
        </UiField>
        <UiField class="ui-settings-field">
          <label for="settingsTranslationTransportRetries">传输重试:</label>
          <UiInput
            type="number"
            id="settingsTranslationTransportRetries"
            v-model.number="localSettings.translationTransportRetries"
            min="0"
            max="10"
            step="1"
          />
          <div class="ui-form-hint">网络超时/429/5xx</div>
        </UiField>
      </UiFormGrid>
      <UiField v-show="showRpmLimit" class="ui-settings-field">
        <label class="ui-checkbox-label">
          <UiInput type="checkbox" class="translation-settings__checkbox-input" v-model="localSettings.useStream" />
          流式调用
        </label>
        <div class="ui-form-hint">同时作用于整页批量和逐气泡翻译</div>
      </UiField>
      <UiField v-show="showRpmLimit" class="ui-settings-field">
        <OpenAIExtraBodyEditor v-model="localSettings.extraBody" />
      </UiField>
      <UiField class="ui-settings-field">
        <label for="settingsTranslationMode">翻译模式:</label>
        <CustomSelect
          :model-value="localSettings.translationMode"
          :options="translationModeOptions"
          @change="handleTranslationModeChange"
        />
        <div class="ui-form-hint translation-mode-hint">
          <span v-if="localSettings.translationMode === 'batch'">
            💡 整页批量翻译：一次发送全部气泡，效率高，需要模型支持复杂指令
          </span>
          <span v-else>
            💡 逐气泡翻译：每个气泡单独翻译，更稳定，适合小模型或格式敏感场景
          </span>
        </div>
        <div v-if="localSettings.modelProvider === 'sakura'" class="ui-form-hint sakura-suggestion">
          ⚠️ 建议 Sakura 服务使用"逐气泡翻译"模式，可获得更稳定的翻译效果
        </div>
      </UiField>
      <UiField v-show="isLocalProvider" class="ui-settings-field">
        <UiButton variant="toolbar" class="settings-test-btn" @click="testLocalConnection" :disabled="isTesting">
          {{ isTesting ? '测试中...' : '🔗 测试连接' }}
        </UiButton>
      </UiField>
      <UiField v-show="!isLocalProvider" class="ui-settings-field">
        <UiButton variant="toolbar" class="settings-test-btn" @click="testCloudConnection" :disabled="isTesting">
          {{ isTesting ? '测试中...' : '🔗 测试连接' }}
        </UiButton>
      </UiField>
    </UiPanel>
    <UiPanel variant="settings">
      <template #title>提示词设置</template>
      <UiField class="ui-settings-field">
        <label for="settingsPromptContent">翻译提示词:</label>
        <UiTextarea id="settingsPromptContent" v-model="localSettings.promptContent" rows="4" placeholder="翻译提示词" />
        <div class="prompt-format-selector">
          <CustomSelect
            :model-value="localSettings.translatePromptMode"
            :options="promptModeOptions"
            @change="handlePromptModeSelect"
          />
          <span class="ui-form-hint">JSON格式输出更结构化</span>
        </div>
        <SavedPromptsPicker
          prompt-type="translate"
          @select="handleTranslatePromptSelect"
        />
        <UiButton variant="toolbar" type="button" class="reset-btn" @click="resetTranslatePromptToDefault">
          重置为默认
        </UiButton>
      </UiField>
      <UiField class="ui-settings-field">
        <label class="ui-checkbox-label">
          <UiInput type="checkbox" class="translation-settings__checkbox-input" v-model="localSettings.enableTextboxPrompt" />
          启用文本框提示词
        </label>
      </UiField>
      <UiField v-show="localSettings.enableTextboxPrompt" class="ui-settings-field">
        <label for="settingsTextboxPromptContent">文本框提示词:</label>
        <UiTextarea
          id="settingsTextboxPromptContent"
          v-model="localSettings.textboxPromptContent"
          rows="3"
          placeholder="文本框提示词"
        />
        <SavedPromptsPicker
          prompt-type="textbox"
          @select="handleTextboxPromptSelect"
        />
      </UiField>
    </UiPanel>
  </div>
</template>
<script setup lang="ts">

import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiPanel from '@/components/ui/UiPanel.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { ref, computed, watch } from 'vue'
import {
  getProviderDisplayName as getProviderDisplayNameFromManifest,
  providerSupportsRpmLimit,
  getProviderOptionsForCapability,
  isLocalProviderId,
  normalizeProviderId,
  providerRequiresBaseUrl,
  providerSupportsCapability
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settings'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
import { DEFAULT_TRANSLATE_PROMPT, DEFAULT_TRANSLATE_JSON_PROMPT, DEFAULT_SINGLE_BUBBLE_PROMPT, DEFAULT_SINGLE_BUBBLE_JSON_PROMPT } from '@/constants'
import type { TranslationMode, TranslationProvider } from '@/types/settings'
import CustomSelect from '@/components/common/CustomSelect.vue'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'
import {
  getTranslationApiKeyLabel,
  getTranslationApiKeyPlaceholder,
  getTranslationModelNameLabel,
  getTranslationModelNamePlaceholder,
} from './translationSettingsLabels'
const providerOptions = getProviderOptionsForCapability('translation')
const promptModeOptions = [
  { label: '普通提示词', value: 'normal' },
  { label: 'JSON提示词', value: 'json' }
]
const translationModeOptions = [
  { label: '整页批量翻译 (推荐)', value: 'batch' },
  { label: '逐气泡翻译 (适合小模型)', value: 'single' }
]
type SelectValue = string | number
const settingsStore = useSettingsStore()
const toast = useToast()
const currentTranslationMode = settingsStore.settings.translation.translationMode || 'batch'
const currentForceJsonOutput = settingsStore.settings.translation.openaiOptions.request.forceJsonOutput || false
const getCurrentPrompt = (): string => {
  const t = settingsStore.settings.translation
  if (currentTranslationMode === 'single') {
    return currentForceJsonOutput ? t.singleJsonPrompt : t.singleNormalPrompt
  } else {
    return currentForceJsonOutput ? t.batchJsonPrompt : t.batchNormalPrompt
  }
}
const localSettings = ref({
  modelProvider: normalizeProviderId(settingsStore.settings.translation.provider),
  apiKey: settingsStore.settings.translation.apiKey,
  modelName: settingsStore.settings.translation.modelName,
  customBaseUrl: settingsStore.settings.translation.customBaseUrl,
  rpmTranslation: settingsStore.settings.translation.openaiOptions.execution.rpmLimit,
  translationTransportRetries: settingsStore.settings.translation.openaiOptions.execution.transportRetries,
  translationBusinessRetries: settingsStore.settings.translation.openaiOptions.execution.businessRetries,
  useStream: settingsStore.settings.translation.openaiOptions.execution.useStream,
  extraBody: settingsStore.settings.translation.openaiOptions.request.extraBody,
  translationMode: currentTranslationMode,
  promptContent: getCurrentPrompt(),
  translatePromptMode: currentForceJsonOutput ? 'json' : 'normal',
  enableTextboxPrompt: settingsStore.settings.useTextboxPrompt,
  textboxPromptContent: settingsStore.settings.textboxPrompt
})
const showApiKey = ref(false)
const isTesting = ref(false)
const isFetchingModels = ref(false)
const modelList = ref<string[]>([])
const localModelList = ref<string[]>([])
const modelListOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  modelList.value.forEach(model => options.push({ label: model, value: model }))
  return options
})
const localModelListOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  localModelList.value.forEach(model => options.push({ label: model, value: model }))
  return options
})
const isLocalProvider = computed(() => {
  return isLocalProviderId(localSettings.value.modelProvider)
})
const showRpmLimit = computed(() => {
  return providerSupportsRpmLimit(localSettings.value.modelProvider)
})
const supportsFetchModels = computed(() => {
  return providerSupportsCapability(localSettings.value.modelProvider, 'modelFetch') && !isLocalProviderId(localSettings.value.modelProvider)
})
const apiKeyLabel = computed(() => getTranslationApiKeyLabel(localSettings.value.modelProvider))
const apiKeyPlaceholder = computed(() => getTranslationApiKeyPlaceholder(localSettings.value.modelProvider))
const modelNameLabel = computed(() => getTranslationModelNameLabel(localSettings.value.modelProvider))
const modelNamePlaceholder = computed(() => getTranslationModelNamePlaceholder(localSettings.value.modelProvider))
function selectValueToString(value: SelectValue): string {
  return String(value)
}
function handleProviderSelect(value: SelectValue) {
  localSettings.value.modelProvider = selectValueToString(value)
  handleProviderChange()
}
function handleModelSelect(value: SelectValue) {
  localSettings.value.modelName = selectValueToString(value)
}
function handlePromptModeSelect(value: SelectValue) {
  localSettings.value.translatePromptMode = selectValueToString(value) === 'json' ? 'json' : 'normal'
  handlePromptModeChange()
}
function handleProviderChange() {
  const newProvider = localSettings.value.modelProvider as TranslationProvider
  localSettings.value.modelProvider = normalizeProviderId(newProvider)
  settingsStore.setTranslationProvider(localSettings.value.modelProvider as TranslationProvider)
  localSettings.value.apiKey = settingsStore.settings.translation.apiKey
  localSettings.value.modelName = settingsStore.settings.translation.modelName
  localSettings.value.customBaseUrl = settingsStore.settings.translation.customBaseUrl
  localSettings.value.rpmTranslation = settingsStore.settings.translation.openaiOptions.execution.rpmLimit
  localSettings.value.translationTransportRetries = settingsStore.settings.translation.openaiOptions.execution.transportRetries
  localSettings.value.translationBusinessRetries = settingsStore.settings.translation.openaiOptions.execution.businessRetries
  localSettings.value.useStream = settingsStore.settings.translation.openaiOptions.execution.useStream
  localSettings.value.extraBody = settingsStore.settings.translation.openaiOptions.request.extraBody
  localSettings.value.translationMode = settingsStore.settings.translation.translationMode || 'batch'
  modelList.value = []
  localModelList.value = []
}

function handlePromptModeChange() {
  const newForceJsonOutput = localSettings.value.translatePromptMode === 'json'
  const previousForceJsonOutput = !newForceJsonOutput
  const isSingleMode = localSettings.value.translationMode === 'single'
  if (isSingleMode) {
    if (previousForceJsonOutput) {
      settingsStore.updateTranslationService({ singleJsonPrompt: localSettings.value.promptContent })
    } else {
      settingsStore.updateTranslationService({ singleNormalPrompt: localSettings.value.promptContent })
    }
  } else {
    if (previousForceJsonOutput) {
      settingsStore.updateTranslationService({ batchJsonPrompt: localSettings.value.promptContent })
    } else {
      settingsStore.updateTranslationService({ batchNormalPrompt: localSettings.value.promptContent })
    }
  }
  const t = settingsStore.settings.translation
  let newPrompt: string
  if (isSingleMode) {
    newPrompt = newForceJsonOutput ? t.singleJsonPrompt : t.singleNormalPrompt
  } else {
    newPrompt = newForceJsonOutput ? t.batchJsonPrompt : t.batchNormalPrompt
  }
  localSettings.value.promptContent = newPrompt
  settingsStore.updateTranslationService({ forceJsonOutput: newForceJsonOutput })
  settingsStore.setTranslatePrompt(newPrompt)
}
function handleTranslationModeChange(value: SelectValue) {
  const newMode: TranslationMode = selectValueToString(value) === 'single' ? 'single' : 'batch'
  const previousMode = localSettings.value.translationMode
  const forceJsonOutput = localSettings.value.translatePromptMode === 'json'
  if (newMode === previousMode) return
  if (previousMode === 'batch') {
    if (forceJsonOutput) {
      settingsStore.updateTranslationService({ batchJsonPrompt: localSettings.value.promptContent })
    } else {
      settingsStore.updateTranslationService({ batchNormalPrompt: localSettings.value.promptContent })
    }
  } else {
    if (forceJsonOutput) {
      settingsStore.updateTranslationService({ singleJsonPrompt: localSettings.value.promptContent })
    } else {
      settingsStore.updateTranslationService({ singleNormalPrompt: localSettings.value.promptContent })
    }
  }
  localSettings.value.translationMode = newMode
  settingsStore.updateTranslationService({ translationMode: newMode })
  const t = settingsStore.settings.translation
  let savedPrompt: string
  if (newMode === 'single') {
    savedPrompt = forceJsonOutput ? t.singleJsonPrompt : t.singleNormalPrompt
  } else {
    savedPrompt = forceJsonOutput ? t.batchJsonPrompt : t.batchNormalPrompt
  }
  localSettings.value.promptContent = savedPrompt
  settingsStore.setTranslatePrompt(savedPrompt)
}
watch(() => localSettings.value.apiKey, (newVal) => {
  settingsStore.updateTranslationService({ apiKey: newVal })
})
watch(() => localSettings.value.modelName, (newVal) => {
  settingsStore.updateTranslationService({ modelName: newVal })
})
watch(() => localSettings.value.customBaseUrl, (newVal) => {
  settingsStore.updateTranslationService({ customBaseUrl: newVal })
})
watch(() => localSettings.value.rpmTranslation, (newVal) => {
  settingsStore.updateTranslationService({ rpmLimit: newVal })
})
watch(() => localSettings.value.translationTransportRetries, (newVal) => {
  settingsStore.updateTranslationService({ transportRetries: newVal })
})
watch(() => localSettings.value.translationBusinessRetries, (newVal) => {
  settingsStore.updateTranslationService({ businessRetries: newVal })
})
watch(() => localSettings.value.useStream, (newVal) => {
  settingsStore.updateTranslationService({ useStream: newVal })
})
watch(() => localSettings.value.extraBody, (newVal) => {
  settingsStore.updateTranslationService({ extraBody: newVal })
})
watch(() => localSettings.value.promptContent, (newVal) => {
  settingsStore.setTranslatePrompt(newVal)
  const isBatch = localSettings.value.translationMode === 'batch'
  const isJson = localSettings.value.translatePromptMode === 'json'
  if (isBatch) {
    if (isJson) {
      settingsStore.updateTranslationService({ batchJsonPrompt: newVal })
    } else {
      settingsStore.updateTranslationService({ batchNormalPrompt: newVal })
    }
  } else {
    if (isJson) {
      settingsStore.updateTranslationService({ singleJsonPrompt: newVal })
    } else {
      settingsStore.updateTranslationService({ singleNormalPrompt: newVal })
    }
  }
})
watch(() => localSettings.value.enableTextboxPrompt, (newVal) => {
  settingsStore.setUseTextboxPrompt(newVal)
})
watch(() => localSettings.value.textboxPromptContent, (newVal) => {
  settingsStore.setTextboxPrompt(newVal)
})
async function fetchModels() {
  const provider = localSettings.value.modelProvider
  const apiKey = localSettings.value.apiKey?.trim()
  const baseUrl = localSettings.value.customBaseUrl?.trim()
  if (!apiKey) {
    toast.warning('请先填写 API Key')
    return
  }
  if (!providerSupportsCapability(provider, 'modelFetch') || isLocalProviderId(provider)) {
    toast.warning(`${getProviderDisplayName(provider)} 不支持自动获取模型列表`)
    return
  }
  if (providerRequiresBaseUrl(provider) && !baseUrl) {
    toast.warning('自定义服务需要先填写 Base URL')
    return
  }
  isFetchingModels.value = true
  try {
    const result = await configApi.fetchModels(provider, apiKey, baseUrl)
    if (result.success && result.models && result.models.length > 0) {
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
function getProviderDisplayName(provider: string): string {
  return getProviderDisplayNameFromManifest(provider)
}

async function fetchLocalModels() {
  const provider = localSettings.value.modelProvider
  isFetchingModels.value = true
  try {
    if (provider === 'sakura') {
      const result = await configApi.testSakuraConnection()
      if (result.success && result.models) {
        localModelList.value = result.models
        toast.success(`获取到 ${result.models.length} 个Sakura模型`)
      } else {
        toast.error(result.error || 'Sakura连接失败')
      }
      return
    }

    if (provider === 'ollama') {
      const result = await configApi.fetchModels(provider, '', '')
      if (result.success && result.models) {
        localModelList.value = result.models.map(model => model.id)
        toast.success(`获取到 ${result.models.length} 个Ollama模型`)
      } else {
        toast.error(result.error || 'Ollama连接失败')
      }
    } else {
      toast.error('未选择本地服务商')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '获取本地模型失败'
    toast.error(errorMessage)
  } finally {
    isFetchingModels.value = false
  }
}
async function testLocalConnection() {
  const provider = localSettings.value.modelProvider
  const modelName = localSettings.value.modelName?.trim()
  if (provider === 'ollama' && !modelName) {
    toast.warning('请填写模型名称')
    return
  }
  isTesting.value = true
  try {
    let result
    if (provider === 'sakura') {
      result = await configApi.testSakuraConnection()
    } else if (provider === 'ollama') {
      result = await configApi.testAiTranslateConnection({
        provider,
        apiKey: '',
        modelName,
        baseUrl: ''
      })
    } else {
      toast.error('未选择本地服务商')
      return
    }
    if (result.success) {
      toast.success(`${provider === 'ollama' ? 'Ollama' : 'Sakura'} 连接成功`)
    } else {
      toast.error(result.error || '连接失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '连接测试失败'
    toast.error(errorMessage)
  } finally {
    isTesting.value = false
  }
}
async function testCloudConnection() {
  const provider = localSettings.value.modelProvider
  const apiKey = localSettings.value.apiKey?.trim()
  const modelName = localSettings.value.modelName?.trim()
  const baseUrl = localSettings.value.customBaseUrl?.trim()
  if (!apiKey) {
    toast.warning('请先填写 API Key')
    return
  }
  if (provider !== 'caiyun' && !modelName) {
    toast.warning('请填写模型名称')
    return
  }
  if (providerRequiresBaseUrl(provider) && !baseUrl) {
    toast.warning('自定义服务需要填写 Base URL')
    return
  }
  isTesting.value = true
  toast.info('正在测试连接...')
  try {
    let result
    switch (provider) {
      case 'baidu_translate':
        result = await configApi.testBaiduTranslateConnection(apiKey, modelName)
        break
      case 'youdao_translate':
        result = await configApi.testYoudaoTranslateConnection(apiKey, modelName)
        break
      default:
        result = await configApi.testAiTranslateConnection({
          provider,
          apiKey,
          modelName,
          baseUrl
        })
    }
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
function handleTranslatePromptSelect(content: string, name: string) {
  localSettings.value.promptContent = content
  toast.success(`已应用提示词: ${name}`)
}
function handleTextboxPromptSelect(content: string, name: string) {
  localSettings.value.textboxPromptContent = content
  toast.success(`已应用提示词: ${name}`)
}
function resetTranslatePromptToDefault() {
  const forceJsonOutput = localSettings.value.translatePromptMode === 'json'
  if (localSettings.value.translationMode === 'single') {
    localSettings.value.promptContent = forceJsonOutput ? DEFAULT_SINGLE_BUBBLE_JSON_PROMPT : DEFAULT_SINGLE_BUBBLE_PROMPT
  } else {
    localSettings.value.promptContent = forceJsonOutput ? DEFAULT_TRANSLATE_JSON_PROMPT : DEFAULT_TRANSLATE_PROMPT
  }
  toast.success('已重置为默认提示词')
}
</script>

<style scoped>.model-hint {
  color: var(--color-text-supporting);
  font-size: 12px;
  margin-top: 5px;
}

.ui-checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.translation-settings__checkbox-input {
  width: auto;
}

.translation-settings .model-input-with-fetch {
  display: flex;
  align-items: center;
  gap: 10px;
}

.translation-settings .model-input-with-fetch .translation-settings__model-input {
  flex: 1;
  min-width: 0;
}

.translation-settings .fetch-models-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  height: 38px;
  padding: 8px 12px;
  border: none;
  border-radius: 6px;
  background: var(--color-action-primary);
  color: var(--color-text-inverse);
  font-size: 0.9em;
  font-weight: 500;
  line-height: 1;
  white-space: nowrap;
  cursor: pointer;
  transition: background 0.2s ease, opacity 0.2s ease;
}

.translation-settings .fetch-models-btn:hover:not(:disabled) {
  background: var(--color-action-primary-hover);
}

.translation-settings .fetch-models-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.translation-settings .settings-test-btn {
  width: 100%;
  padding: 10px 16px;
  border: 1px solid var(--color-border-muted);
  border-radius: 6px;
  background-color: var(--color-surface-subtle);
  color: var(--color-text-default);
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.translation-settings .settings-test-btn:hover:not(:disabled) {
  border-color: var(--color-action-primary);
  background-color: var(--color-surface-hover);
  color: var(--color-action-primary);
}

.translation-settings .settings-test-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.password-input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.password-input-wrapper .secure-input {
  flex: 1;
  padding-right: 36px;
}

.password-toggle-btn {
  position: absolute;
  right: 8px;
  top: 50%;
  transform: translateY(-50%);
  background: none;
  border: none;
  cursor: pointer;
  padding: 4px;
  font-size: 16px;
  opacity: 0.6;
  transition: opacity 0.2s;
}

.password-toggle-btn:hover {
  opacity: 1;
}

.eye-icon,
.eye-off-icon {
  display: inline-block;
  line-height: 1;
}

.reset-btn {
  margin-top: 8px;
  padding: 6px 12px;
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  background: transparent;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.reset-btn:hover {
  color: var(--color-action-primary);
  border-color: var(--color-action-primary, var(--color-border-info));
  background: var(--color-surface-interactive-hover);
}

.translation-mode-hint {
  margin-top: 6px;
  padding: 8px 12px;
  background: var(--color-surface-subtle);
  border-radius: 6px;
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  border-left: 3px solid var(--color-action-primary, var(--color-border-info));
}

.sakura-suggestion {
  margin-top: 6px;
  padding: 8px 12px;
  background: var(--color-surface-warning-tint);
  border-radius: 6px;
  font-size: 12px;
  color: var(--color-status-warning);
  border-left: 3px solid var(--color-status-warning);
  font-weight: 500;
}
</style>
