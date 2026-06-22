<template>
  <div class="translation-settings">
    <UiPanel variant="settings">
      <template #title>翻译服务配置</template>
      <div class="ui-settings-row">
        <UiField class="ui-settings-field">
          <label for="settingsModelProvider">翻译服务商:</label>
          <CustomSelect
            :model-value="localSettings.modelProvider"
            :options="providerOptions"
            @change="(v: any) => { localSettings.modelProvider = v; handleProviderChange() }"
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
      </div>
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
            @change="(v: any) => { localSettings.modelName = v }"
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
            @change="(v: any) => localSettings.modelName = v"
          />
          <span class="model-count">共 {{ localModelList.length }} 个模型</span>
        </div>
      </UiField>
      <div class="ui-settings-row">
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
      </div>
      <UiField v-show="showRpmLimit" class="ui-settings-field">
        <label class="ui-checkbox-label">
          <UiInput type="checkbox" v-model="localSettings.useStream" />
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
            @change="(v: any) => { localSettings.translatePromptMode = v; handlePromptModeChange() }"
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
          <UiInput type="checkbox" v-model="localSettings.enableTextboxPrompt" />
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
import type { TranslationProvider } from '@/types/settings'
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
const settingsStore = useSettingsStore()
const toast = useToast()
// 本地状态（双向绑定用）
// 根据翻译模式和JSON模式选择对应的提示词（4个独立存储字段之一）
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
// 计算属性：是否为本地服务商
const isLocalProvider = computed(() => {
  return isLocalProviderId(localSettings.value.modelProvider)
})
// 计算属性：是否显示RPM限制
const showRpmLimit = computed(() => {
  return providerSupportsRpmLimit(localSettings.value.modelProvider)
})
// 计算属性：是否支持获取模型列表
const supportsFetchModels = computed(() => {
  return providerSupportsCapability(localSettings.value.modelProvider, 'modelFetch') && !isLocalProviderId(localSettings.value.modelProvider)
})
const apiKeyLabel = computed(() => getTranslationApiKeyLabel(localSettings.value.modelProvider))
const apiKeyPlaceholder = computed(() => getTranslationApiKeyPlaceholder(localSettings.value.modelProvider))
const modelNameLabel = computed(() => getTranslationModelNameLabel(localSettings.value.modelProvider))
const modelNamePlaceholder = computed(() => getTranslationModelNamePlaceholder(localSettings.value.modelProvider))
function handleProviderChange() {
  const newProvider = localSettings.value.modelProvider as TranslationProvider
  localSettings.value.modelProvider = normalizeProviderId(newProvider)
  // 切换服务商时保存当前配置并加载目标服务商配置
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
  // 清空所有模型列表（无论是云服务商还是本地服务商）
  modelList.value = []
  localModelList.value = []
}
// 处理提示词模式切换（普通 ↔ JSON）
function handlePromptModeChange() {
  const newForceJsonOutput = localSettings.value.translatePromptMode === 'json'
  const oldForceJsonOutput = !newForceJsonOutput  // 切换前的状态
  const isSingleMode = localSettings.value.translationMode === 'single'
  // 先保存当前提示词到对应的字段（切换前的字段）
  if (isSingleMode) {
    if (oldForceJsonOutput) {
      settingsStore.updateTranslationService({ singleJsonPrompt: localSettings.value.promptContent })
    } else {
      settingsStore.updateTranslationService({ singleNormalPrompt: localSettings.value.promptContent })
    }
  } else {
    if (oldForceJsonOutput) {
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
function handleTranslationModeChange(value: any) {
  const newMode = String(value) as 'batch' | 'single'
  const oldMode = localSettings.value.translationMode
  const forceJsonOutput = localSettings.value.translatePromptMode === 'json'
  // 如果模式没变，不做任何操作
  if (newMode === oldMode) return
  // 先保存当前模式的提示词到对应字段（4个字段之一）
  if (oldMode === 'batch') {
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
  // 加载新模式的已保存提示词（根据当前 JSON 模式选择对应字段）
  const t = settingsStore.settings.translation
  let savedPrompt: string
  if (newMode === 'single') {
    savedPrompt = forceJsonOutput ? t.singleJsonPrompt : t.singleNormalPrompt
  } else {
    savedPrompt = forceJsonOutput ? t.batchJsonPrompt : t.batchNormalPrompt
  }
  localSettings.value.promptContent = savedPrompt
  settingsStore.setTranslatePrompt(savedPrompt)
  console.log(`翻译模式已切换为: ${newMode === 'batch' ? '整页批量翻译' : '逐气泡翻译'}`)
}
// 监听本地设置变化，同步到 store
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
  // 同时保存到当前模式和 JSON 模式对应的字段（4个字段之一）
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
// 注意：translationMode 不需要 watch，因为 handleTranslationModeChange 已经处理了 store 同步
// 获取模型列表（模型列表获取流程）
async function fetchModels() {
  const provider = localSettings.value.modelProvider
  const apiKey = localSettings.value.apiKey?.trim()
  const baseUrl = localSettings.value.customBaseUrl?.trim()
  // 验证（按业务契约）
  if (!apiKey) {
    toast.warning('请先填写 API Key')
    return
  }
  // 检查是否支持模型获取（按业务契约）
  if (!providerSupportsCapability(provider, 'modelFetch') || isLocalProviderId(provider)) {
    toast.warning(`${getProviderDisplayName(provider)} 不支持自动获取模型列表`)
    return
  }
  // 自定义服务需要 base_url（按业务契约）
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
// 获取服务商显示名称（按业务契约）
function getProviderDisplayName(provider: string): string {
  return getProviderDisplayNameFromManifest(provider)
}
// 获取本地模型列表（Ollama 或 Sakura）
async function fetchLocalModels() {
  const provider = localSettings.value.modelProvider
  isFetchingModels.value = true
  try {
    let result
    if (provider === 'sakura') {
      result = await configApi.testSakuraConnection()
    } else if (provider === 'ollama') {
      result = await configApi.fetchModels(provider, '', '')
    } else {
      toast.error('未选择本地服务商')
      return
    }
    if (result.success && result.models) {
      localModelList.value = result.models.map((model: string | { id: string; name?: string }) =>
        typeof model === 'string' ? model : model.id
      )
      toast.success(`获取到 ${result.models.length} 个${provider === 'ollama' ? 'Ollama' : 'Sakura'}模型`)
    } else {
      toast.error(result.error || `${provider === 'ollama' ? 'Ollama' : 'Sakura'}连接失败`)
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
// 测试云服务商连接（连接测试流程）
async function testCloudConnection() {
  const provider = localSettings.value.modelProvider
  const apiKey = localSettings.value.apiKey?.trim()
  const modelName = localSettings.value.modelName?.trim()
  const baseUrl = localSettings.value.customBaseUrl?.trim()
  // 验证必填字段（按业务契约）
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
    // 根据服务商类型分发到不同的测试函数（按业务契约）
    switch (provider) {
      case 'baidu_translate':
        // 百度翻译使用 apiKey 作为 App ID，modelName 作为 App Key
        result = await configApi.testBaiduTranslateConnection(apiKey, modelName)
        break
      case 'youdao_translate':
        // 有道翻译使用 apiKey 作为 App Key，modelName 作为 App Secret
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

.ui-checkbox-label input[type='checkbox'] {
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
  background: var(--color-action-primary, var(--translation-settings-surface-base));
  color: var(--color-text-inverse);
  font-size: 0.9em;
  font-weight: 500;
  line-height: 1;
  white-space: nowrap;
  cursor: pointer;
  transition: background 0.2s ease, opacity 0.2s ease;
}

.translation-settings .fetch-models-btn:hover:not(:disabled) {
  background: var(--translation-settings-surface-raised);
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

/* 密码输入框包装器 */
.password-input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.password-input-wrapper .secure-input {
  flex: 1;
  padding-right: 36px;
}
/* 密码显示/隐藏切换按钮 */
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
/* 重置为默认按钮 */
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
  color: var(--color-action-primary, var(--translation-settings-text-primary));
  border-color: var(--color-action-primary, var(--color-border-info));
  background: var(--translation-settings-surface-muted);
}
/* 翻译模式提示样式 */
.translation-mode-hint {
  margin-top: 6px;
  padding: 8px 12px;
  background: var(--translation-settings-surface-subtle);
  border-radius: 6px;
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  border-left: 3px solid var(--color-action-primary, var(--color-border-info));
}
/* Sakura 服务商专属建议样式 */
.sakura-suggestion {
  margin-top: 6px;
  padding: 8px 12px;
  background: var(--color-surface-warning-tint);
  border-radius: 6px;
  font-size: 12px;
  color: var(--translation-settings-text-secondary);
  border-left: 3px solid var(--translation-settings-border-default);
  font-weight: 500;
}
</style>
