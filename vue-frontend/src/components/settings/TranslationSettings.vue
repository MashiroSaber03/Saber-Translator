<template>
  <div class="translation-settings">
    <ProductFormSection>
      <template #title>翻译服务配置</template>
      <UiFormGrid>
        <AiProviderSelectField
          :model-value="localSettings.modelProvider"
          input-id="settingsModelProvider"
          :options="providerOptions"
          label="翻译服务商"
          @change="handleProviderSelect"
        />
        <AiProviderCredentialFields
          :api-key="localSettings.apiKey"
          api-key-input-id="settingsApiKey"
          :base-url="localSettings.customBaseUrl"
          base-url-input-id="settingsCustomBaseUrl"
          :show-api-key="!isLocalProvider"
          :show-base-url="false"
          :include-base-url="false"
          :api-key-label="apiKeyLabel"
          :api-key-placeholder="apiKeyPlaceholder"
          api-key-show-label="显示翻译 API Key"
          api-key-hide-label="隐藏翻译 API Key"
          @update:api-key="localSettings.apiKey = $event"
        />
      </UiFormGrid>
      <AiProviderCredentialFields
        :api-key="localSettings.apiKey"
        api-key-input-id="settingsApiKey"
        :base-url="localSettings.customBaseUrl"
        base-url-input-id="settingsCustomBaseUrl"
        :show-api-key="false"
        :show-base-url="providerRequiresBaseUrl(localSettings.modelProvider)"
        :include-api-key="false"
        base-url-placeholder="例如: https://api.example.com/v1"
        @update:base-url="localSettings.customBaseUrl = $event"
      />
      <UiField
        v-show="!isLocalProvider"
        variant="settings"
        :label="modelNameLabel"
        control-id="settingsModelName"
      >
        <UiModelPicker
          input-id="settingsModelName"
          v-model="localSettings.modelName"
          :placeholder="modelNamePlaceholder"
          :show-fetch="supportsFetchModels"
          fetch-variant="primary"
          :fetching="isFetchingModels"
          :fetch-disabled="isFetchingModels"
          :options="modelListOptions"
          :model-count="modelList.length"
          @change="handleModelSelect"
          @fetch="fetchModels"
        />
      </UiField>
      <UiField v-show="isLocalProvider" variant="settings" label="模型名称" control-id="settingsLocalModelName">
        <UiModelPicker
          input-id="settingsLocalModelName"
          v-model="localSettings.modelName"
          :placeholder="
            localSettings.modelProvider === 'ollama'
              ? '例如: qwen2.5:7b'
              : '例如: sakura-14b-qwen2.5-v1.0'
          "
          fetch-title="获取本地可用模型列表"
          fetch-variant="primary"
          :fetching="isFetchingModels"
          :fetch-disabled="isFetchingModels"
          :options="localModelListOptions"
          :model-count="localModelList.length"
          @change="handleModelSelect"
          @fetch="fetchLocalModels"
        />
      </UiField>
      <UiFormGrid>
        <UiField
          v-show="showRpmLimit"
          variant="settings"
          label="RPM限制"
          control-id="settingsRpmTranslation"
          hint="每分钟请求数，0表示无限制"
        >
          <UiNumberField
            input-id="settingsRpmTranslation"
            v-model="localSettings.rpmTranslation"
            :min="0"
            :step="1"
          />
        </UiField>
        <UiField
          variant="settings"
          label="重试次数"
          control-id="settingsTranslationMaxRetries"
          hint="业务重试：空结果/结构解析失败"
        >
          <UiNumberField
            input-id="settingsTranslationMaxRetries"
            v-model="localSettings.translationBusinessRetries"
            :min="0"
            :max="10"
            :step="1"
          />
        </UiField>
        <UiField
          variant="settings"
          label="传输重试"
          control-id="settingsTranslationTransportRetries"
          hint="网络超时/429/5xx"
        >
          <UiNumberField
            input-id="settingsTranslationTransportRetries"
            v-model="localSettings.translationTransportRetries"
            :min="0"
            :max="10"
            :step="1"
          />
        </UiField>
      </UiFormGrid>
      <UiField
        v-show="showRpmLimit"
        variant="settings"
        control="checkbox"
        hint="同时作用于整页批量和逐气泡翻译"
      >
        <UiCheckbox v-model="localSettings.useStream" label="流式调用" />
      </UiField>
      <UiField v-show="showRpmLimit" variant="settings">
        <OpenAIExtraBodyEditor v-model="localSettings.extraBody" />
      </UiField>
      <UiField
        variant="settings"
        label="翻译模式"
        control-id="settingsTranslationMode"
        :hint="translationModeHint"
      >
        <UiSelect
          id="settingsTranslationMode"
          :model-value="localSettings.translationMode"
          :options="translationModeOptions"
          @change="handleTranslationModeChange"
        />
      </UiField>
      <ProductStatusBanner
        v-if="localSettings.modelProvider === 'sakura'"
        tone="warning"
        role="note"
      >
        建议 Sakura 服务使用"逐气泡翻译"模式，可获得更稳定的翻译效果
      </ProductStatusBanner>
      <ProductActionRow v-show="isLocalProvider" aria-label="本地翻译连接测试" justify="start">
        <UiButton
          variant="secondary"
          @click="testLocalConnection"
          :disabled="isTesting"
        >
          <span v-if="isTesting">测试中...</span>
          <template v-else>
            <UiIcon name="link" />
            <span>测试连接</span>
          </template>
        </UiButton>
      </ProductActionRow>
      <ProductActionRow v-show="!isLocalProvider" aria-label="云端翻译连接测试" justify="start">
        <UiButton
          variant="secondary"
          @click="testCloudConnection"
          :disabled="isTesting"
        >
          <span v-if="isTesting">测试中...</span>
          <template v-else>
            <UiIcon name="link" />
            <span>测试连接</span>
          </template>
        </UiButton>
      </ProductActionRow>
    </ProductFormSection>
    <ProductFormSection>
      <template #title>提示词设置</template>
      <UiField variant="settings" label="翻译提示词" control-id="settingsPromptContent">
        <UiTextarea
          id="settingsPromptContent"
          v-model="localSettings.promptContent"
          variant="panel"
          rows="4"
          placeholder="翻译提示词"
        />
        <ProductActionRow aria-label="翻译提示词格式" justify="start">
          <UiSelect
            :model-value="localSettings.translatePromptMode"
            :options="promptModeOptions"
            @change="handlePromptModeSelect"
          />
          <span class="translation-settings__prompt-mode-hint">JSON格式输出更结构化</span>
        </ProductActionRow>
        <SavedPromptsPicker prompt-type="translate" @select="handleTranslatePromptSelect" />
        <ProductActionRow aria-label="翻译提示词操作" justify="start">
          <UiButton variant="secondary" type="button" size="sm" @click="resetTranslatePromptToDefault">
            重置为默认
          </UiButton>
        </ProductActionRow>
      </UiField>
      <UiField variant="settings" control="checkbox">
        <UiCheckbox v-model="localSettings.enableTextboxPrompt" label="启用文本框提示词" />
      </UiField>
      <UiField
        v-show="localSettings.enableTextboxPrompt"
        variant="settings"
        label="文本框提示词"
        control-id="settingsTextboxPromptContent"
      >
        <UiTextarea
          id="settingsTextboxPromptContent"
          v-model="localSettings.textboxPromptContent"
          variant="panel"
          rows="3"
          placeholder="文本框提示词"
        />
        <SavedPromptsPicker prompt-type="textbox" @select="handleTextboxPromptSelect" />
      </UiField>
    </ProductFormSection>
  </div>
</template>
<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import AiProviderCredentialFields from '@/components/settings/AiProviderCredentialFields.vue'
import AiProviderSelectField from '@/components/settings/AiProviderSelectField.vue'
import type { UiSelectValue } from '@/components/ui/selectTypes'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import { ref, computed, watch } from 'vue'
import {
  getProviderDisplayName as getProviderDisplayNameFromManifest,
  providerSupportsRpmLimit,
  getProviderOptionsForCapability,
  isLocalProviderId,
  normalizeProviderId,
  providerRequiresBaseUrl,
  providerSupportsCapability,
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settings'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
import {
  DEFAULT_TRANSLATE_PROMPT,
  DEFAULT_TRANSLATE_JSON_PROMPT,
  DEFAULT_SINGLE_BUBBLE_PROMPT,
  DEFAULT_SINGLE_BUBBLE_JSON_PROMPT,
} from '@/constants'
import type { TranslationMode, TranslationProvider } from '@/types/settings'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'
import { useLatestRequestGuard } from '@/composables/useLatestRequestGuard'
import { useAiModelDiscovery, type AiModelDiscoveryMessageTone } from '@/composables/useAiModelDiscovery'
import {
  getTranslationApiKeyLabel,
  getTranslationApiKeyPlaceholder,
  getTranslationModelNameLabel,
  getTranslationModelNamePlaceholder,
} from './translationSettingsLabels'
const providerOptions = getProviderOptionsForCapability('translation')
const promptModeOptions = [
  { label: '普通提示词', value: 'normal' },
  { label: 'JSON提示词', value: 'json' },
]
const translationModeOptions = [
  { label: '整页批量翻译 (推荐)', value: 'batch' },
  { label: '逐气泡翻译 (适合小模型)', value: 'single' },
]
const settingsStore = useSettingsStore()
const toast = useToast()
const currentTranslationMode = settingsStore.settings.translation.translationMode || 'batch'
const currentForceJsonOutput =
  settingsStore.settings.translation.openaiOptions.request.forceJsonOutput || false
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
  translationTransportRetries:
    settingsStore.settings.translation.openaiOptions.execution.transportRetries,
  translationBusinessRetries:
    settingsStore.settings.translation.openaiOptions.execution.businessRetries,
  useStream: settingsStore.settings.translation.openaiOptions.execution.useStream,
  extraBody: settingsStore.settings.translation.openaiOptions.request.extraBody,
  translationMode: currentTranslationMode,
  promptContent: getCurrentPrompt(),
  translatePromptMode: currentForceJsonOutput ? 'json' : 'normal',
  enableTextboxPrompt: settingsStore.settings.useTextboxPrompt,
  textboxPromptContent: settingsStore.settings.textboxPrompt,
})
const isTesting = ref(false)
const isFetchingLocalModels = ref(false)
const localModelList = ref<string[]>([])
const localModelFetchGuard = useLatestRequestGuard()
function notifyModelDiscovery(message: string, tone: AiModelDiscoveryMessageTone): void {
  toast[tone](message)
}
const remoteModelDiscovery = useAiModelDiscovery({
  source: () => ({
    provider: localSettings.value.modelProvider,
    apiKey: localSettings.value.apiKey,
    baseUrl: localSettings.value.customBaseUrl,
  }),
  notify: notifyModelDiscovery,
  supportsProvider: provider => (
    providerSupportsCapability(provider, 'modelFetch') && !isLocalProviderId(provider)
  ),
  requiresApiKey: provider => !isLocalProviderId(provider),
  emptyBaseUrl: '',
})
const isFetchingModels = computed(() => (
  remoteModelDiscovery.isFetchingModels.value || isFetchingLocalModels.value
))
const modelList = computed(() => remoteModelDiscovery.models.value.map(model => model.id))
const modelListOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  remoteModelDiscovery.models.value.forEach(model => options.push({ label: model.id, value: model.id }))
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
  return (
    providerSupportsCapability(localSettings.value.modelProvider, 'modelFetch') &&
    !isLocalProviderId(localSettings.value.modelProvider)
  )
})
const apiKeyLabel = computed(() => getTranslationApiKeyLabel(localSettings.value.modelProvider))
const apiKeyPlaceholder = computed(() =>
  getTranslationApiKeyPlaceholder(localSettings.value.modelProvider)
)
const modelNameLabel = computed(() =>
  getTranslationModelNameLabel(localSettings.value.modelProvider)
)
const modelNamePlaceholder = computed(() =>
  getTranslationModelNamePlaceholder(localSettings.value.modelProvider)
)
const translationModeHint = computed(() =>
  localSettings.value.translationMode === 'batch'
    ? '整页批量翻译：一次发送全部气泡，效率高，需要模型支持复杂指令'
    : '逐气泡翻译：每个气泡单独翻译，更稳定，适合小模型或格式敏感场景'
)
function selectValueToString(value: UiSelectValue): string {
  return String(value)
}
function handleProviderSelect(value: UiSelectValue) {
  localSettings.value.modelProvider = selectValueToString(value)
  handleProviderChange()
}
function handleModelSelect(value: UiSelectValue) {
  localSettings.value.modelName = selectValueToString(value)
}
function handlePromptModeSelect(value: UiSelectValue) {
  localSettings.value.translatePromptMode =
    selectValueToString(value) === 'json' ? 'json' : 'normal'
  handlePromptModeChange()
}
function handleProviderChange() {
  const newProvider = localSettings.value.modelProvider as TranslationProvider
  localSettings.value.modelProvider = normalizeProviderId(newProvider)
  settingsStore.setTranslationProvider(localSettings.value.modelProvider as TranslationProvider)
  localSettings.value.apiKey = settingsStore.settings.translation.apiKey
  localSettings.value.modelName = settingsStore.settings.translation.modelName
  localSettings.value.customBaseUrl = settingsStore.settings.translation.customBaseUrl
  localSettings.value.rpmTranslation =
    settingsStore.settings.translation.openaiOptions.execution.rpmLimit
  localSettings.value.translationTransportRetries =
    settingsStore.settings.translation.openaiOptions.execution.transportRetries
  localSettings.value.translationBusinessRetries =
    settingsStore.settings.translation.openaiOptions.execution.businessRetries
  localSettings.value.useStream =
    settingsStore.settings.translation.openaiOptions.execution.useStream
  localSettings.value.extraBody = settingsStore.settings.translation.openaiOptions.request.extraBody
  localSettings.value.translationMode =
    settingsStore.settings.translation.translationMode || 'batch'
  invalidateModelFetchRequests()
}

function invalidateModelFetchRequests() {
  remoteModelDiscovery.invalidate()
  localModelFetchGuard.invalidate()
  isFetchingLocalModels.value = false
  localModelList.value = []
}

function handlePromptModeChange() {
  const newForceJsonOutput = localSettings.value.translatePromptMode === 'json'
  const previousForceJsonOutput = !newForceJsonOutput
  const isSingleMode = localSettings.value.translationMode === 'single'
  if (isSingleMode) {
    if (previousForceJsonOutput) {
      settingsStore.updateTranslationService({
        singleJsonPrompt: localSettings.value.promptContent,
      })
    } else {
      settingsStore.updateTranslationService({
        singleNormalPrompt: localSettings.value.promptContent,
      })
    }
  } else {
    if (previousForceJsonOutput) {
      settingsStore.updateTranslationService({ batchJsonPrompt: localSettings.value.promptContent })
    } else {
      settingsStore.updateTranslationService({
        batchNormalPrompt: localSettings.value.promptContent,
      })
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
function handleTranslationModeChange(value: UiSelectValue) {
  const newMode: TranslationMode = selectValueToString(value) === 'single' ? 'single' : 'batch'
  const previousMode = localSettings.value.translationMode
  const forceJsonOutput = localSettings.value.translatePromptMode === 'json'
  if (newMode === previousMode) return
  if (previousMode === 'batch') {
    if (forceJsonOutput) {
      settingsStore.updateTranslationService({ batchJsonPrompt: localSettings.value.promptContent })
    } else {
      settingsStore.updateTranslationService({
        batchNormalPrompt: localSettings.value.promptContent,
      })
    }
  } else {
    if (forceJsonOutput) {
      settingsStore.updateTranslationService({
        singleJsonPrompt: localSettings.value.promptContent,
      })
    } else {
      settingsStore.updateTranslationService({
        singleNormalPrompt: localSettings.value.promptContent,
      })
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
watch(
  () => localSettings.value.apiKey,
  newVal => {
    settingsStore.updateTranslationService({ apiKey: newVal })
  }
)
watch(
  () => localSettings.value.modelName,
  newVal => {
    settingsStore.updateTranslationService({ modelName: newVal })
  }
)
watch(
  () => localSettings.value.customBaseUrl,
  newVal => {
    settingsStore.updateTranslationService({ customBaseUrl: newVal })
  }
)
watch(
  () => localSettings.value.rpmTranslation,
  newVal => {
    settingsStore.updateTranslationService({ rpmLimit: newVal })
  }
)
watch(
  () => localSettings.value.translationTransportRetries,
  newVal => {
    settingsStore.updateTranslationService({ transportRetries: newVal })
  }
)
watch(
  () => localSettings.value.translationBusinessRetries,
  newVal => {
    settingsStore.updateTranslationService({ businessRetries: newVal })
  }
)
watch(
  () => localSettings.value.useStream,
  newVal => {
    settingsStore.updateTranslationService({ useStream: newVal })
  }
)
watch(
  () => localSettings.value.extraBody,
  newVal => {
    settingsStore.updateTranslationService({ extraBody: newVal })
  }
)
watch(
  () => localSettings.value.promptContent,
  newVal => {
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
  }
)
watch(
  () => localSettings.value.enableTextboxPrompt,
  newVal => {
    settingsStore.setUseTextboxPrompt(newVal)
  }
)
watch(
  () => localSettings.value.textboxPromptContent,
  newVal => {
    settingsStore.setTextboxPrompt(newVal)
  }
)
const fetchModels = remoteModelDiscovery.fetchModels
function getProviderDisplayName(provider: string): string {
  return getProviderDisplayNameFromManifest(provider)
}

async function fetchLocalModels() {
  const provider = localSettings.value.modelProvider
  const requestId = localModelFetchGuard.next()
  isFetchingLocalModels.value = true
  try {
    if (provider === 'sakura') {
      const result = await configApi.testSakuraConnection()
      if (!localModelFetchGuard.isCurrent(requestId)) return
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
      if (!localModelFetchGuard.isCurrent(requestId)) return
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
    if (!localModelFetchGuard.isCurrent(requestId)) return
    const errorMessage = error instanceof Error ? error.message : '获取本地模型失败'
    toast.error(errorMessage)
  } finally {
    if (localModelFetchGuard.isCurrent(requestId)) {
      isFetchingLocalModels.value = false
    }
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
        baseUrl: '',
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
          baseUrl,
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
    localSettings.value.promptContent = forceJsonOutput
      ? DEFAULT_SINGLE_BUBBLE_JSON_PROMPT
      : DEFAULT_SINGLE_BUBBLE_PROMPT
  } else {
    localSettings.value.promptContent = forceJsonOutput
      ? DEFAULT_TRANSLATE_JSON_PROMPT
      : DEFAULT_TRANSLATE_PROMPT
  }
  toast.success('已重置为默认提示词')
}
</script>

<style scoped>
.translation-settings__prompt-mode-hint {
  color: var(--color-text-supporting);
  font-size: 12px;
}
</style>
