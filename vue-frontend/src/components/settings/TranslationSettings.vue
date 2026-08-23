<template>
  <div class="translation-settings">
    <ProductFormSection>
      <template #title>翻译服务配置</template>
      <UiFormGrid>
        <AiProviderSelectField
          :model-value="modelProvider"
          input-id="settingsModelProvider"
          :options="providerOptions"
          label="翻译服务商"
          custom-profile-kind="chatVision"
          :custom-profile-api-key="translationSettings.apiKey"
          :custom-profile-base-url="translationSettings.customBaseUrl"
          :custom-profile-model="translationSettings.modelName"
          @change="handleProviderSelect"
          @apply-custom-profile="applyCustomProfile"
        />
        <AiProviderCredentialFields
          :api-key="translationSettings.apiKey"
          api-key-input-id="settingsApiKey"
          :base-url="translationSettings.customBaseUrl"
          base-url-input-id="settingsCustomBaseUrl"
          :show-api-key="!isLocalProvider"
          :show-base-url="false"
          :include-base-url="false"
          :api-key-label="apiKeyLabel"
          :api-key-placeholder="apiKeyPlaceholder"
          api-key-show-label="显示翻译 API Key"
          api-key-hide-label="隐藏翻译 API Key"
          @update:api-key="updateTranslationString('apiKey', $event)"
        />
      </UiFormGrid>
      <AiProviderCredentialFields
        :api-key="translationSettings.apiKey"
        api-key-input-id="settingsApiKey"
        :base-url="translationSettings.customBaseUrl"
        base-url-input-id="settingsCustomBaseUrl"
        :show-api-key="false"
        :show-base-url="providerRequiresBaseUrl(modelProvider)"
        :include-api-key="false"
        base-url-placeholder="例如: https://api.example.com/v1"
        @update:base-url="updateTranslationString('customBaseUrl', $event)"
      />
      <UiField
        v-show="!isLocalProvider"
        variant="settings"
        :label="modelNameLabel"
        control-id="settingsModelName"
      >
        <UiModelPicker
          input-id="settingsModelName"
          :model-value="translationSettings.modelName"
          :placeholder="modelNamePlaceholder"
          :show-fetch="supportsFetchModels"
          fetch-variant="primary"
          :fetching="isFetchingModels"
          :fetch-disabled="isFetchingModels"
          :options="modelListOptions"
          :model-count="modelList.length"
          @update:model-value="handleModelSelect"
          @fetch="fetchModels"
        />
      </UiField>
      <UiField
        v-show="isLocalProvider"
        variant="settings"
        label="模型名称"
        control-id="settingsLocalModelName"
      >
        <UiModelPicker
          input-id="settingsLocalModelName"
          :model-value="translationSettings.modelName"
          :placeholder="
            modelProvider === 'ollama'
              ? '例如: qwen2.5:7b'
              : '例如: sakura-14b-qwen2.5-v1.0'
          "
          fetch-title="获取本地可用模型列表"
          fetch-variant="primary"
          :fetching="isFetchingModels"
          :fetch-disabled="isFetchingModels"
          :options="localModelListOptions"
          :model-count="localModelList.length"
          @update:model-value="handleModelSelect"
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
            :model-value="translationSettings.openaiOptions.execution.rpmLimit"
            :min="0"
            :max="100000"
            :step="1"
            @update:model-value="updateTranslationNumber('rpmLimit', $event)"
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
            :model-value="translationSettings.openaiOptions.execution.businessRetries"
            :min="0"
            :max="100"
            :step="1"
            @update:model-value="updateTranslationNumber('businessRetries', $event)"
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
            :model-value="translationSettings.openaiOptions.execution.transportRetries"
            :min="0"
            :max="100"
            :step="1"
            @update:model-value="updateTranslationNumber('transportRetries', $event)"
          />
        </UiField>
      </UiFormGrid>
      <UiField
        v-show="showRpmLimit"
        variant="settings"
        control="checkbox"
        hint="同时作用于整页批量和逐气泡翻译"
      >
        <UiCheckbox
          :model-value="translationSettings.openaiOptions.execution.useStream"
          label="流式调用"
          @update:model-value="updateTranslationBoolean('useStream', $event)"
        />
      </UiField>
      <UiField v-show="showRpmLimit" variant="settings">
        <OpenAIExtraBodyEditor
          :model-value="translationSettings.openaiOptions.request.extraBody"
          @update:model-value="updateTranslationExtraBody"
        />
      </UiField>
      <UiField
        variant="settings"
        label="翻译模式"
        control-id="settingsTranslationMode"
        :hint="translationModeHint"
      >
        <UiSelect
          id="settingsTranslationMode"
          :model-value="translationSettings.translationMode"
          :options="translationModeOptions"
          @change="handleTranslationModeChange"
        />
      </UiField>
      <ProductStatusBanner
        v-if="modelProvider === 'sakura'"
        tone="warning"
        role="note"
      >
        建议 Sakura 服务使用"逐气泡翻译"模式，可获得更稳定的翻译效果
      </ProductStatusBanner>
      <ProductActionRow v-show="isLocalProvider" aria-label="本地翻译连接测试" justify="start">
        <UiButton variant="secondary" tone="info" @click="testLocalConnection" :disabled="isTesting">
          <span v-if="isTesting">测试中...</span>
          <template v-else>
            <span aria-hidden="true">🔗</span>
            <span>测试连接</span>
          </template>
        </UiButton>
      </ProductActionRow>
      <ProductActionRow v-show="!isLocalProvider" aria-label="云端翻译连接测试" justify="start">
        <UiButton variant="secondary" tone="info" @click="testCloudConnection" :disabled="isTesting">
          <span v-if="isTesting">测试中...</span>
          <template v-else>
            <span aria-hidden="true">🔗</span>
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
          :model-value="currentPrompt"
          variant="panel"
          rows="4"
          placeholder="翻译提示词"
          @update:model-value="settingsStore.setTranslatePrompt"
        />
        <ProductActionRow aria-label="翻译提示词格式" justify="start">
          <UiSelect
            :model-value="translatePromptMode"
            :options="promptModeOptions"
            @change="handlePromptModeSelect"
          />
          <span class="translation-settings__prompt-mode-hint">JSON格式输出更结构化</span>
        </ProductActionRow>
        <SavedPromptsPicker prompt-type="translate" @select="handleTranslatePromptSelect" />
        <ProductActionRow aria-label="翻译提示词操作" justify="start">
          <UiButton
            variant="secondary"
            type="button"
            size="sm"
            @click="resetTranslatePromptToDefault"
          >
            重置为默认
          </UiButton>
        </ProductActionRow>
      </UiField>
      <UiField variant="settings" control="checkbox">
        <UiCheckbox
          :model-value="settings.useTextboxPrompt"
          label="启用文本框提示词"
          @update:model-value="settingsStore.setUseTextboxPrompt"
        />
      </UiField>
      <UiField
        v-show="settings.useTextboxPrompt"
        variant="settings"
        label="文本框提示词"
        control-id="settingsTextboxPromptContent"
      >
        <UiTextarea
          id="settingsTextboxPromptContent"
          :model-value="settings.textboxPrompt"
          variant="panel"
          rows="3"
          placeholder="文本框提示词"
          @update:model-value="settingsStore.setTextboxPrompt"
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
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import AiProviderCredentialFields from '@/components/settings/AiProviderCredentialFields.vue'
import AiProviderSelectField from '@/components/settings/AiProviderSelectField.vue'
import type { UiSelectValue } from '@/components/ui/selectTypes'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import { ref, computed } from 'vue'
import {
  getProviderDisplayName as getProviderDisplayNameFromManifest,
  providerSupportsRpmLimit,
  getProviderOptionsForCapability,
  isLocalProviderId,
  normalizeProviderId,
  providerRequiresApiKeyForBaseUrl,
  providerRequiresBaseUrl,
  providerSupportsCapability,
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settings'
import {
  fetchModels as fetchV2Models,
  testAiTranslateConnection,
  testBaiduTranslateConnection,
  testSakuraConnection,
  testYoudaoTranslateConnection,
} from '@/api/v2/diagnostics'
import { useToast } from '@/utils/toast'
import {
  DEFAULT_TRANSLATE_PROMPT,
  DEFAULT_TRANSLATE_JSON_PROMPT,
  DEFAULT_SINGLE_BUBBLE_PROMPT,
  DEFAULT_SINGLE_BUBBLE_JSON_PROMPT,
} from '@/constants'
import type { TranslationMode, TranslationProvider } from '@/types/settings'
import type { CustomAiProfile } from '@/types/customAiProfile'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'
import { useLatestRequestGuard } from '@/composables/useLatestRequestGuard'
import {
  useAiModelDiscovery,
  type AiModelDiscoveryMessageTone,
} from '@/composables/useAiModelDiscovery'
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
const settings = computed(() => settingsStore.settings)
const translationSettings = computed(() => settings.value.translation)
const modelProvider = computed(() => normalizeProviderId(translationSettings.value.provider))
const translatePromptMode = computed(() => (
  translationSettings.value.openaiOptions.request.forceJsonOutput ? 'json' : 'normal'
))
const currentPrompt = computed(() => {
  const translation = translationSettings.value
  const useJson = translation.openaiOptions.request.forceJsonOutput
  if (translation.translationMode === 'single') {
    return useJson ? translation.singleJsonPrompt : translation.singleNormalPrompt
  }
  return useJson ? translation.batchJsonPrompt : translation.batchNormalPrompt
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
    provider: modelProvider.value,
    apiKey: translationSettings.value.apiKey,
    baseUrl: translationSettings.value.customBaseUrl,
  }),
  fetcher: (provider, apiKey, baseUrl) => fetchV2Models(provider, apiKey, baseUrl, 'translation'),
  notify: notifyModelDiscovery,
  supportsProvider: provider =>
    providerSupportsCapability(provider, 'modelFetch') && !isLocalProviderId(provider),
  requiresApiKey: providerRequiresApiKeyForBaseUrl,
  emptyBaseUrl: '',
})
const isFetchingModels = computed(
  () => remoteModelDiscovery.isFetchingModels.value || isFetchingLocalModels.value
)
const modelList = computed(() => remoteModelDiscovery.models.value.map(model => model.id))
const modelListOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  remoteModelDiscovery.models.value.forEach(model =>
    options.push({ label: model.id, value: model.id })
  )
  return options
})
const localModelListOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  localModelList.value.forEach(model => options.push({ label: model, value: model }))
  return options
})
const isLocalProvider = computed(() => {
  return isLocalProviderId(modelProvider.value)
})
const showRpmLimit = computed(() => {
  return providerSupportsRpmLimit(modelProvider.value)
})
const supportsFetchModels = computed(() => {
  return (
    providerSupportsCapability(modelProvider.value, 'modelFetch') &&
    !isLocalProviderId(modelProvider.value)
  )
})
const apiKeyLabel = computed(() => getTranslationApiKeyLabel(modelProvider.value))
const apiKeyPlaceholder = computed(() =>
  getTranslationApiKeyPlaceholder(modelProvider.value)
)
const modelNameLabel = computed(() =>
  getTranslationModelNameLabel(modelProvider.value)
)
const modelNamePlaceholder = computed(() =>
  getTranslationModelNamePlaceholder(modelProvider.value)
)
const translationModeHint = computed(() =>
  translationSettings.value.translationMode === 'batch'
    ? '整页批量翻译：一次发送全部气泡，效率高，需要模型支持复杂指令'
    : '逐气泡翻译：每个气泡单独翻译，更稳定，适合小模型或格式敏感场景'
)
function handleProviderSelect(value: UiSelectValue) {
  if (typeof value !== 'string' || !providerSupportsCapability(value, 'translation')) return
  const newProvider = normalizeProviderId(value)
  if (!providerSupportsCapability(newProvider, 'translation')) return
  invalidateModelFetchRequests()
  settingsStore.setTranslationProvider(newProvider as TranslationProvider)
  settingsStore.setTranslatePromptMode(
    settingsStore.settings.translation.openaiOptions.request.forceJsonOutput,
  )
}
function handleModelSelect(value: UiSelectValue) {
  if (typeof value === 'string') settingsStore.updateTranslationService({ modelName: value })
}
function handlePromptModeSelect(value: UiSelectValue) {
  if (value !== 'normal' && value !== 'json') return
  if (value === translatePromptMode.value) return
  settingsStore.setTranslatePromptMode(value === 'json')
}

function updateTranslationString(
  field: 'apiKey' | 'customBaseUrl',
  value: string,
): void {
  settingsStore.updateTranslationService({ [field]: value })
}

function applyCustomProfile(profile: CustomAiProfile): void {
  settingsStore.updateTranslationService({
    apiKey: profile.apiKey,
    customBaseUrl: profile.baseUrl,
    modelName: profile.model,
  })
}

function updateTranslationNumber(
  field: 'rpmLimit' | 'businessRetries' | 'transportRetries',
  value: number | null,
): void {
  if (value === null) return
  settingsStore.updateTranslationService({ [field]: value })
}

function updateTranslationBoolean(field: 'useStream', value: boolean): void {
  settingsStore.updateTranslationService({ [field]: value })
}

function updateTranslationExtraBody(value: Record<string, unknown> | undefined): void {
  settingsStore.updateTranslationService({ extraBody: value })
}

function invalidateModelFetchRequests() {
  remoteModelDiscovery.invalidate()
  localModelFetchGuard.invalidate()
  isFetchingLocalModels.value = false
  localModelList.value = []
}

function handleTranslationModeChange(value: UiSelectValue) {
  if (value !== 'single' && value !== 'batch') return
  const newMode: TranslationMode = value
  const previousMode = translationSettings.value.translationMode
  if (newMode === previousMode) return
  settingsStore.updateTranslationService({ translationMode: newMode })
  settingsStore.setTranslatePromptMode(translatePromptMode.value === 'json')
}
const fetchModels = remoteModelDiscovery.fetchModels
function getProviderDisplayName(provider: string): string {
  return getProviderDisplayNameFromManifest(provider)
}

async function fetchLocalModels() {
  const provider = modelProvider.value
  const requestId = localModelFetchGuard.next()
  isFetchingLocalModels.value = true
  try {
    if (provider === 'sakura' || provider === 'ollama') {
      const result = await fetchV2Models(provider, '', '')
      if (!localModelFetchGuard.isCurrent(requestId)) return
      if (result.models.length) {
        localModelList.value = result.models.map(model => model.id)
        toast.success(
          `获取到 ${result.models.length} 个${provider === 'sakura' ? 'Sakura' : 'Ollama'}模型`
        )
      } else {
        toast.error('未获取到可用的本地模型')
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
  const provider = modelProvider.value
  const modelName = translationSettings.value.modelName?.trim()
  if (provider === 'ollama' && !modelName) {
    toast.warning('请填写模型名称')
    return
  }
  isTesting.value = true
  try {
    let result
    if (provider === 'sakura') {
      result = await testSakuraConnection()
    } else if (provider === 'ollama') {
      result = await testAiTranslateConnection({
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
      toast.error(result.message || '连接失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '连接测试失败'
    toast.error(errorMessage)
  } finally {
    isTesting.value = false
  }
}
async function testCloudConnection() {
  const provider = modelProvider.value
  const apiKey = translationSettings.value.apiKey?.trim()
  const modelName = translationSettings.value.modelName?.trim()
  const baseUrl = translationSettings.value.customBaseUrl?.trim()
  if (
    providerRequiresApiKeyForBaseUrl(provider, baseUrl) &&
    !apiKey
  ) {
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
        result = await testBaiduTranslateConnection(apiKey, modelName)
        break
      case 'youdao_translate':
        result = await testYoudaoTranslateConnection(apiKey, modelName)
        break
      default:
        result = await testAiTranslateConnection({
          provider,
          apiKey,
          modelName,
          baseUrl,
          domain: 'translation',
        })
    }
    if (result.success) {
      toast.success(result.message || `${getProviderDisplayName(provider)} 连接成功!`)
    } else {
      toast.error(result.message || '连接失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '连接测试失败'
    toast.error(errorMessage)
  } finally {
    isTesting.value = false
  }
}
function handleTranslatePromptSelect(content: string, name: string) {
  settingsStore.setTranslatePrompt(content)
  toast.success(`已应用提示词: ${name}`)
}
function handleTextboxPromptSelect(content: string, name: string) {
  settingsStore.setTextboxPrompt(content)
  toast.success(`已应用提示词: ${name}`)
}
function resetTranslatePromptToDefault() {
  const forceJsonOutput = translatePromptMode.value === 'json'
  let prompt: string
  if (translationSettings.value.translationMode === 'single') {
    prompt = forceJsonOutput
      ? DEFAULT_SINGLE_BUBBLE_JSON_PROMPT
      : DEFAULT_SINGLE_BUBBLE_PROMPT
  } else {
    prompt = forceJsonOutput
      ? DEFAULT_TRANSLATE_JSON_PROMPT
      : DEFAULT_TRANSLATE_PROMPT
  }
  settingsStore.setTranslatePrompt(prompt)
  toast.success('已重置为默认提示词')
}
</script>

<style scoped>
.translation-settings__prompt-mode-hint {
  color: var(--color-text-supporting);
  font-size: 12px;
}
</style>
