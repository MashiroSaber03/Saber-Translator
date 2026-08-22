<template>
  <div class="ocr-settings">
    <ProductFormSection>
      <template #title>OCR引擎选择</template>
      <UiField variant="settings" label="OCR引擎" control-id="settingsOcrEngine">
        <UiSelect
          id="settingsOcrEngine"
          :model-value="settings.ocrEngine"
          :options="ocrEngineOptions"
          @change="handleOcrEngineChange"
        />
      </UiField>
    </ProductFormSection>
    <ProductFormSection>
      <template #title>混合OCR设置</template>
      <UiField
        variant="settings"
        control="checkbox"
        label="启用混合OCR"
        control-id="settingsHybridOcrEnabled"
        :hint="
          settings.hybridOcr.enabled
            ? '首批混合OCR仅支持 MangaOCR / 48px OCR，推荐顺序为 48px OCR → MangaOCR。启用后会优先走 textline 级专用链路。'
            : ''
        "
      >
        <UiCheckbox
          input-id="settingsHybridOcrEnabled"
          :model-value="settings.hybridOcr.enabled"
          @change="handleHybridOcrEnabledChange"
        />
      </UiField>
      <UiFormGrid v-show="settings.hybridOcr.enabled">
        <UiField variant="settings" label="备用OCR" control-id="settingsHybridSecondaryOcr">
          <UiSelect
            id="settingsHybridSecondaryOcr"
            :model-value="settings.hybridOcr.secondaryEngine"
            :options="hybridSecondaryEngineOptions"
            @change="handleHybridSecondaryEngineChange"
          />
        </UiField>
      </UiFormGrid>
      <UiFormGrid v-show="settings.hybridOcr.enabled">
        <UiField variant="settings" label="混合阈值" control-id="settingsHybridThreshold">
          <UiNumberField
            input-id="settingsHybridThreshold"
            :model-value="settings.hybridOcr.confidenceThreshold"
            :min="0"
            :max="1"
            :step="0.01"
            @change="handleHybridThresholdChange"
          />
        </UiField>
      </UiFormGrid>
    </ProductFormSection>
    <ProductFormSection v-show="settings.ocrEngine === 'paddleocr_vl'">
      <template #title>PaddleOCR-VL 1.6 设置</template>
      <UiField
        variant="settings"
        label="源语言"
        control-id="settingsPaddleOcrVlSourceLanguage"
        hint="根据所选语言生成 PaddleOCR-VL 识别提示词"
      >
        <UiCombobox
          input-id="settingsPaddleOcrVlSourceLanguage"
          aria-label="PaddleOCR-VL 源语言"
          :model-value="settings.paddleOcrVl.sourceLanguage"
          :groups="paddleOcrVlSourceLanguageGroups"
          @change="handlePaddleOcrVlSourceLanguageChange"
        />
      </UiField>
    </ProductFormSection>
    <ProductFormSection v-show="settings.ocrEngine === 'baidu_ocr'">
      <template #title>百度OCR 设置</template>
      <UiFormGrid>
        <UiField
          variant="settings"
          label="API Key"
          control-id="settingsBaiduApiKey"
          :hint="baiduStoredCredentialHint"
        >
          <UiPasswordField
            input-id="settingsBaiduApiKey"
            :model-value="settings.baiduOcr.apiKey"
            :placeholder="baiduStoredCredentialPlaceholder || '请输入百度OCR API Key'"
            show-label="显示百度 API Key"
            hide-label="隐藏百度 API Key"
            @update:model-value="updateBaiduString('apiKey', $event)"
          />
        </UiField>
        <UiField
          variant="settings"
          label="Secret Key"
          control-id="settingsBaiduSecretKey"
          :hint="baiduStoredCredentialHint"
        >
          <UiPasswordField
            input-id="settingsBaiduSecretKey"
            :model-value="settings.baiduOcr.secretKey"
            :placeholder="baiduStoredCredentialPlaceholder || '请输入Secret Key'"
            show-label="显示百度 Secret Key"
            hide-label="隐藏百度 Secret Key"
            @update:model-value="updateBaiduString('secretKey', $event)"
          />
        </UiField>
      </UiFormGrid>
      <UiFormGrid>
        <UiField variant="settings" label="识别版本" control-id="settingsBaiduVersion">
          <UiSelect
            id="settingsBaiduVersion"
            :model-value="settings.baiduOcr.version"
            :options="baiduVersionOptions"
            @change="updateBaiduSelect('version', $event)"
          />
        </UiField>
        <UiField variant="settings" label="源语言" control-id="settingsBaiduSourceLanguage">
          <UiSelect
            id="settingsBaiduSourceLanguage"
            :model-value="settings.baiduOcr.sourceLanguage"
            :options="baiduSourceLanguageOptions"
            @change="updateBaiduSelect('sourceLanguage', $event)"
          />
        </UiField>
      </UiFormGrid>
      <ProductActionRow aria-label="百度 OCR 操作" justify="start">
        <UiButton variant="secondary" tone="info" @click="testBaiduOcr" :disabled="isTesting">
          <span v-if="isTesting">测试中...</span>
          <template v-else>
            <span aria-hidden="true">🔗</span>
            <span>测试连接</span>
          </template>
        </UiButton>
      </ProductActionRow>
    </ProductFormSection>
    <ProductFormSection v-show="settings.ocrEngine === 'ai_vision'">
      <template #title>AI视觉OCR 设置</template>
      <UiFormGrid>
        <AiProviderSelectField
          :model-value="settings.aiVisionOcr.provider"
          input-id="settingsAiVisionProvider"
          :options="aiVisionProviderOptions"
          @change="handleAiVisionProviderChange"
        />
        <AiProviderCredentialFields
          :api-key="settings.aiVisionOcr.apiKey"
          api-key-input-id="settingsAiVisionApiKey"
          :base-url="settings.aiVisionOcr.customBaseUrl"
          base-url-input-id="settingsCustomAiVisionBaseUrl"
          :show-api-key="providerRequiresApiKey(settings.aiVisionOcr.provider)"
          :show-base-url="false"
          :include-base-url="false"
          :api-key-placeholder="aiVisionApiKeyRequired ? '请输入API Key' : '本地无鉴权服务可留空'"
          :has-stored-credential="
            settingsStore.hasCredential('ai_vision_ocr', settings.aiVisionOcr.provider)
          "
          api-key-show-label="显示 AI 视觉 API Key"
          api-key-hide-label="隐藏 AI 视觉 API Key"
          @update:api-key="updateAiVisionString('apiKey', $event)"
        />
      </UiFormGrid>
      <AiProviderCredentialFields
        :api-key="settings.aiVisionOcr.apiKey"
        api-key-input-id="settingsAiVisionApiKey"
        :base-url="settings.aiVisionOcr.customBaseUrl"
        base-url-input-id="settingsCustomAiVisionBaseUrl"
        :show-api-key="false"
        :show-base-url="providerRequiresBaseUrl(settings.aiVisionOcr.provider)"
        :include-api-key="false"
        base-url-placeholder="例如: https://api.example.com/v1"
        @update:base-url="updateAiVisionString('customBaseUrl', $event)"
      />
      <UiField variant="settings" label="模型名称" control-id="settingsAiVisionModelName">
        <UiModelPicker
          input-id="settingsAiVisionModelName"
          :model-value="settings.aiVisionOcr.modelName"
          placeholder="如: silicon-llava2-34b"
          fetch-variant="primary"
          :fetching="isFetchingModels"
          :fetch-disabled="isFetchingModels"
          :options="aiVisionModelOptions"
          :model-count="aiVisionModels.length"
          @update:model-value="updateAiVisionModel"
          @fetch="fetchAiVisionModels"
        />
      </UiField>
      <UiField variant="settings" label="OCR提示词" control-id="settingsAiVisionOcrPrompt">
        <UiTextarea
          id="settingsAiVisionOcrPrompt"
          :model-value="settings.aiVisionOcr.prompt"
          variant="panel"
          rows="3"
          placeholder="AI视觉OCR提示词"
          @update:model-value="updateAiVisionString('prompt', $event)"
        />
        <SavedPromptsPicker prompt-type="ai_vision_ocr" @select="handleAiVisionPromptSelect" />
        <ProductActionRow aria-label="AI 视觉 OCR 提示词格式" justify="start">
          <UiSelect
            :model-value="currentPromptMode"
            :options="promptModeOptions"
            @change="handlePromptModeChange"
          />
          <span class="ocr-settings__prompt-mode-hint">{{ getPromptModeHint() }}</span>
        </ProductActionRow>
        <UiField
          v-if="currentPromptMode === 'paddleocr_vl'"
          class="ocr-settings__prompt-language-field"
          variant="settings"
          label="源语言"
          control-id="settingsAiVisionPaddleOcrVlSourceLanguage"
          hint="切换后会按所选语言重新生成 OCR 模型提示词"
        >
          <UiCombobox
            class="ocr-settings__prompt-language-combobox"
            input-id="settingsAiVisionPaddleOcrVlSourceLanguage"
            aria-label="AI 视觉 OCR 专用模型源语言"
            :model-value="settings.paddleOcrVl.sourceLanguage"
            :groups="paddleOcrVlSourceLanguageGroups"
            @change="handlePaddleOcrVlSourceLanguageChange"
          />
        </UiField>
      </UiField>
      <UiField
        variant="settings"
        label="RPM限制 (每分钟请求数)"
        control-id="settingsRpmAiVisionOcr"
        hint="0 表示无限制"
      >
        <UiNumberField
          input-id="settingsRpmAiVisionOcr"
          :model-value="settings.aiVisionOcr.openaiOptions.execution.rpmLimit"
          :min="0"
          :max="100000"
          :step="1"
          @update:model-value="updateAiVisionNumber('rpmLimit', $event)"
        />
      </UiField>
      <UiField variant="settings" label="业务重试" control-id="settingsAiVisionBusinessRetries">
        <UiNumberField
          input-id="settingsAiVisionBusinessRetries"
          :model-value="settings.aiVisionOcr.openaiOptions.execution.businessRetries"
          :min="0"
          :max="100"
          :step="1"
          @update:model-value="updateAiVisionNumber('businessRetries', $event)"
        />
      </UiField>
      <UiField variant="settings" label="传输重试" control-id="settingsAiVisionTransportRetries">
        <UiNumberField
          input-id="settingsAiVisionTransportRetries"
          :model-value="settings.aiVisionOcr.openaiOptions.execution.transportRetries"
          :min="0"
          :max="100"
          :step="1"
          @update:model-value="updateAiVisionNumber('transportRetries', $event)"
        />
      </UiField>
      <UiField
        variant="settings"
        control="checkbox"
        label="流式调用"
        control-id="settingsAiVisionUseStream"
        hint="使用流式请求并在终端输出流式日志"
      >
        <UiCheckbox
          input-id="settingsAiVisionUseStream"
          :model-value="settings.aiVisionOcr.openaiOptions.execution.useStream"
          @update:model-value="updateAiVisionBoolean('useStream', $event)"
        />
      </UiField>
      <UiField variant="settings">
        <OpenAIExtraBodyEditor
          :model-value="settings.aiVisionOcr.openaiOptions.request.extraBody"
          @update:model-value="updateAiVisionExtraBody"
        />
      </UiField>
      <UiField
        variant="settings"
        label="最小图片尺寸 (像素)"
        control-id="settingsMinImageSize"
        hint="VLM模型通常要求图片尺寸 ≥28px，设为0则不自动放大小图"
      >
        <UiNumberField
          input-id="settingsMinImageSize"
          :model-value="settings.aiVisionOcr.minImageSize"
          :min="0"
          :step="1"
          @update:model-value="updateAiVisionNumber('minImageSize', $event)"
        />
      </UiField>
      <ProductActionRow aria-label="AI 视觉 OCR 操作" justify="start">
        <UiButton variant="secondary" tone="info" @click="testAiVisionOcr" :disabled="isTesting">
          <span v-if="isTesting">测试中...</span>
          <template v-else>
            <span aria-hidden="true">🔗</span>
            <span>测试连接</span>
          </template>
        </UiButton>
      </ProductActionRow>
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
import UiPasswordField from '@/components/ui/UiPasswordField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import AiProviderCredentialFields from '@/components/settings/AiProviderCredentialFields.vue'
import AiProviderSelectField from '@/components/settings/AiProviderSelectField.vue'
import type { UiSelectValue } from '@/components/ui/selectTypes'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import { ref, computed } from 'vue'
import {
  normalizeProviderId,
  providerRequiresApiKey,
  providerRequiresApiKeyForBaseUrl,
  providerRequiresBaseUrl,
  providerSupportsCapability,
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settings'
import {
  fetchModels as fetchV2Models,
  testAiVisionOcrConnection,
  testBaiduOcrConnection,
} from '@/api/v2/diagnostics'
import { useToast } from '@/utils/toast'
import {
  inferPaddleOcrVlPromptLanguage,
  isPaddleOcrVlLanguage,
} from '@/constants'
import type { OcrEngine } from '@/types/settings'
import UiCombobox from '@/components/ui/UiCombobox.vue'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'
import {
  isSupportedHybridOcrEngine,
  SUPPORTED_HYBRID_OCR_ENGINES,
} from '@/utils/hybridOcr'
import {
  aiVisionProviderOptions,
  allOcrEngineOptions,
  baiduSourceLanguageOptions,
  baiduVersionOptions,
  paddleOcrVlSourceLanguageGroups,
  promptModeOptions,
} from './ocrSettingsOptions'
import {
  useAiModelDiscovery,
  type AiModelDiscoveryMessageTone,
} from '@/composables/useAiModelDiscovery'
import { usePublicUserAccess } from '@/composables/usePublicUserAccess'
const settingsStore = useSettingsStore()
const toast = useToast()
const publicAccess = usePublicUserAccess()
const settings = computed(() => settingsStore.settings)
const hasStoredBaiduCredential = computed(() => settingsStore.hasCredential('ocr', 'baidu'))
const baiduStoredCredentialHint = computed(() =>
  hasStoredBaiduCredential.value && !settings.value.baiduOcr.apiKey && !settings.value.baiduOcr.secretKey
    ? '百度 OCR 凭据已安全保存在后端；留空表示保持不变，更换时必须同时填写两项'
    : ''
)
const baiduStoredCredentialPlaceholder = computed(() =>
  baiduStoredCredentialHint.value ? '已保存在后端，留空保持不变' : ''
)
const isTesting = ref(false)
const aiVisionApiKeyRequired = computed(() => providerRequiresApiKeyForBaseUrl(
  settings.value.aiVisionOcr.provider,
  settings.value.aiVisionOcr.customBaseUrl,
))
function notifyModelDiscovery(message: string, tone: AiModelDiscoveryMessageTone): void {
  toast[tone](message)
}
const aiVisionModelDiscovery = useAiModelDiscovery({
  source: () => ({
    provider: settingsStore.settings.aiVisionOcr.provider,
    apiKey: settingsStore.settings.aiVisionOcr.apiKey,
    baseUrl: settingsStore.settings.aiVisionOcr.customBaseUrl,
    hasStoredCredential: settingsStore.hasCredential(
      'ai_vision_ocr',
      settingsStore.settings.aiVisionOcr.provider
    ),
  }),
  fetcher: (provider, apiKey, baseUrl) => fetchV2Models(provider, apiKey, baseUrl, 'ai_vision_ocr'),
  requiresApiKey: provider => providerRequiresApiKeyForBaseUrl(
    provider,
    settingsStore.settings.aiVisionOcr.customBaseUrl,
  ),
  notify: notifyModelDiscovery,
  emptyMessage: () => '服务未返回模型列表，也可以直接填写模型名称',
  errorMessage: error => `${error instanceof Error ? error.message : '获取模型列表失败'}；也可以直接填写模型名称`,
  emptyBaseUrl: '',
})
const { isFetchingModels } = aiVisionModelDiscovery
const aiVisionModels = computed(() => aiVisionModelDiscovery.models.value.map(model => model.id))
const aiVisionModelOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  aiVisionModelDiscovery.models.value.forEach(model => {
    options.push({ label: model.id, value: model.id })
  })
  return options
})
const policyOcrEngineOptions = computed(() => publicAccess.modelOptions(allOcrEngineOptions, {
  manga_ocr: 'manga_ocr',
  paddle_ocr: 'paddle_ocr',
  paddleocr_vl: 'paddleocr_vl',
  '48px_ocr': 'ocr_48px',
}))
const ocrEngineOptions = computed(() => {
  if (!settings.value.hybridOcr.enabled) {
    return policyOcrEngineOptions.value
  }
  const supported = new Set<string>(SUPPORTED_HYBRID_OCR_ENGINES)
  return policyOcrEngineOptions.value.filter(option => supported.has(String(option.value)))
})
const hybridSecondaryEngineOptions = computed(() =>
  policyOcrEngineOptions.value
    .filter(
      option =>
        isSupportedHybridOcrEngine(option.value) && option.value !== settings.value.ocrEngine
    )
)
function isOcrEngine(value: unknown): value is OcrEngine {
  if (typeof value !== 'string') return false
  return allOcrEngineOptions.some(option => option.value === value)
}
function handleOcrEngineChange(value: UiSelectValue) {
  if (isOcrEngine(value)) {
    settingsStore.setOcrEngine(value)
  }
}
function handleHybridOcrEnabledChange(value: boolean) {
  settingsStore.updateHybridOcr({ enabled: value })
}
function handleHybridSecondaryEngineChange(value: UiSelectValue) {
  if (typeof value !== 'string' || !isSupportedHybridOcrEngine(value)) return
  settingsStore.updateHybridOcr({ secondaryEngine: value })
}
function handleHybridThresholdChange(value: number | null) {
  if (value === null || !Number.isFinite(value) || value < 0 || value > 1) return
  settingsStore.updateHybridOcr({ confidenceThreshold: value })
}
function handlePaddleOcrVlSourceLanguageChange(value: UiSelectValue) {
  if (!isPaddleOcrVlLanguage(value)) return
  settingsStore.updatePaddleOcrVl({ sourceLanguage: value })
}
function updateBaiduString(field: 'apiKey' | 'secretKey', value: string): void {
  settingsStore.updateBaiduOcr({ [field]: value })
}
function updateBaiduSelect(
  field: 'version' | 'sourceLanguage',
  value: UiSelectValue,
): void {
  if (typeof value !== 'string') return
  settingsStore.updateBaiduOcr({ [field]: value })
}
function updateAiVisionString(
  field: 'apiKey' | 'customBaseUrl' | 'prompt',
  value: string,
): void {
  settingsStore.updateAiVisionOcr({ [field]: value })
}
function updateAiVisionModel(value: UiSelectValue): void {
  if (typeof value !== 'string') return
  settingsStore.updateAiVisionOcr({ modelName: value })
}
function updateAiVisionNumber(
  field: 'rpmLimit' | 'businessRetries' | 'transportRetries' | 'minImageSize',
  value: number | null,
): void {
  if (value === null) return
  settingsStore.updateAiVisionOcr({ [field]: value })
}
function updateAiVisionBoolean(field: 'useStream', value: boolean): void {
  settingsStore.updateAiVisionOcr({ [field]: value })
}
function updateAiVisionExtraBody(value: Record<string, unknown> | undefined): void {
  settingsStore.updateAiVisionOcr({ extraBody: value })
}
function handleAiVisionProviderChange(providerValue: UiSelectValue) {
  if (typeof providerValue !== 'string') return
  const newProvider = normalizeProviderId(providerValue)
  if (!providerSupportsCapability(newProvider, 'visionOcr')) return
  aiVisionModelDiscovery.invalidate()
  settingsStore.setAiVisionOcrProvider(newProvider)
}
const currentPromptMode = computed(() => {
  return settingsStore.settings.aiVisionOcr.promptMode
})
function getPromptModeHint(): string {
  switch (currentPromptMode.value) {
    case 'paddleocr_vl':
      return 'PaddleOCR-VL、GLM-OCR 等专用 OCR 模型专用提示词'
    case 'json':
      return 'JSON 格式输出更结构化'
    default:
      return '通用 VLM 提示词，若使用 PaddleOCR-VL、GLM-OCR 等专用模型，请选择「OCR模型提示词」'
  }
}
function handlePromptModeChange(mode: UiSelectValue) {
  if (mode !== 'normal' && mode !== 'json' && mode !== 'paddleocr_vl') return
  settingsStore.setAiVisionOcrPromptMode(mode)
}
async function testBaiduOcr() {
  const apiKey = settings.value.baiduOcr.apiKey?.trim()
  const secretKey = settings.value.baiduOcr.secretKey?.trim()
  if (Boolean(apiKey) !== Boolean(secretKey)) {
    toast.warning('更换百度 OCR 凭据时必须同时填写 API Key 和 Secret Key')
    return
  }
  if ((!apiKey || !secretKey) && !settingsStore.hasCredential('ocr', 'baidu')) {
    toast.warning('请填写百度OCR的API Key和Secret Key')
    return
  }
  isTesting.value = true
  toast.info('正在测试百度OCR连接...')
  try {
    const result = await testBaiduOcrConnection(apiKey, secretKey)
    if (result.success) {
      toast.success(result.message || '百度OCR连接成功!')
    } else {
      toast.error(result.message || '百度OCR连接失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '连接测试失败'
    toast.error(errorMessage)
  } finally {
    isTesting.value = false
  }
}
async function testAiVisionOcr() {
  const provider = settingsStore.settings.aiVisionOcr.provider
  const apiKey = settings.value.aiVisionOcr.apiKey?.trim()
  const modelName = settings.value.aiVisionOcr.modelName?.trim()
  const customBaseUrl = settings.value.aiVisionOcr.customBaseUrl?.trim()
  if (
    providerRequiresApiKeyForBaseUrl(provider, customBaseUrl)
    && !apiKey
    && !settingsStore.hasCredential('ai_vision_ocr', provider)
  ) {
    toast.warning('请先填写 API Key')
    return
  }
  if (!modelName) {
    toast.warning('请填写模型名称')
    return
  }
  if (providerRequiresBaseUrl(provider) && !customBaseUrl) {
    toast.warning('自定义服务需要填写 Base URL')
    return
  }
  isTesting.value = true
  try {
    const result = await testAiVisionOcrConnection({
      provider,
      apiKey,
      modelName,
      customBaseUrl,
      prompt: settings.value.aiVisionOcr.prompt,
      domain: 'ai_vision_ocr',
    })
    if (result.success) {
      toast.success('AI视觉OCR连接成功')
    } else {
      toast.error(`AI视觉OCR连接失败: ${result.message || '未知错误'}`)
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '连接测试失败'
    toast.error(errorMessage)
  } finally {
    isTesting.value = false
  }
}
const fetchAiVisionModels = aiVisionModelDiscovery.fetchModels
function handleAiVisionPromptSelect(content: string, name: string) {
  settingsStore.updateAiVisionOcr({ prompt: content })
  if (currentPromptMode.value === 'paddleocr_vl') {
    const currentLanguage = settings.value.paddleOcrVl.sourceLanguage
    const inferredLanguage = inferPaddleOcrVlPromptLanguage(content, currentLanguage)
    if (inferredLanguage !== currentLanguage) {
      settingsStore.updatePaddleOcrVl({ sourceLanguage: inferredLanguage })
    }
  }
  toast.success(`已应用提示词: ${name}`)
}
</script>

<style scoped>
.ocr-settings__prompt-mode-hint {
  color: var(--color-text-supporting);
  font-size: 0.85em;
  line-height: 1.45;
}

.ocr-settings__prompt-language-field {
  margin-top: 10px;
  padding: 10px 12px;
  background: var(--color-surface-subtle);
  border: 1px solid var(--color-border-muted);
  border-radius: 6px;
}

.ocr-settings__prompt-language-combobox {
  min-width: 150px;
}

</style>
