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
      <UiField
        v-show="settings.ocrEngine === 'paddle_ocr'"
        variant="settings"
        label="源语言"
        control-id="settingsSourceLanguage"
        :hint="getSourceLanguageHint()"
      >
        <UiCombobox
          input-id="settingsSourceLanguage"
          aria-label="源语言"
          :model-value="settings.sourceLanguage"
          :groups="sourceLanguageGroups"
          @change="handleSourceLanguageSelect"
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
      <template #title>PaddleOCR-VL 设置</template>
      <UiField
        variant="settings"
        label="源语言"
        control-id="settingsPaddleOcrVlSourceLanguage"
        hint="选择图像中的源语言，用于优化 OCR 识别效果"
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
            v-model="localBaiduOcr.apiKey"
            :placeholder="baiduStoredCredentialPlaceholder || '请输入百度OCR API Key'"
            show-label="显示百度 API Key"
            hide-label="隐藏百度 API Key"
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
            v-model="localBaiduOcr.secretKey"
            :placeholder="baiduStoredCredentialPlaceholder || '请输入Secret Key'"
            show-label="显示百度 Secret Key"
            hide-label="隐藏百度 Secret Key"
          />
        </UiField>
      </UiFormGrid>
      <UiFormGrid>
        <UiField variant="settings" label="识别版本" control-id="settingsBaiduVersion">
          <UiSelect
            id="settingsBaiduVersion"
            v-model="localBaiduOcr.version"
            :options="baiduVersionOptions"
          />
        </UiField>
        <UiField variant="settings" label="源语言" control-id="settingsBaiduSourceLanguage">
          <UiSelect
            id="settingsBaiduSourceLanguage"
            v-model="localBaiduOcr.sourceLanguage"
            :options="baiduSourceLanguageOptions"
          />
        </UiField>
      </UiFormGrid>
      <ProductActionRow aria-label="百度 OCR 操作" justify="start">
        <UiButton variant="secondary" @click="testBaiduOcr" :disabled="isTesting">
          <span v-if="isTesting">测试中...</span>
          <template v-else>
            <UiIcon name="link" />
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
          :api-key="localAiVisionOcr.apiKey"
          api-key-input-id="settingsAiVisionApiKey"
          :base-url="localAiVisionOcr.customBaseUrl"
          base-url-input-id="settingsCustomAiVisionBaseUrl"
          :show-api-key="providerRequiresApiKey(settings.aiVisionOcr.provider)"
          :show-base-url="false"
          :include-base-url="false"
          api-key-placeholder="请输入API Key"
          :has-stored-credential="
            settingsStore.hasCredential('ai_vision_ocr', settings.aiVisionOcr.provider)
          "
          api-key-show-label="显示 AI 视觉 API Key"
          api-key-hide-label="隐藏 AI 视觉 API Key"
          @update:api-key="localAiVisionOcr.apiKey = $event"
        />
      </UiFormGrid>
      <AiProviderCredentialFields
        :api-key="localAiVisionOcr.apiKey"
        api-key-input-id="settingsAiVisionApiKey"
        :base-url="localAiVisionOcr.customBaseUrl"
        base-url-input-id="settingsCustomAiVisionBaseUrl"
        :show-api-key="false"
        :show-base-url="providerRequiresBaseUrl(settings.aiVisionOcr.provider)"
        :include-api-key="false"
        base-url-placeholder="例如: https://api.example.com/v1"
        @update:base-url="localAiVisionOcr.customBaseUrl = $event"
      />
      <UiField variant="settings" label="模型名称" control-id="settingsAiVisionModelName">
        <UiModelPicker
          input-id="settingsAiVisionModelName"
          v-model="localAiVisionOcr.modelName"
          placeholder="如: silicon-llava2-34b"
          fetch-variant="primary"
          :fetching="isFetchingModels"
          :fetch-disabled="isFetchingModels"
          :options="aiVisionModelOptions"
          :model-count="aiVisionModels.length"
          @fetch="fetchAiVisionModels"
        />
      </UiField>
      <UiField variant="settings" label="OCR提示词" control-id="settingsAiVisionOcrPrompt">
        <UiTextarea
          id="settingsAiVisionOcrPrompt"
          v-model="localAiVisionOcr.prompt"
          variant="panel"
          rows="3"
          placeholder="AI视觉OCR提示词"
        />
        <SavedPromptsPicker prompt-type="ai_vision_ocr" @select="handleAiVisionPromptSelect" />
        <ProductActionRow aria-label="AI 视觉 OCR 提示词格式" justify="start">
          <UiSelect
            :model-value="currentPromptMode"
            :options="promptModeOptions"
            @change="(v: UiSelectValue) => handlePromptModeChange(String(v))"
          />
          <span class="ocr-settings__prompt-mode-hint">{{ getPromptModeHint() }}</span>
        </ProductActionRow>
        <UiField
          v-if="currentPromptMode === 'paddleocr_vl'"
          class="ocr-settings__prompt-language-field"
          variant="settings"
          label="源语言"
          control-id="settingsAiVisionPaddleOcrVlSourceLanguage"
        >
          <UiCombobox
            class="ocr-settings__prompt-language-combobox"
            input-id="settingsAiVisionPaddleOcrVlSourceLanguage"
            aria-label="AI 视觉 OCR 专用模型源语言"
            :model-value="paddleOcrVlSourceLang"
            :groups="paddleOcrVlSourceLanguageGroups"
            @change="(v: UiSelectValue) => handlePaddleOcrVlLangChange(String(v))"
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
          v-model="localAiVisionOcr.rpmLimit"
          :min="0"
          :step="1"
        />
      </UiField>
      <UiField variant="settings" label="业务重试" control-id="settingsAiVisionBusinessRetries">
        <UiNumberField
          input-id="settingsAiVisionBusinessRetries"
          v-model="localAiVisionOcr.businessRetries"
          :min="0"
          :max="10"
          :step="1"
        />
      </UiField>
      <UiField variant="settings" label="传输重试" control-id="settingsAiVisionTransportRetries">
        <UiNumberField
          input-id="settingsAiVisionTransportRetries"
          v-model="localAiVisionOcr.transportRetries"
          :min="0"
          :max="10"
          :step="1"
        />
      </UiField>
      <UiField
        variant="settings"
        control="checkbox"
        label="流式调用"
        control-id="settingsAiVisionUseStream"
        hint="使用流式请求并在终端输出流式日志"
      >
        <UiCheckbox input-id="settingsAiVisionUseStream" v-model="localAiVisionOcr.useStream" />
      </UiField>
      <UiField variant="settings">
        <OpenAIExtraBodyEditor v-model="localAiVisionOcr.extraBody" />
      </UiField>
      <UiField
        variant="settings"
        label="最小图片尺寸 (像素)"
        control-id="settingsMinImageSize"
        hint="VLM模型通常要求图片尺寸 ≥28px，设为0则不自动放大小图"
      >
        <UiNumberField
          input-id="settingsMinImageSize"
          v-model="localAiVisionOcr.minImageSize"
          :min="0"
          :step="1"
        />
      </UiField>
      <ProductActionRow aria-label="AI 视觉 OCR 操作" justify="start">
        <UiButton variant="secondary" @click="testAiVisionOcr" :disabled="isTesting">
          <span v-if="isTesting">测试中...</span>
          <template v-else>
            <UiIcon name="link" />
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
import UiIcon from '@/components/ui/UiIcon.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiPasswordField from '@/components/ui/UiPasswordField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import AiProviderCredentialFields from '@/components/settings/AiProviderCredentialFields.vue'
import AiProviderSelectField from '@/components/settings/AiProviderSelectField.vue'
import type { UiSelectValue } from '@/components/ui/selectTypes'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import { ref, computed, watch } from 'vue'
import {
  normalizeProviderId,
  providerRequiresApiKey,
  providerRequiresBaseUrl,
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settings'
import {
  fetchModels as fetchV2Models,
  testAiVisionOcrConnection,
  testBaiduOcrConnection,
} from '@/api/v2/diagnostics'
import { useToast } from '@/utils/toast'
import {
  DEFAULT_AI_VISION_OCR_PROMPT,
  DEFAULT_AI_VISION_OCR_JSON_PROMPT,
  getPaddleOcrVlPrompt,
  inferPaddleOcrVlPromptLanguage,
  PADDLEOCR_VL_LANG_MAP,
} from '@/constants'
import type { OcrEngine } from '@/types/settings'
import UiCombobox from '@/components/ui/UiCombobox.vue'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'
import {
  getHybridCounterpartEngine,
  isSupportedHybridOcrEngine,
  RECOMMENDED_HYBRID_SECONDARY_ENGINE,
  SUPPORTED_HYBRID_OCR_ENGINES,
} from '@/utils/hybridOcr'
import {
  aiVisionProviderOptions,
  allOcrEngineOptions,
  baiduSourceLanguageOptions,
  baiduVersionOptions,
  paddleOcrVlSourceLanguageGroups,
  promptModeOptions,
  sourceLanguageGroups,
} from './ocrSettingsOptions'
import {
  useAiModelDiscovery,
  type AiModelDiscoveryMessageTone,
} from '@/composables/useAiModelDiscovery'
const settingsStore = useSettingsStore()
const toast = useToast()
const localBaiduOcr = ref({
  apiKey: settingsStore.settings.baiduOcr.apiKey,
  secretKey: settingsStore.settings.baiduOcr.secretKey,
  version: settingsStore.settings.baiduOcr.version,
  sourceLanguage: settingsStore.settings.baiduOcr.sourceLanguage,
})
const localAiVisionOcr = ref({
  apiKey: settingsStore.settings.aiVisionOcr.apiKey,
  modelName: settingsStore.settings.aiVisionOcr.modelName,
  customBaseUrl: settingsStore.settings.aiVisionOcr.customBaseUrl,
  prompt: settingsStore.settings.aiVisionOcr.prompt,
  promptMode: settingsStore.settings.aiVisionOcr.promptMode,
  rpmLimit: settingsStore.settings.aiVisionOcr.openaiOptions.execution.rpmLimit,
  transportRetries: settingsStore.settings.aiVisionOcr.openaiOptions.execution.transportRetries,
  businessRetries: settingsStore.settings.aiVisionOcr.openaiOptions.execution.businessRetries,
  extraBody: settingsStore.settings.aiVisionOcr.openaiOptions.request.extraBody,
  useStream: settingsStore.settings.aiVisionOcr.openaiOptions.execution.useStream,
  minImageSize: settingsStore.settings.aiVisionOcr.minImageSize,
})
const settings = computed(() => settingsStore.settings)
const hasStoredBaiduCredential = computed(() => settingsStore.hasCredential('ocr', 'baidu'))
const baiduStoredCredentialHint = computed(() =>
  hasStoredBaiduCredential.value && !localBaiduOcr.value.apiKey && !localBaiduOcr.value.secretKey
    ? '百度 OCR 凭据已安全保存在后端；留空表示保持不变，更换时必须同时填写两项'
    : ''
)
const baiduStoredCredentialPlaceholder = computed(() =>
  baiduStoredCredentialHint.value ? '已保存在后端，留空保持不变' : ''
)
watch(
  () => localBaiduOcr.value.apiKey,
  val => {
    settingsStore.updateBaiduOcr({ apiKey: val })
  }
)
watch(
  () => localBaiduOcr.value.secretKey,
  val => {
    settingsStore.updateBaiduOcr({ secretKey: val })
  }
)
watch(
  () => localBaiduOcr.value.version,
  val => {
    settingsStore.updateBaiduOcr({ version: val })
  }
)
watch(
  () => localBaiduOcr.value.sourceLanguage,
  val => {
    settingsStore.updateBaiduOcr({ sourceLanguage: val })
  }
)
watch(
  () => localAiVisionOcr.value.apiKey,
  val => {
    settingsStore.updateAiVisionOcr({ apiKey: val })
  }
)
watch(
  () => localAiVisionOcr.value.modelName,
  val => {
    settingsStore.updateAiVisionOcr({ modelName: val })
  }
)
watch(
  () => localAiVisionOcr.value.customBaseUrl,
  val => {
    settingsStore.updateAiVisionOcr({ customBaseUrl: val })
  }
)
watch(
  () => localAiVisionOcr.value.prompt,
  val => {
    settingsStore.updateAiVisionOcr({ prompt: val })
  }
)
watch(
  () => localAiVisionOcr.value.rpmLimit,
  val => {
    settingsStore.updateAiVisionOcr({ rpmLimit: val })
  }
)
watch(
  () => localAiVisionOcr.value.transportRetries,
  val => {
    settingsStore.updateAiVisionOcr({ transportRetries: val })
  }
)
watch(
  () => localAiVisionOcr.value.businessRetries,
  val => {
    settingsStore.updateAiVisionOcr({ businessRetries: val })
  }
)
watch(
  () => localAiVisionOcr.value.extraBody,
  val => {
    settingsStore.updateAiVisionOcr({ extraBody: val })
  }
)
watch(
  () => localAiVisionOcr.value.useStream,
  val => {
    settingsStore.updateAiVisionOcr({ useStream: val })
  }
)
watch(
  () => localAiVisionOcr.value.minImageSize,
  val => {
    settingsStore.updateAiVisionOcr({ minImageSize: val })
  }
)
const isTesting = ref(false)
function notifyModelDiscovery(message: string, tone: AiModelDiscoveryMessageTone): void {
  toast[tone](message)
}
const aiVisionModelDiscovery = useAiModelDiscovery({
  source: () => ({
    provider: settingsStore.settings.aiVisionOcr.provider,
    apiKey: localAiVisionOcr.value.apiKey,
    baseUrl: localAiVisionOcr.value.customBaseUrl,
    hasStoredCredential: settingsStore.hasCredential(
      'ai_vision_ocr',
      settingsStore.settings.aiVisionOcr.provider
    ),
  }),
  fetcher: (provider, apiKey, baseUrl) => fetchV2Models(provider, apiKey, baseUrl, 'ai_vision_ocr'),
  notify: notifyModelDiscovery,
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
const ocrEngineOptions = computed(() => {
  if (!settings.value.hybridOcr.enabled) {
    return allOcrEngineOptions
  }
  const supported = new Set<string>(SUPPORTED_HYBRID_OCR_ENGINES)
  return allOcrEngineOptions.filter(option => supported.has(option.value))
})
const hybridSecondaryEngineOptions = computed(() =>
  allOcrEngineOptions
    .filter(
      option =>
        isSupportedHybridOcrEngine(option.value) && option.value !== settings.value.ocrEngine
    )
    .map(option => ({ ...option }))
)
function toSelectString(value: UiSelectValue): string {
  return String(value)
}
function isOcrEngine(value: string): value is OcrEngine {
  return allOcrEngineOptions.some(option => option.value === value)
}
function handleOcrEngineChange(value: UiSelectValue) {
  const nextEngine = toSelectString(value)
  if (isOcrEngine(nextEngine)) {
    settingsStore.setOcrEngine(nextEngine)
  }
}
function handleSourceLanguageSelect(value: UiSelectValue) {
  settingsStore.setSourceLanguage(toSelectString(value))
}
function handleHybridOcrEnabledChange(value: boolean) {
  settingsStore.updateHybridOcr({ enabled: value })
}
function handleHybridSecondaryEngineChange(value: UiSelectValue) {
  const secondaryEngine = toSelectString(value)
  if (isSupportedHybridOcrEngine(secondaryEngine)) {
    settingsStore.updateHybridOcr({ secondaryEngine })
    return
  }
  const fallback = isSupportedHybridOcrEngine(settings.value.ocrEngine)
    ? getHybridCounterpartEngine(settings.value.ocrEngine)
    : RECOMMENDED_HYBRID_SECONDARY_ENGINE
  settingsStore.updateHybridOcr({ secondaryEngine: fallback })
}
function handleHybridThresholdChange(value: number | null) {
  if (value === null) return
  const normalized = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0
  settingsStore.updateHybridOcr({ confidenceThreshold: normalized })
}
function handlePaddleOcrVlSourceLanguageChange(value: UiSelectValue) {
  settingsStore.updatePaddleOcrVl({ sourceLanguage: toSelectString(value) })
}
function getSourceLanguageHint(): string {
  const engine = settingsStore.settings.ocrEngine
  switch (engine) {
    case 'manga_ocr':
      return 'MangaOCR 专为日语漫画优化，源语言设置不影响识别'
    case 'paddle_ocr':
      return 'PaddleOCR 会根据源语言加载对应的识别模型'
    case 'paddleocr_vl':
      return 'PaddleOCR-VL 基于 VLM 微调，专为日语漫画优化，准确率高达 70%'
    case 'baidu_ocr':
      return '百度OCR 使用独立的源语言设置（见下方）'
    case 'ai_vision':
      return 'AI视觉OCR 通过提示词指定识别语言'
    case '48px_ocr':
      return '48px OCR 支持日中英韩等多语言，源语言设置不影响识别'
    default:
      return '选择要识别的原文语言'
  }
}
function handleAiVisionProviderChange(providerValue: UiSelectValue) {
  aiVisionModelDiscovery.invalidate()
  const newProvider = normalizeProviderId(toSelectString(providerValue))
  settingsStore.setAiVisionOcrProvider(newProvider)
  syncLocalAiVisionOcr()
}
function syncLocalAiVisionOcr() {
  localAiVisionOcr.value.apiKey = settingsStore.settings.aiVisionOcr.apiKey
  localAiVisionOcr.value.modelName = settingsStore.settings.aiVisionOcr.modelName
  localAiVisionOcr.value.customBaseUrl = settingsStore.settings.aiVisionOcr.customBaseUrl
  localAiVisionOcr.value.prompt = settingsStore.settings.aiVisionOcr.prompt
  localAiVisionOcr.value.promptMode = settingsStore.settings.aiVisionOcr.promptMode
  localAiVisionOcr.value.rpmLimit =
    settingsStore.settings.aiVisionOcr.openaiOptions.execution.rpmLimit
  localAiVisionOcr.value.transportRetries =
    settingsStore.settings.aiVisionOcr.openaiOptions.execution.transportRetries
  localAiVisionOcr.value.businessRetries =
    settingsStore.settings.aiVisionOcr.openaiOptions.execution.businessRetries
  localAiVisionOcr.value.extraBody =
    settingsStore.settings.aiVisionOcr.openaiOptions.request.extraBody
  localAiVisionOcr.value.useStream =
    settingsStore.settings.aiVisionOcr.openaiOptions.execution.useStream
  localAiVisionOcr.value.minImageSize = settingsStore.settings.aiVisionOcr.minImageSize
  paddleOcrVlSourceLang.value = inferPaddleOcrVlPromptLanguage(
    settingsStore.settings.aiVisionOcr.prompt,
    paddleOcrVlSourceLang.value
  )
}
const currentPromptMode = computed(() => {
  return settingsStore.settings.aiVisionOcr.promptMode || 'normal'
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
function handlePromptModeChange(mode: string) {
  let newPrompt: string
  switch (mode) {
    case 'json':
      newPrompt = DEFAULT_AI_VISION_OCR_JSON_PROMPT
      break
    case 'paddleocr_vl': {
      const langName = PADDLEOCR_VL_LANG_MAP[paddleOcrVlSourceLang.value] || '日语'
      newPrompt = getPaddleOcrVlPrompt(langName)
      break
    }
    default:
      newPrompt = DEFAULT_AI_VISION_OCR_PROMPT
      break
  }
  settingsStore.updateAiVisionOcr({
    prompt: newPrompt,
    promptMode: mode as 'normal' | 'json' | 'paddleocr_vl',
    forceJsonOutput: mode === 'json',
  })
  localAiVisionOcr.value.prompt = newPrompt
  localAiVisionOcr.value.promptMode = mode as 'normal' | 'json' | 'paddleocr_vl'
}
const paddleOcrVlSourceLang = ref(inferPaddleOcrVlPromptLanguage(localAiVisionOcr.value.prompt))
function handlePaddleOcrVlLangChange(langCode: string) {
  paddleOcrVlSourceLang.value = langCode
  const langName = PADDLEOCR_VL_LANG_MAP[langCode] || '日语'
  const newPrompt = getPaddleOcrVlPrompt(langName)
  settingsStore.updateAiVisionOcr({
    prompt: newPrompt,
    promptMode: 'paddleocr_vl',
    forceJsonOutput: false,
  })
  localAiVisionOcr.value.prompt = newPrompt
  localAiVisionOcr.value.promptMode = 'paddleocr_vl'
}
async function testBaiduOcr() {
  const apiKey = localBaiduOcr.value.apiKey?.trim()
  const secretKey = localBaiduOcr.value.secretKey?.trim()
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
  isTesting.value = true
  try {
    const result = await testAiVisionOcrConnection({
      provider: settingsStore.settings.aiVisionOcr.provider,
      apiKey: localAiVisionOcr.value.apiKey,
      modelName: localAiVisionOcr.value.modelName,
      customBaseUrl: localAiVisionOcr.value.customBaseUrl,
      prompt: localAiVisionOcr.value.prompt,
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
  const inferredMode: 'normal' | 'json' | 'paddleocr_vl' = content.includes('"extracted_text"')
    ? 'json'
    : content.startsWith('对图中的') && content.endsWith('进行OCR:')
      ? 'paddleocr_vl'
      : 'normal'
  settingsStore.updateAiVisionOcr({
    prompt: content,
    promptMode: inferredMode,
    forceJsonOutput: inferredMode === 'json',
  })
  localAiVisionOcr.value.prompt = content
  localAiVisionOcr.value.promptMode = inferredMode
  if (inferredMode === 'paddleocr_vl') {
    paddleOcrVlSourceLang.value = inferPaddleOcrVlPromptLanguage(
      content,
      paddleOcrVlSourceLang.value
    )
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
  border-radius: 6px;
  border: 1px solid var(--color-border-muted);
}

.ocr-settings__prompt-language-combobox {
  min-width: 150px;
}
</style>
