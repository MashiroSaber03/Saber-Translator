<template>
  <div class="ocr-settings">
    <UiPanel variant="settings">
      <template #title>OCR引擎选择</template>
      <UiField class="ui-settings-field">
        <label for="settingsOcrEngine">OCR引擎:</label>
        <CustomSelect
          :model-value="settings.ocrEngine"
          :options="ocrEngineOptions"
          @change="(v: any) => handleOcrEngineChange(String(v))"
        />
      </UiField>
      <UiField v-show="settings.ocrEngine === 'paddle_ocr'" class="ui-settings-field">
        <label for="settingsSourceLanguage">源语言:</label>
        <CustomSelect
          :model-value="settings.sourceLanguage"
          :groups="sourceLanguageGroups"
          @change="(v: any) => { settings.sourceLanguage = v; handleSourceLanguageChange() }"
        />
        <div class="ui-form-hint">
          {{ getSourceLanguageHint() }}
        </div>
      </UiField>
    </UiPanel>
    <UiPanel variant="settings">
      <template #title>混合OCR设置</template>
      <UiField class="ui-settings-field ui-settings-field--checkbox">
        <label for="settingsHybridOcrEnabled">启用混合OCR:</label>
        <UiInput
          id="settingsHybridOcrEnabled"
          type="checkbox"
          :checked="settings.hybridOcr.enabled"
          @change="handleHybridOcrEnabledEvent"
        />
      </UiField>
      <div v-show="settings.hybridOcr.enabled" class="ui-settings-row">
        <UiField class="ui-settings-field">
          <label for="settingsHybridSecondaryOcr">备用OCR:</label>
          <CustomSelect
            :model-value="settings.hybridOcr.secondaryEngine"
            :options="hybridSecondaryEngineOptions"
            @change="(v: any) => handleHybridSecondaryEngineChange(v)"
          />
        </UiField>
      </div>
      <div v-show="settings.hybridOcr.enabled" class="ui-settings-row">
        <UiField class="ui-settings-field">
          <label for="settingsHybridThreshold">混合阈值:</label>
          <UiInput
            id="settingsHybridThreshold"
            type="number"
            min="0"
            max="1"
            step="0.01"
            :value="settings.hybridOcr.confidenceThreshold"
            @change="handleHybridThresholdInput($event)"
          />
        </UiField>
      </div>
      <div v-show="settings.hybridOcr.enabled" class="ui-form-hint">
        首批混合OCR仅支持 MangaOCR / 48px OCR，推荐顺序为 48px OCR → MangaOCR。启用后会优先走 textline 级专用链路。
      </div>
    </UiPanel>
    <UiPanel variant="settings" v-show="settings.ocrEngine === 'paddleocr_vl'">
      <template #title>PaddleOCR-VL 设置</template>
      <UiField class="ui-settings-field">
        <label for="settingsPaddleOcrVlSourceLanguage">源语言:</label>
        <CustomSelect
          :model-value="settings.paddleOcrVl.sourceLanguage"
          :groups="paddleOcrVlSourceLanguageGroups"
          @change="(v: any) => handlePaddleOcrVlSourceLanguageChange(v)"
        />
        <div class="ui-form-hint">
          选择图像中的源语言，用于优化 OCR 识别效果
        </div>
      </UiField>
    </UiPanel>
    <UiPanel variant="settings" v-show="settings.ocrEngine === 'baidu_ocr'">
      <template #title>百度OCR 设置</template>
      <div class="ui-settings-row">
        <UiField class="ui-settings-field">
          <label for="settingsBaiduApiKey">API Key:</label>
          <div class="password-input-wrapper">
            <UiInput
              :type="showBaiduApiKey ? 'text' : 'password'"
              id="settingsBaiduApiKey"
              v-model="localBaiduOcr.apiKey"
              class="secure-input"
              placeholder="请输入百度OCR API Key"
              autocomplete="off"
            />
            <UiButton variant="toolbar" type="button" class="password-toggle-btn" tabindex="-1" @click="showBaiduApiKey = !showBaiduApiKey">
              <span class="eye-icon" v-if="!showBaiduApiKey">👁</span>
              <span class="eye-off-icon" v-else>👁‍🗨</span>
            </UiButton>
          </div>
        </UiField>
        <UiField class="ui-settings-field">
          <label for="settingsBaiduSecretKey">Secret Key:</label>
          <div class="password-input-wrapper">
            <UiInput
              :type="showBaiduSecretKey ? 'text' : 'password'"
              id="settingsBaiduSecretKey"
              v-model="localBaiduOcr.secretKey"
              class="secure-input"
              placeholder="请输入Secret Key"
              autocomplete="off"
            />
            <UiButton variant="toolbar" type="button" class="password-toggle-btn" tabindex="-1" @click="showBaiduSecretKey = !showBaiduSecretKey">
              <span class="eye-icon" v-if="!showBaiduSecretKey">👁</span>
              <span class="eye-off-icon" v-else>👁‍🗨</span>
            </UiButton>
          </div>
        </UiField>
      </div>
      <div class="ui-settings-row">
        <UiField class="ui-settings-field">
          <label for="settingsBaiduVersion">识别版本:</label>
          <CustomSelect
            v-model="localBaiduOcr.version"
            :options="baiduVersionOptions"
          />
        </UiField>
        <UiField class="ui-settings-field">
          <label for="settingsBaiduSourceLanguage">源语言:</label>
          <CustomSelect
            v-model="localBaiduOcr.sourceLanguage"
            :options="baiduSourceLanguageOptions"
          />
        </UiField>
      </div>
      <UiButton variant="toolbar" class="settings-test-btn" @click="testBaiduOcr" :disabled="isTesting">
        {{ isTesting ? '测试中...' : '🔗 测试连接' }}
      </UiButton>
    </UiPanel>
    <UiPanel variant="settings" v-show="settings.ocrEngine === 'ai_vision'">
      <template #title>AI视觉OCR 设置</template>
      <div class="ui-settings-row">
        <UiField class="ui-settings-field">
          <label for="settingsAiVisionProvider">服务商:</label>
          <CustomSelect
            :model-value="settings.aiVisionOcr.provider"
            :options="aiVisionProviderOptions"
            @change="(v: any) => handleAiVisionProviderChange(v)"
          />
        </UiField>
        <UiField v-show="providerRequiresApiKey(settings.aiVisionOcr.provider)" class="ui-settings-field">
          <label for="settingsAiVisionApiKey">API Key:</label>
          <div class="password-input-wrapper">
            <UiInput
              :type="showAiVisionApiKey ? 'text' : 'password'"
              id="settingsAiVisionApiKey"
              v-model="localAiVisionOcr.apiKey"
              class="secure-input"
              placeholder="请输入API Key"
              autocomplete="off"
            />
            <UiButton variant="toolbar" type="button" class="password-toggle-btn" tabindex="-1" @click="showAiVisionApiKey = !showAiVisionApiKey">
              <span class="eye-icon" v-if="!showAiVisionApiKey">👁</span>
              <span class="eye-off-icon" v-else>👁‍🗨</span>
            </UiButton>
          </div>
        </UiField>
      </div>
      <UiField v-show="providerRequiresBaseUrl(settings.aiVisionOcr.provider)" class="ui-settings-field">
        <label for="settingsCustomAiVisionBaseUrl">Base URL:</label>
        <UiInput
          type="text"
          id="settingsCustomAiVisionBaseUrl"
          v-model="localAiVisionOcr.customBaseUrl"
          placeholder="例如: https://api.example.com/v1"
        />
      </UiField>
      <UiField class="ui-settings-field">
        <label for="settingsAiVisionModelName">模型名称:</label>
        <div class="model-input-with-fetch">
          <UiInput
            type="text"
            id="settingsAiVisionModelName"
            v-model="localAiVisionOcr.modelName"
            placeholder="如: silicon-llava2-34b"
          />
          <UiButton
            variant="toolbar"
            type="button"
            class="fetch-models-btn"
            title="获取可用模型列表"
            @click="fetchAiVisionModels"
            :disabled="isFetchingModels"
          >
            <span class="fetch-icon">🔍</span>
            <span class="fetch-text">{{ isFetchingModels ? '获取中...' : '获取模型' }}</span>
          </UiButton>
        </div>
        <div v-if="aiVisionModels.length > 0" class="model-select-container">
          <CustomSelect
            v-model="localAiVisionOcr.modelName"
            :options="aiVisionModelOptions"
          />
          <span class="model-count">共 {{ aiVisionModels.length }} 个模型</span>
        </div>
      </UiField>
      <UiField class="ui-settings-field">
        <label for="settingsAiVisionOcrPrompt">OCR提示词:</label>
        <UiTextarea
          id="settingsAiVisionOcrPrompt"
          v-model="localAiVisionOcr.prompt"
          rows="3"
          placeholder="AI视觉OCR提示词"
        />
        <SavedPromptsPicker
          prompt-type="ai_vision_ocr"
          @select="handleAiVisionPromptSelect"
        />
        <div class="prompt-format-selector">
          <CustomSelect
            :model-value="currentPromptMode"
            :options="promptModeOptions"
            @change="(v: string | number) => handlePromptModeChange(String(v))"
          />
          <span class="ui-form-hint">{{ getPromptModeHint() }}</span>
        </div>
        <div v-if="currentPromptMode === 'paddleocr_vl'" class="paddleocr-vl-lang-selector">
          <label>源语言:</label>
          <CustomSelect
            :model-value="paddleOcrVlSourceLang"
            :groups="paddleOcrVlSourceLanguageGroups"
            @change="(v: string | number) => handlePaddleOcrVlLangChange(String(v))"
          />
        </div>
      </UiField>
      <UiField class="ui-settings-field">
        <label for="settingsRpmAiVisionOcr">RPM限制 (每分钟请求数):</label>
        <UiInput type="number" id="settingsRpmAiVisionOcr" v-model.number="localAiVisionOcr.rpmLimit" min="0" step="1" />
        <div class="ui-form-hint">0 表示无限制</div>
      </UiField>
      <UiField class="ui-settings-field">
        <label for="settingsAiVisionBusinessRetries">业务重试:</label>
        <UiInput type="number" id="settingsAiVisionBusinessRetries" v-model.number="localAiVisionOcr.businessRetries" min="0" max="10" step="1" />
      </UiField>
      <UiField class="ui-settings-field">
        <label for="settingsAiVisionTransportRetries">传输重试:</label>
        <UiInput type="number" id="settingsAiVisionTransportRetries" v-model.number="localAiVisionOcr.transportRetries" min="0" max="10" step="1" />
      </UiField>
      <UiField class="ui-settings-field">
        <label class="ui-checkbox-label">
          <UiInput type="checkbox" v-model="localAiVisionOcr.useStream" />
          流式调用
        </label>
        <div class="ui-form-hint">使用流式请求并在终端输出流式日志</div>
      </UiField>
      <UiField class="ui-settings-field">
        <OpenAIExtraBodyEditor v-model="localAiVisionOcr.extraBody" />
      </UiField>
      <UiField class="ui-settings-field">
        <label for="settingsMinImageSize">最小图片尺寸 (像素):</label>
        <UiInput type="number" id="settingsMinImageSize" v-model.number="localAiVisionOcr.minImageSize" min="0" step="1" />
        <div class="ui-form-hint">VLM模型通常要求图片尺寸 ≥28px，设为0则不自动放大小图</div>
      </UiField>
      <UiButton variant="toolbar" class="settings-test-btn" @click="testAiVisionOcr" :disabled="isTesting">
        {{ isTesting ? '测试中...' : '🔗 测试连接' }}
      </UiButton>
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
  normalizeProviderId,
  providerRequiresApiKey,
  providerRequiresBaseUrl,
  providerSupportsCapability
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settingsStore'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
import {
  DEFAULT_AI_VISION_OCR_PROMPT,
  DEFAULT_AI_VISION_OCR_JSON_PROMPT,
  getPaddleOcrVlPrompt,
  PADDLEOCR_VL_LANG_MAP
} from '@/constants'
import type { OcrEngine } from '@/types/settings'
import CustomSelect from '@/components/common/CustomSelect.vue'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'
import {
  isSupportedHybridOcrEngine,
  SUPPORTED_HYBRID_OCR_ENGINES
} from '@/utils/hybridOcr'
import {
  aiVisionProviderOptions,
  allOcrEngineOptions,
  baiduSourceLanguageOptions,
  baiduVersionOptions,
  paddleOcrVlSourceLanguageGroups,
  promptModeOptions,
  sourceLanguageGroups
} from './ocrSettingsOptions'
const settingsStore = useSettingsStore()
const toast = useToast()
// 本地设置状态（用于双向绑定，修改后自动同步到 store）
const localBaiduOcr = ref({
  apiKey: settingsStore.settings.baiduOcr.apiKey,
  secretKey: settingsStore.settings.baiduOcr.secretKey,
  version: settingsStore.settings.baiduOcr.version,
  sourceLanguage: settingsStore.settings.baiduOcr.sourceLanguage
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
  minImageSize: settingsStore.settings.aiVisionOcr.minImageSize
})
// 直接访问 store 的只读设置（用于显示条件判断）
const settings = computed(() => settingsStore.settings)
// Watch 同步：本地状态变化时自动保存到 store
watch(() => localBaiduOcr.value.apiKey, (val) => {
  settingsStore.updateBaiduOcr({ apiKey: val })
})
watch(() => localBaiduOcr.value.secretKey, (val) => {
  settingsStore.updateBaiduOcr({ secretKey: val })
})
watch(() => localBaiduOcr.value.version, (val) => {
  settingsStore.updateBaiduOcr({ version: val })
})
watch(() => localBaiduOcr.value.sourceLanguage, (val) => {
  settingsStore.updateBaiduOcr({ sourceLanguage: val })
})
watch(() => localAiVisionOcr.value.apiKey, (val) => {
  settingsStore.updateAiVisionOcr({ apiKey: val })
})
watch(() => localAiVisionOcr.value.modelName, (val) => {
  settingsStore.updateAiVisionOcr({ modelName: val })
})
watch(() => localAiVisionOcr.value.customBaseUrl, (val) => {
  settingsStore.updateAiVisionOcr({ customBaseUrl: val })
})
watch(() => localAiVisionOcr.value.prompt, (val) => {
  settingsStore.updateAiVisionOcr({ prompt: val })
})
watch(() => localAiVisionOcr.value.rpmLimit, (val) => {
  settingsStore.updateAiVisionOcr({ rpmLimit: val })
})
watch(() => localAiVisionOcr.value.transportRetries, (val) => {
  settingsStore.updateAiVisionOcr({ transportRetries: val })
})
watch(() => localAiVisionOcr.value.businessRetries, (val) => {
  settingsStore.updateAiVisionOcr({ businessRetries: val })
})
watch(() => localAiVisionOcr.value.extraBody, (val) => {
  settingsStore.updateAiVisionOcr({ extraBody: val })
})
watch(() => localAiVisionOcr.value.useStream, (val) => {
  settingsStore.updateAiVisionOcr({ useStream: val })
})
watch(() => localAiVisionOcr.value.minImageSize, (val) => {
  settingsStore.updateAiVisionOcr({ minImageSize: val })
})
const showBaiduApiKey = ref(false)
const showBaiduSecretKey = ref(false)
const showAiVisionApiKey = ref(false)
const isTesting = ref(false)
const isFetchingModels = ref(false)
const aiVisionModels = ref<string[]>([])
const aiVisionModelOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  aiVisionModels.value.forEach(model => {
    options.push({ label: model, value: model })
  })
  return options
})
const ocrEngineOptions = computed(() => {
  if (!settings.value.hybridOcr.enabled) {
    return allOcrEngineOptions
  }
  const supported = new Set<string>(SUPPORTED_HYBRID_OCR_ENGINES)
  return allOcrEngineOptions.filter((option) => supported.has(option.value))
})
const hybridSecondaryEngineOptions = computed(() =>
  allOcrEngineOptions
    .filter(
      (option) =>
        isSupportedHybridOcrEngine(option.value) &&
        option.value !== settings.value.ocrEngine
    )
    .map(option => ({ ...option }))
)
function handleOcrEngineChange(value: string) {
  settingsStore.setOcrEngine(value as OcrEngine)
}
function handleSourceLanguageChange() {
  settingsStore.saveToStorage()
}
function handleHybridOcrEnabledChange(value: boolean) {
  settingsStore.updateHybridOcr({ enabled: value })
}
function handleHybridOcrEnabledEvent(event: Event) {
  const target = event.target as HTMLInputElement | null
  handleHybridOcrEnabledChange(Boolean(target?.checked))
}
function handleHybridSecondaryEngineChange(value: string) {
  settingsStore.updateHybridOcr({ secondaryEngine: value as any })
}
function handleHybridThresholdChange(value: number) {
  const normalized = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0
  settingsStore.updateHybridOcr({ confidenceThreshold: normalized })
}
function handleHybridThresholdInput(event: Event) {
  const target = event.target as HTMLInputElement | null
  handleHybridThresholdChange(Number(target?.value))
}
function handlePaddleOcrVlSourceLanguageChange(value: string) {
  settingsStore.updatePaddleOcrVl({ sourceLanguage: value })
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
// 处理AI视觉服务商切换（当前行为逻辑：独立保存每个服务商的配置）
function handleAiVisionProviderChange(newProvider: string) {
  newProvider = normalizeProviderId(newProvider)
  // 使用 store 的方法切换服务商（会自动保存旧配置、恢复新配置）
  settingsStore.setAiVisionOcrProvider(newProvider)
  aiVisionModels.value = []
  // 同步本地状态（服务商切换后 store 会恢复新服务商的配置）
  syncLocalAiVisionOcr()
}
function syncLocalAiVisionOcr() {
  localAiVisionOcr.value.apiKey = settingsStore.settings.aiVisionOcr.apiKey
  localAiVisionOcr.value.modelName = settingsStore.settings.aiVisionOcr.modelName
  localAiVisionOcr.value.customBaseUrl = settingsStore.settings.aiVisionOcr.customBaseUrl
  localAiVisionOcr.value.prompt = settingsStore.settings.aiVisionOcr.prompt
  localAiVisionOcr.value.promptMode = settingsStore.settings.aiVisionOcr.promptMode
  localAiVisionOcr.value.rpmLimit = settingsStore.settings.aiVisionOcr.openaiOptions.execution.rpmLimit
  localAiVisionOcr.value.transportRetries = settingsStore.settings.aiVisionOcr.openaiOptions.execution.transportRetries
  localAiVisionOcr.value.businessRetries = settingsStore.settings.aiVisionOcr.openaiOptions.execution.businessRetries
  localAiVisionOcr.value.extraBody = settingsStore.settings.aiVisionOcr.openaiOptions.request.extraBody
  localAiVisionOcr.value.useStream = settingsStore.settings.aiVisionOcr.openaiOptions.execution.useStream
  localAiVisionOcr.value.minImageSize = settingsStore.settings.aiVisionOcr.minImageSize
}
// 当前提示词模式（计算属性）
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
    default: // 'normal'
      newPrompt = DEFAULT_AI_VISION_OCR_PROMPT
      break
  }
  settingsStore.updateAiVisionOcr({ 
    prompt: newPrompt,
    promptMode: mode as 'normal' | 'json' | 'paddleocr_vl',
    isJsonMode: mode === 'json'
  })
  localAiVisionOcr.value.prompt = newPrompt
  localAiVisionOcr.value.promptMode = mode as 'normal' | 'json' | 'paddleocr_vl'
}
const paddleOcrVlSourceLang = ref('japanese')
function handlePaddleOcrVlLangChange(langCode: string) {
  paddleOcrVlSourceLang.value = langCode
  const langName = PADDLEOCR_VL_LANG_MAP[langCode] || '日语'
  const newPrompt = getPaddleOcrVlPrompt(langName)
  settingsStore.updateAiVisionOcr({
    prompt: newPrompt,
    promptMode: 'paddleocr_vl',
    isJsonMode: false
  })
  localAiVisionOcr.value.prompt = newPrompt
  localAiVisionOcr.value.promptMode = 'paddleocr_vl'
}
// 测试百度OCR连接（当前行为逻辑）
async function testBaiduOcr() {
  const apiKey = localBaiduOcr.value.apiKey?.trim()
  const secretKey = localBaiduOcr.value.secretKey?.trim()
  if (!apiKey || !secretKey) {
    toast.warning('请填写百度OCR的API Key和Secret Key')
    return
  }
  isTesting.value = true
  toast.info('正在测试百度OCR连接...')
  try {
    const result = await configApi.testBaiduOcrConnection(apiKey, secretKey)
    if (result.success) {
      toast.success(result.message || '百度OCR连接成功!')
    } else {
      toast.error(result.message || result.error || '百度OCR连接失败')
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
    const result = await configApi.testAiVisionOcrConnection({
      provider: settingsStore.settings.aiVisionOcr.provider,
      apiKey: localAiVisionOcr.value.apiKey,
      modelName: localAiVisionOcr.value.modelName,
      customBaseUrl: localAiVisionOcr.value.customBaseUrl,
      prompt: localAiVisionOcr.value.prompt
    })
    if (result.success) {
      toast.success('AI视觉OCR连接成功')
    } else {
      toast.error(`AI视觉OCR连接失败: ${result.error || '未知错误'}`)
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '连接测试失败'
    toast.error(errorMessage)
  } finally {
    isTesting.value = false
  }
}
// 获取AI视觉模型列表（当前行为 doFetchModels 逻辑）
async function fetchAiVisionModels() {
  const provider = settingsStore.settings.aiVisionOcr.provider
  const apiKey = localAiVisionOcr.value.apiKey?.trim()
  const baseUrl = localAiVisionOcr.value.customBaseUrl?.trim()
  // 验证（与当前行为一致）
  if (providerRequiresApiKey(provider) && !apiKey) {
    toast.warning('请先填写 API Key')
    return
  }
  if (!providerSupportsCapability(provider, 'modelFetch')) {
    toast.warning('当前服务商不支持自动获取模型列表')
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
      // 后端返回的是 {id, name} 对象数组，提取 id 作为模型列表
      aiVisionModels.value = result.models.map(m => m.id)
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
function handleAiVisionPromptSelect(content: string, name: string) {
  const inferredMode: 'normal' | 'json' | 'paddleocr_vl' =
    content.includes('"extracted_text"')
      ? 'json'
      : content.startsWith('对图中的') && content.endsWith('进行OCR:')
        ? 'paddleocr_vl'
        : 'normal'
  settingsStore.updateAiVisionOcr({
    prompt: content,
    promptMode: inferredMode,
    isJsonMode: inferredMode === 'json'
  })
  localAiVisionOcr.value.prompt = content
  localAiVisionOcr.value.promptMode = inferredMode
  toast.success(`已应用提示词: ${name}`)
}
</script>

<style scoped>.settings-test-btn {
  width: 100%;
  padding: 10px 16px;
  background-color: var(--color-surface-subtle);
  border: 1px solid var(--color-border-muted);
  border-radius: 6px;
  color: var(--color-text-default);
  font-weight: 500;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.settings-test-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.settings-test-btn:hover:not(:disabled) {
  background-color: var(--color-surface-hover);
  border-color: var(--color-action-primary);
  color: var(--color-action-primary);
}

.model-input-with-fetch {
  display: flex;
  gap: 10px;
  align-items: center;
}

.fetch-models-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background-color: var(--color-surface-subtle);
  border: 1px solid var(--color-border-muted);
  border-radius: 6px;
  color: var(--color-text-default);
  font-size: 13px;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.2s ease;
  height: 38px;
}

.fetch-models-btn:hover:not(:disabled) {
  background-color: var(--color-action-primary);
  color: var(--color-text-inverse);
  border-color: var(--color-action-primary);
}
/* PaddleOCR-VL 语言选择器 */
.paddleocr-vl-lang-selector {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
  padding: 10px 12px;
  background: var(--color-surface-subtle);
  border-radius: 6px;
  border: 1px solid var(--color-border-muted);
}

.paddleocr-vl-lang-selector label {
  font-size: 13px;
  color: var(--color-text-supporting);
  white-space: nowrap;
}

.paddleocr-vl-lang-selector .custom-select {
  flex: 1;
  min-width: 150px;
}
</style>
