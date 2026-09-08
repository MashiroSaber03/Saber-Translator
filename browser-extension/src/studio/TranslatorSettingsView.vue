<script setup lang="ts">
import { computed, ref } from 'vue'
import SelectControl from './SelectControl.vue'
import NumberControl from './NumberControl.vue'
import { getProviderManifest } from '../../../vue-frontend/src/config/aiProviders'
import ServiceFields, { type ServiceConfig } from './ServiceFields.vue'
import { serviceDomains, useTranslatorSettings, type ServiceKey } from './useTranslatorSettings'
import type { PluginSettingsApi } from '../../../vue-frontend/src/types/browserExtensionSettings'
import {
  allOcrEngineOptions,
  baiduVersionOptions,
  baiduSourceLanguageOptions,
  paddleOcrVlSourceLanguageGroups,
  promptModeOptions,
} from '../../../vue-frontend/src/config/ocrChoices'
import {
  DEFAULT_AI_VISION_OCR_PROMPT,
  DEFAULT_AI_VISION_OCR_JSON_PROMPT,
  getPaddleOcrVlPrompt,
  DEFAULT_HQ_TRANSLATE_PROMPT,
  DEFAULT_TRANSLATE_PROMPT,
  DEFAULT_TRANSLATE_JSON_PROMPT,
  DEFAULT_SINGLE_BUBBLE_PROMPT,
  DEFAULT_SINGLE_BUBBLE_JSON_PROMPT,
} from '../../../vue-frontend/src/constants/prompts'
const props = defineProps<{ api: PluginSettingsApi; section: string; active: boolean }>()
const {
  settings: s,
  loading,
  saving,
  dirty,
  error,
  secretValue,
  globalApi,
  change,
  selectProvider,
  updateSecret,
  load,
  save,
  form,
} = useTranslatorSettings(
  props.api,
  computed(() => props.active)
)
function bindForm(node: unknown) {
  form.value = node as HTMLFormElement | undefined
}
const reveal = ref(false)
const baiduChecking = ref(false)
const baiduMessage = ref('')
async function testBaidu() {
  baiduChecking.value = true
  const secret = {
    baidu_api_key: secretValue('ocr', 'baidu', 'baidu_api_key'),
    baidu_secret_key: secretValue('ocr', 'baidu', 'baidu_secret_key'),
  }
  try {
    if (Boolean(secret?.baidu_api_key?.trim()) !== Boolean(secret?.baidu_secret_key?.trim()))
      throw new Error('请同时填写百度 API Key 和 Secret Key')
    const result = await globalApi<{ success: boolean; message: string }>(
      '/connection-tests/baidu_ocr',
      'POST',
      {
        domain: 'ocr',
        ...(secret?.baidu_api_key?.trim() && secret?.baidu_secret_key?.trim()
          ? {
              secret: {
                baidu_api_key: secret.baidu_api_key.trim(),
                baidu_secret_key: secret.baidu_secret_key.trim(),
              },
            }
          : {}),
      }
    )
    baiduMessage.value = result.message
  } catch (e) {
    baiduMessage.value = (e as Error).message
  } finally {
    baiduChecking.value = false
  }
}
const languageOptions = paddleOcrVlSourceLanguageGroups.flatMap(g => g.options)
const titles: Record<string, string> = {
  ocr: 'OCR 识别',
  translation: '翻译服务',
  detection: '检测设置',
  hq: '高质量翻译',
}
const serviceKey = computed<ServiceKey>(() =>
  props.section === 'hq' ? 'hqTranslation' : props.section === 'ocr' ? 'aiVisionOcr' : 'translation'
)
const service = computed(() => s.value![serviceKey.value])
const serviceMetadata = computed(() => getProviderManifest(service.value.provider))
const domain = computed(() => serviceDomains[serviceKey.value])
const secretKey = computed(() =>
  serviceKey.value === 'aiVisionOcr' ? 'ai_vision_api_key' : 'api_key'
)
const keyValue = computed(
  () => secretValue(domain.value, service.value.provider, secretKey.value)
)
const promptKey = computed(() => {
  const t = s.value!.translation
  return t.translationMode === 'single'
    ? t.openaiOptions.request.forceJsonOutput
      ? 'singleJsonPrompt'
      : 'singleNormalPrompt'
    : t.openaiOptions.request.forceJsonOutput
      ? 'batchJsonPrompt'
      : 'batchNormalPrompt'
})
function patchService(patch: Partial<ServiceConfig>) {
  Object.assign(service.value, patch)
  if (serviceKey.value === 'translation')
    s.value!.translatePrompt = s.value!.translation[promptKey.value]
  change(serviceKey.value)
}
function provider(value: string) {
  selectProvider(serviceKey.value, value)
  if (serviceKey.value === 'translation')
    s.value!.translatePrompt = s.value!.translation[promptKey.value]
}
function translatePrompt(value: string) {
  s.value!.translation[promptKey.value] = value
  s.value!.translatePrompt = value
  change()
}
function translateMode(value: string | number) {
  s.value!.translation.translationMode = value as 'batch' | 'single'
  s.value!.translatePrompt = s.value!.translation[promptKey.value]
  change('translation')
}
function jsonOutput(value: boolean) {
  const options = service.value.openaiOptions
  patchService({
    openaiOptions: { ...options, request: { ...options.request, forceJsonOutput: value } },
  })
}
function ocrEngine(value: string | number) {
  s.value!.ocrEngine = value as NonNullable<typeof s.value>['ocrEngine']
  if (!['manga_ocr', '48px_ocr'].includes(String(value))) s.value!.hybridOcr.enabled = false
  else if (s.value!.hybridOcr.secondaryEngine === value)
    s.value!.hybridOcr.secondaryEngine = value === 'manga_ocr' ? '48px_ocr' : 'manga_ocr'
  change()
}
function hybrid(enabled: boolean) {
  if (enabled && !['manga_ocr', '48px_ocr'].includes(s.value!.ocrEngine)) ocrEngine('48px_ocr')
  s.value!.hybridOcr.enabled = enabled
  if (s.value!.hybridOcr.secondaryEngine === s.value!.ocrEngine)
    s.value!.hybridOcr.secondaryEngine =
      s.value!.ocrEngine === 'manga_ocr' ? '48px_ocr' : 'manga_ocr'
  change()
}
function ocrPromptMode(value: string | number) {
  const config = s.value!.aiVisionOcr
  config.promptMode = value as typeof config.promptMode
  config.openaiOptions.request.forceJsonOutput = value === 'json'
  config.prompt =
    value === 'json'
      ? DEFAULT_AI_VISION_OCR_JSON_PROMPT
      : value === 'normal'
        ? DEFAULT_AI_VISION_OCR_PROMPT
        : getPaddleOcrVlPrompt(s.value!.paddleOcrVl.sourceLanguage)
  change('aiVisionOcr')
}
function resetPrompt() {
  if (props.section === 'hq') {
    s.value!.hqTranslation.prompt = DEFAULT_HQ_TRANSLATE_PROMPT
    change('hqTranslation')
  } else {
    const defaults = {
      batchNormalPrompt: DEFAULT_TRANSLATE_PROMPT,
      batchJsonPrompt: DEFAULT_TRANSLATE_JSON_PROMPT,
      singleNormalPrompt: DEFAULT_SINGLE_BUBBLE_PROMPT,
      singleJsonPrompt: DEFAULT_SINGLE_BUBBLE_JSON_PROMPT,
    }
    translatePrompt(defaults[promptKey.value])
  }
}
const detectorOptions = [
  { value: 'ctd', label: 'CTD (Comic Text Detector)' },
  { value: 'yolo', label: 'YOLO' },
  { value: 'default', label: 'Default (DBNet)' },
]
const expansions = [
  { key: 'ratio', label: '整体扩展 (%)' },
  { key: 'top', label: '上方扩展 (%)' },
  { key: 'bottom', label: '下方扩展 (%)' },
  { key: 'left', label: '左侧扩展 (%)' },
  { key: 'right', label: '右侧扩展 (%)' },
] as const
</script>
<template>
  <div class="view-heading">
    <h2>{{ titles[section] }}</h2>
    <span class="save-state" :class="{ error }">{{
      error ? '未保存' : saving ? '保存中…' : dirty ? '待保存' : '自动保存'
    }}</span>
  </div>
  <p class="footnote">与翻译页面共用设置，修改会同时影响翻译器。新任务使用新配置。</p>
  <div v-if="error" class="notice error" role="status">{{ error }}</div>
  <div v-if="error" class="actions">
    <button v-if="dirty" class="button" :disabled="saving || loading" @click="save">重试保存</button
    ><button class="button subtle" :disabled="saving || loading" @click="load">
      {{ dirty ? '放弃修改并重新读取' : '重新读取' }}
    </button>
  </div>
  <p v-if="loading" class="muted">正在读取翻译器设置…</p>
  <form v-if="s" :ref="bindForm" @submit.prevent="save">
    <fieldset :disabled="loading">
      <template v-if="section === 'ocr'">
        <section class="setting-group">
          <h3>识别引擎</h3>
          <label class="field"
            >OCR 引擎<SelectControl
              :model-value="s.ocrEngine"
              :options="allOcrEngineOptions"
              label="OCR 引擎"
              @update:model-value="ocrEngine"
          /></label>
          <label class="switch-field"
            >启用混合 OCR<input
              class="switch"
              type="checkbox"
              :checked="s.hybridOcr.enabled"
              @change="hybrid(($event.target as HTMLInputElement).checked)"
          /></label>
          <template v-if="s.hybridOcr.enabled"
            ><p class="footnote">混合识别支持 48px OCR 与 MangaOCR。</p>
            <label class="field"
              >备用 OCR<SelectControl
                v-model="s.hybridOcr.secondaryEngine"
                :options="
                  allOcrEngineOptions.filter(
                    o => ['manga_ocr', '48px_ocr'].includes(o.value) && o.value !== s!.ocrEngine
                  )
                "
                label="备用 OCR"
                @update:model-value="change()" /></label
            ><NumberControl
              v-model="s.hybridOcr.confidenceThreshold"
              label="混合阈值"
              :min="0"
              :max="1"
              :step="0.01"
              @update:model-value="change()"
          /></template>
          <label v-if="s.ocrEngine === 'paddleocr_vl'" class="field"
            >源语言<SelectControl
              v-model="s.paddleOcrVl.sourceLanguage"
              :options="languageOptions"
              label="PaddleOCR-VL 源语言"
              searchable
              @update:model-value="change()"
          /></label>
        </section>
        <section v-if="s.ocrEngine === 'baidu_ocr'" class="setting-group">
          <h3>百度 OCR</h3>
          <label class="field"
            >版本<SelectControl
              v-model="s.baiduOcr.version"
              :options="baiduVersionOptions"
              label="百度 OCR 版本"
              @update:model-value="change('baiduOcr')" /></label
          ><label class="field"
            >源语言<SelectControl
              v-model="s.baiduOcr.sourceLanguage"
              :options="baiduSourceLanguageOptions"
              label="百度 OCR 源语言"
              @update:model-value="change('baiduOcr')"
          /></label>
          <label class="field"
            >百度 API Key<input
              :type="reveal ? 'text' : 'password'"
              :value="secretValue('ocr', 'baidu', 'baidu_api_key')"
              autocomplete="new-password"
              @input="
                updateSecret('baiduOcr', 'baidu_api_key', ($event.target as HTMLInputElement).value)
              " /></label
          ><label class="field"
            >百度 Secret Key<input
              :type="reveal ? 'text' : 'password'"
              :value="secretValue('ocr', 'baidu', 'baidu_secret_key')"
              autocomplete="new-password"
              @input="
                updateSecret(
                  'baiduOcr',
                  'baidu_secret_key',
                  ($event.target as HTMLInputElement).value
                )
              " /></label
          ><button type="button" class="text-button" @click="reveal = !reveal">
            {{ reveal ? '隐藏密钥' : '显示密钥' }}
          </button>
          <button type="button" class="button" :disabled="baiduChecking" @click="testBaidu">
            测试百度 OCR 连接
          </button>
          <p v-if="baiduMessage" class="notice" role="status">{{ baiduMessage }}</p>
        </section>
      </template>
      <section
        v-if="
          section === 'translation' ||
          section === 'hq' ||
          (section === 'ocr' && s.ocrEngine === 'ai_vision')
        "
        class="setting-group"
      >
        <h3>{{ section === 'ocr' ? 'AI 视觉 OCR' : '服务配置' }}</h3>
        <ServiceFields
          :key="serviceKey"
          :config="service"
          :domain="domain"
          :capability="
            section === 'ocr' ? 'visionOcr' : section === 'hq' ? 'hqTranslation' : 'translation'
          "
          :api="globalApi"
          :secret="keyValue"
          @provider="provider"
          @change="patchService"
          @secret="value => updateSecret(serviceKey, secretKey, value)"
        />
        <template v-if="section === 'ocr'"
          ><NumberControl
            v-model="s.aiVisionOcr.minImageSize"
            label="最小图片尺寸"
            :min="1"
            @update:model-value="change('aiVisionOcr')" /><label class="field"
            >提示词格式<SelectControl
              :model-value="s.aiVisionOcr.promptMode"
              :options="promptModeOptions"
              label="OCR 提示词格式"
              @update:model-value="ocrPromptMode" /></label
          ><label v-if="s.aiVisionOcr.promptMode === 'paddleocr_vl'" class="field"
            >源语言<SelectControl
              v-model="s.paddleOcrVl.sourceLanguage"
              :options="languageOptions"
              label="OCR 模型源语言"
              searchable
              @update:model-value="ocrPromptMode('paddleocr_vl')" /></label
          ><label class="field"
            >OCR 提示词<textarea
              v-model="s.aiVisionOcr.prompt"
              rows="6"
              @input="change('aiVisionOcr')"
            /></label
        ></template>
        <template v-if="section === 'translation'"
          ><label class="field"
            >翻译模式<SelectControl
              :model-value="s.translation.translationMode"
              :options="[
                { value: 'batch', label: '整页批量翻译' },
                { value: 'single', label: '逐气泡翻译' },
              ]"
              label="翻译模式"
              @update:model-value="translateMode" /></label
        ></template>
        <NumberControl
          v-if="section === 'hq'"
          v-model="s.hqTranslation.batchSize"
          label="批次大小"
          :min="1"
          :max="100"
          @update:model-value="change('hqTranslation')"
        />
        <label v-if="section !== 'ocr' && serviceMetadata?.supportsJsonResponse" class="switch-field"
          >强制 JSON 输出<input
            class="switch"
            type="checkbox"
            :checked="service.openaiOptions.request.forceJsonOutput"
            @change="jsonOutput(($event.target as HTMLInputElement).checked)"
        /></label>
      </section>
      <section v-if="(section === 'translation' || section === 'hq') && serviceMetadata?.kind !== 'adapter'" class="setting-group">
        <h3>提示词</h3>
        <label v-if="section === 'translation'" class="field"
          >翻译提示词<textarea
            :value="s.translation[promptKey]"
            rows="6"
            @input="translatePrompt(($event.target as HTMLTextAreaElement).value)"
          /></label
        ><label v-else class="field"
          >翻译偏好<textarea
            v-model="s.hqTranslation.prompt"
            rows="6"
            @input="change('hqTranslation')"
          /></label
        ><button type="button" class="text-button" @click="resetPrompt">重置为默认提示词</button
        ><template v-if="section === 'translation'"
          ><label class="switch-field"
            >启用文本框提示词<input
              class="switch"
              type="checkbox"
              v-model="s.useTextboxPrompt"
              @change="change()" /></label
          ><label v-if="s.useTextboxPrompt" class="field"
            >文本框提示词<textarea v-model="s.textboxPrompt" rows="4" @input="change()" /></label
        ></template>
      </section>
      <template v-if="section === 'detection'"
        ><section class="setting-group">
          <h3>文字检测</h3>
          <label class="field"
            >检测器类型<SelectControl
              v-model="s.textDetector"
              :options="detectorOptions"
              label="检测器类型"
              @update:model-value="change()" /></label
          ><NumberControl
            v-model="s.minTextBlockAreaPercent"
            label="最小文本框面积占比 (%)"
            :min="0"
            :max="100"
            :step="0.01"
            @update:model-value="change()"
          />
          <label class="switch-field"
            >启用辅助 YSGYolo 检测<input
              class="switch"
              type="checkbox"
              v-model="s.enableAuxYoloDetection"
              @change="change()"
          /></label>
          <div class="field-grid">
            <NumberControl
              v-model="s.auxYoloConfThreshold"
              label="辅助置信度"
              :min="0"
              :max="1"
              :step="0.05"
              @update:model-value="change()"
            /><NumberControl
              v-model="s.auxYoloOverlapThreshold"
              label="辅助重叠阈值"
              :min="0"
              :max="1"
              :step="0.05"
              @update:model-value="change()"
            />
          </div>
          <label class="switch-field"
            >启用 SaberYOLO 二阶段纠错<input
              class="switch"
              type="checkbox"
              v-model="s.enableSaberYoloRefine"
              @change="change()" /></label
          ><NumberControl
            v-model="s.saberYoloRefineOverlapThreshold"
            label="SaberYOLO 拆分阈值 (%)"
            :min="0"
            :max="100"
            :step="0.1"
            @update:model-value="change()"
          />
        </section>
        <section class="setting-group">
          <h3>文本框扩展</h3>
          <NumberControl
            v-for="field in expansions"
            :key="field.key"
            v-model="s.boxExpand[field.key]"
            :label="field.label"
            :min="0"
            :max="50"
            :step="0.1"
            @update:model-value="change()"
          />
        </section>
        <section class="setting-group">
          <h3>精确文字掩膜</h3>
          <NumberControl
            v-model="s.preciseMask.dilateSize"
            label="掩膜膨胀大小"
            :min="0"
            @update:model-value="change()"
          /><NumberControl
            v-model="s.preciseMask.boxExpandRatio"
            label="标注框扩大比例 (%)"
            :min="0"
            :max="100"
            :step="0.1"
            @update:model-value="change()"
          /><label class="switch-field"
            >显示检测框调试信息<input
              class="switch"
              type="checkbox"
              v-model="s.showDetectionDebug"
              @change="change()"
          /></label>
        </section>
      </template>
    </fieldset>
  </form>
</template>
