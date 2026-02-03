<template>
  <div class="ocr-settings">
    <!-- OCR引擎选择 -->
    <div class="settings-group">
      <div class="settings-group-title">OCR引擎选择</div>
      <div class="settings-item">
        <label for="settingsOcrEngine">OCR引擎:</label>
        <CustomSelect
          :model-value="settings.ocrEngine"
          :options="ocrEngineOptions"
          @change="(v: any) => { settings.ocrEngine = v; handleOcrEngineChange() }"
        />
      </div>
      
      <!-- 通用源语言选择（仅PaddleOCR使用） -->
      <div v-show="settings.ocrEngine === 'paddle_ocr'" class="settings-item">
        <label for="settingsSourceLanguage">源语言:</label>
        <CustomSelect
          :model-value="settings.sourceLanguage"
          :groups="sourceLanguageGroups"
          @change="(v: any) => { settings.sourceLanguage = v; handleSourceLanguageChange() }"
        />
        <div class="input-hint">
          {{ getSourceLanguageHint() }}
        </div>
      </div>
    </div>

    <!-- PaddleOCR-VL 源语言选择 -->
    <div v-show="settings.ocrEngine === 'paddleocr_vl'" class="settings-group">
      <div class="settings-group-title">PaddleOCR-VL 设置</div>
      <div class="settings-item">
        <label for="settingsPaddleOcrVlSourceLanguage">源语言:</label>
        <CustomSelect
          :model-value="settings.paddleOcrVl.sourceLanguage"
          :groups="paddleOcrVlSourceLanguageGroups"
          @change="(v: any) => handlePaddleOcrVlSourceLanguageChange(v)"
        />
        <div class="input-hint">
          选择图像中的源语言，用于优化 OCR 识别效果
        </div>
      </div>
    </div>

    <!-- 百度OCR设置 -->
    <div v-show="settings.ocrEngine === 'baidu_ocr'" class="settings-group">
      <div class="settings-group-title">百度OCR 设置</div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsBaiduApiKey">API Key:</label>
          <div class="password-input-wrapper">
            <input
              :type="showBaiduApiKey ? 'text' : 'password'"
              id="settingsBaiduApiKey"
              v-model="localBaiduOcr.apiKey"
              class="secure-input"
              placeholder="请输入百度OCR API Key"
              autocomplete="off"
            />
            <button type="button" class="password-toggle-btn" tabindex="-1" @click="showBaiduApiKey = !showBaiduApiKey">
              <span class="eye-icon" v-if="!showBaiduApiKey">👁</span>
              <span class="eye-off-icon" v-else>👁‍🗨</span>
            </button>
          </div>
        </div>
        <div class="settings-item">
          <label for="settingsBaiduSecretKey">Secret Key:</label>
          <div class="password-input-wrapper">
            <input
              :type="showBaiduSecretKey ? 'text' : 'password'"
              id="settingsBaiduSecretKey"
              v-model="localBaiduOcr.secretKey"
              class="secure-input"
              placeholder="请输入Secret Key"
              autocomplete="off"
            />
            <button type="button" class="password-toggle-btn" tabindex="-1" @click="showBaiduSecretKey = !showBaiduSecretKey">
              <span class="eye-icon" v-if="!showBaiduSecretKey">👁</span>
              <span class="eye-off-icon" v-else>👁‍🗨</span>
            </button>
          </div>
        </div>
      </div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsBaiduVersion">识别版本:</label>
          <CustomSelect
            v-model="localBaiduOcr.version"
            :options="baiduVersionOptions"
          />
        </div>
        <div class="settings-item">
          <label for="settingsBaiduSourceLanguage">源语言:</label>
          <CustomSelect
            v-model="localBaiduOcr.sourceLanguage"
            :options="baiduSourceLanguageOptions"
          />
        </div>
      </div>
      <button class="settings-test-btn" @click="testBaiduOcr" :disabled="isTesting">
        {{ isTesting ? '测试中...' : '🔗 测试连接' }}
      </button>
    </div>

    <!-- AI视觉OCR设置 -->
    <div v-show="settings.ocrEngine === 'ai_vision'" class="settings-group">
      <div class="settings-group-title">AI视觉OCR 设置</div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsAiVisionProvider">服务商:</label>
          <CustomSelect
            :model-value="settings.aiVisionOcr.provider"
            :options="aiVisionProviderOptions"
            @change="(v: any) => handleAiVisionProviderChange(v)"
          />
        </div>
        <div class="settings-item">
          <label for="settingsAiVisionApiKey">API Key:</label>
          <div class="password-input-wrapper">
            <input
              :type="showAiVisionApiKey ? 'text' : 'password'"
              id="settingsAiVisionApiKey"
              v-model="localAiVisionOcr.apiKey"
              class="secure-input"
              placeholder="请输入API Key"
              autocomplete="off"
            />
            <button type="button" class="password-toggle-btn" tabindex="-1" @click="showAiVisionApiKey = !showAiVisionApiKey">
              <span class="eye-icon" v-if="!showAiVisionApiKey">👁</span>
              <span class="eye-off-icon" v-else>👁‍🗨</span>
            </button>
          </div>
        </div>
      </div>

      <!-- 自定义Base URL -->
      <div v-show="settings.aiVisionOcr.provider === 'custom_openai_vision'" class="settings-item">
        <label for="settingsCustomAiVisionBaseUrl">Base URL:</label>
        <input
          type="text"
          id="settingsCustomAiVisionBaseUrl"
          v-model="localAiVisionOcr.customBaseUrl"
          placeholder="例如: https://api.example.com/v1"
        />
      </div>

      <!-- 模型名称 -->
      <div class="settings-item">
        <label for="settingsAiVisionModelName">模型名称:</label>
        <div class="model-input-with-fetch">
          <input
            type="text"
            id="settingsAiVisionModelName"
            v-model="localAiVisionOcr.modelName"
            placeholder="如: silicon-llava2-34b"
          />
          <button
            type="button"
            class="fetch-models-btn"
            title="获取可用模型列表"
            @click="fetchAiVisionModels"
            :disabled="isFetchingModels"
          >
            <span class="fetch-icon">🔍</span>
            <span class="fetch-text">{{ isFetchingModels ? '获取中...' : '获取模型' }}</span>
          </button>
        </div>
        <!-- 模型选择下拉框 -->
        <div v-if="aiVisionModels.length > 0" class="model-select-container">
          <CustomSelect
            v-model="localAiVisionOcr.modelName"
            :options="aiVisionModelOptions"
          />
          <span class="model-count">共 {{ aiVisionModels.length }} 个模型</span>
        </div>
      </div>

      <!-- OCR提示词 -->
      <div class="settings-item">
        <label for="settingsAiVisionOcrPrompt">OCR提示词:</label>
        <textarea
          id="settingsAiVisionOcrPrompt"
          v-model="localAiVisionOcr.prompt"
          rows="3"
          placeholder="AI视觉OCR提示词"
        ></textarea>
        <!-- 快速选择提示词 -->
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
          <span class="input-hint">{{ getPromptModeHint() }}</span>
        </div>
        <!-- PaddleOCR-VL 源语言选择器 -->
        <div v-if="currentPromptMode === 'paddleocr_vl'" class="paddleocr-vl-lang-selector">
          <label>源语言:</label>
          <CustomSelect
            :model-value="paddleOcrVlSourceLang"
            :groups="paddleOcrVlSourceLanguageGroups"
            @change="(v: string | number) => handlePaddleOcrVlLangChange(String(v))"
          />
        </div>
      </div>

      <!-- RPM限制 -->
      <div class="settings-item">
        <label for="settingsRpmAiVisionOcr">RPM限制 (每分钟请求数):</label>
        <input type="number" id="settingsRpmAiVisionOcr" v-model.number="localAiVisionOcr.rpmLimit" min="0" step="1" />
        <div class="input-hint">0 表示无限制</div>
      </div>

      <!-- 最小图片尺寸 -->
      <div class="settings-item">
        <label for="settingsMinImageSize">最小图片尺寸 (像素):</label>
        <input type="number" id="settingsMinImageSize" v-model.number="localAiVisionOcr.minImageSize" min="0" step="1" />
        <div class="input-hint">VLM模型通常要求图片尺寸 ≥28px，设为0则不自动放大小图</div>
      </div>

      <button class="settings-test-btn" @click="testAiVisionOcr" :disabled="isTesting">
        {{ isTesting ? '测试中...' : '🔗 测试连接' }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * OCR设置组件
 * 管理OCR引擎选择和各引擎的配置
 */
import { ref, computed, watch } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
import {
  DEFAULT_AI_VISION_OCR_PROMPT,
  DEFAULT_AI_VISION_OCR_JSON_PROMPT,
  getPaddleOcrVlPrompt,
  PADDLEOCR_VL_LANG_MAP
} from '@/constants'
import CustomSelect from '@/components/common/CustomSelect.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'

/** OCR引擎选项 */
const ocrEngineOptions = [
  { label: 'MangaOCR (日语专用)', value: 'manga_ocr' },
  { label: 'PaddleOCR (多语言)', value: 'paddle_ocr' },
  { label: 'PaddleOCR-VL', value: 'paddleocr_vl' },
  { label: '百度OCR', value: 'baidu_ocr' },
  { label: '48px OCR', value: '48px_ocr' },
  { label: 'AI视觉OCR', value: 'ai_vision' }
]

/** 百度OCR版本选项 */
const baiduVersionOptions = [
  { label: '标准版', value: 'standard' },
  { label: '高精度版', value: 'high_precision' }
]

/** 百度OCR源语言选项 */
const baiduSourceLanguageOptions = [
  { label: '自动检测', value: 'auto_detect' },
  { label: '中英文混合', value: 'CHN_ENG' },
  { label: '英文', value: 'ENG' },
  { label: '日语', value: 'JAP' },
  { label: '韩语', value: 'KOR' },
  { label: '法语', value: 'FRE' },
  { label: '德语', value: 'GER' },
  { label: '俄语', value: 'RUS' }
]

/** AI视觉服务商选项 */
const aiVisionProviderOptions = [
  { label: 'SiliconFlow (硅基流动)', value: 'siliconflow' },
  { label: '火山引擎', value: 'volcano' },
  { label: 'Google Gemini', value: 'gemini' },
  { label: '自定义 OpenAI 兼容服务', value: 'custom_openai_vision' }
]

/** PaddleOCR-VL 源语言选项（分组） */
const paddleOcrVlSourceLanguageGroups = [
  {
    label: '🎌 东亚语言',
    options: [
      { label: '日语', value: 'japanese' },
      { label: '简体中文', value: 'chinese' },
      { label: '繁体中文', value: 'chinese_cht' },
      { label: '韩语', value: 'korean' }
    ]
  },
  {
    label: '🌍 拉丁语系',
    options: [
      { label: '英语', value: 'english' },
      { label: '法语', value: 'french' },
      { label: '德语', value: 'german' },
      { label: '西班牙语', value: 'spanish' },
      { label: '意大利语', value: 'italian' },
      { label: '葡萄牙语', value: 'portuguese' },
      { label: '荷兰语', value: 'dutch' },
      { label: '波兰语', value: 'polish' }
    ]
  },
  {
    label: '🌏 东南亚语言',
    options: [
      { label: '泰语', value: 'thai' },
      { label: '越南语', value: 'vietnamese' },
      { label: '印尼语', value: 'indonesian' },
      { label: '马来语', value: 'malay' }
    ]
  },
  {
    label: '🌐 其他语系',
    options: [
      { label: '俄语', value: 'russian' },
      { label: '阿拉伯语', value: 'arabic' },
      { label: '印地语', value: 'hindi' },
      { label: '土耳其语', value: 'turkish' },
      { label: '希腊语', value: 'greek' },
      { label: '希伯来语', value: 'hebrew' }
    ]
  }
]

/** 提示词模式选项 */
const promptModeOptions = [
  { label: '普通提示词', value: 'normal' },
  { label: 'JSON提示词', value: 'json' },
  { label: 'OCR模型提示词', value: 'paddleocr_vl' }
]

/** 源语言选项（分组） */
const sourceLanguageGroups = [
  {
    label: '🚀 常用语言',
    options: [
      { label: '日语', value: 'japanese' },
      { label: '英语', value: 'en' },
      { label: '简体中文', value: 'chinese' },
      { label: '繁体中文', value: 'chinese_cht' },
      { label: '韩语', value: 'korean' }
    ]
  },
  {
    label: '🌍 拉丁语系',
    options: [
      { label: '法语', value: 'french' },
      { label: '德语', value: 'german' },
      { label: '西班牙语', value: 'spanish' },
      { label: '意大利语', value: 'italian' },
      { label: '葡萄牙语', value: 'portuguese' }
    ]
  },
  {
    label: '🌏 其他语系',
    options: [
      { label: '俄语', value: 'russian' }
    ]
  }
]

// Store
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
  rpmLimit: settingsStore.settings.aiVisionOcr.rpmLimit,
  minImageSize: settingsStore.settings.aiVisionOcr.minImageSize
})

// 直接访问 store 的只读设置（用于显示条件判断）
const settings = computed(() => settingsStore.settings)

// ============================================================
// Watch 同步：本地状态变化时自动保存到 store
// ============================================================

// 百度OCR设置同步
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

// AI视觉OCR设置同步
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
watch(() => localAiVisionOcr.value.minImageSize, (val) => {
  settingsStore.updateAiVisionOcr({ minImageSize: val })
})

// 密码显示状态
const showBaiduApiKey = ref(false)
const showBaiduSecretKey = ref(false)
const showAiVisionApiKey = ref(false)

// 测试状态
const isTesting = ref(false)

// 模型获取状态
const isFetchingModels = ref(false)
const aiVisionModels = ref<string[]>([])

/** AI视觉模型选项（用于CustomSelect） */
const aiVisionModelOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  aiVisionModels.value.forEach(model => {
    options.push({ label: model, value: model })
  })
  return options
})

// 处理OCR引擎切换
function handleOcrEngineChange() {
  settingsStore.saveToStorage()
}

// 处理源语言切换
function handleSourceLanguageChange() {
  settingsStore.saveToStorage()
}

// 处理 PaddleOCR-VL 源语言切换
function handlePaddleOcrVlSourceLanguageChange(value: string) {
  settingsStore.updatePaddleOcrVl({ sourceLanguage: value })
}

// 获取源语言提示信息
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

// 处理AI视觉服务商切换（复刻原版逻辑：独立保存每个服务商的配置）
function handleAiVisionProviderChange(newProvider: string) {
  // 使用 store 的方法切换服务商（会自动保存旧配置、恢复新配置）
  settingsStore.setAiVisionOcrProvider(newProvider)
  // 清空模型列表
  aiVisionModels.value = []
  // 同步本地状态（服务商切换后 store 会恢复新服务商的配置）
  syncLocalAiVisionOcr()
}

// 同步本地 AI 视觉 OCR 状态
function syncLocalAiVisionOcr() {
  localAiVisionOcr.value.apiKey = settingsStore.settings.aiVisionOcr.apiKey
  localAiVisionOcr.value.modelName = settingsStore.settings.aiVisionOcr.modelName
  localAiVisionOcr.value.customBaseUrl = settingsStore.settings.aiVisionOcr.customBaseUrl
  localAiVisionOcr.value.prompt = settingsStore.settings.aiVisionOcr.prompt
  localAiVisionOcr.value.rpmLimit = settingsStore.settings.aiVisionOcr.rpmLimit
  localAiVisionOcr.value.minImageSize = settingsStore.settings.aiVisionOcr.minImageSize
}
// 当前提示词模式（计算属性）
const currentPromptMode = computed(() => {
  const prompt = settingsStore.settings.aiVisionOcr.prompt
  // 检查是否为 PaddleOCR-VL 格式提示词
  if (prompt && prompt.includes('进行OCR:') && !prompt.includes('助手')) {
    return 'paddleocr_vl'
  }
  // 检查是否为 JSON 模式
  if (settingsStore.settings.aiVisionOcr.isJsonMode) {
    return 'json'
  }
  return 'normal'
})

// 获取提示词模式提示信息
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

// 处理提示词模式切换
function handlePromptModeChange(mode: string) {
  let newPrompt: string
  let isJsonMode = false
  
  switch (mode) {
    case 'json':
      newPrompt = DEFAULT_AI_VISION_OCR_JSON_PROMPT
      isJsonMode = true
      break
    case 'paddleocr_vl':
      // 使用当前选择的语言生成提示词
      const langName = PADDLEOCR_VL_LANG_MAP[paddleOcrVlSourceLang.value] || '日语'
      newPrompt = getPaddleOcrVlPrompt(langName)
      isJsonMode = false
      break
    default: // 'normal'
      newPrompt = DEFAULT_AI_VISION_OCR_PROMPT
      isJsonMode = false
      break
  }
  
  // 更新 store
  settingsStore.updateAiVisionOcr({ 
    prompt: newPrompt,
    isJsonMode: isJsonMode
  })
  
  // 同步本地状态
  localAiVisionOcr.value.prompt = newPrompt
}

// PaddleOCR-VL 源语言状态
const paddleOcrVlSourceLang = ref('japanese')

// 处理 PaddleOCR-VL 源语言切换
function handlePaddleOcrVlLangChange(langCode: string) {
  paddleOcrVlSourceLang.value = langCode
  
  // 根据新语言更新提示词
  const langName = PADDLEOCR_VL_LANG_MAP[langCode] || '日语'
  const newPrompt = getPaddleOcrVlPrompt(langName)
  
  // 更新 store
  settingsStore.updateAiVisionOcr({ prompt: newPrompt })
  
  // 同步本地状态
  localAiVisionOcr.value.prompt = newPrompt
}

// 测试百度OCR连接（复刻原版逻辑）
async function testBaiduOcr() {
  const apiKey = localBaiduOcr.value.apiKey?.trim()
  const secretKey = localBaiduOcr.value.secretKey?.trim()

  // 验证必填字段
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

// 测试AI视觉OCR连接
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

// 获取AI视觉模型列表（复刻原版 doFetchModels 逻辑）
async function fetchAiVisionModels() {
  const provider = settingsStore.settings.aiVisionOcr.provider
  const apiKey = localAiVisionOcr.value.apiKey?.trim()
  const baseUrl = localAiVisionOcr.value.customBaseUrl?.trim()

  // 验证（与原版一致）
  if (!apiKey) {
    toast.warning('请先填写 API Key')
    return
  }

  // 检查是否支持模型获取
  const supportedProviders = ['siliconflow', 'volcano', 'gemini', 'custom_openai_vision']
  if (!supportedProviders.includes(provider)) {
    toast.warning('当前服务商不支持自动获取模型列表')
    return
  }

  // 自定义服务需要 base_url
  if (provider === 'custom_openai_vision' && !baseUrl) {
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

// 处理 AI 视觉 OCR 提示词选择
function handleAiVisionPromptSelect(content: string, name: string) {
  settingsStore.updateAiVisionOcr({ prompt: content })
  // 同步本地状态
  localAiVisionOcr.value.prompt = content
  toast.success(`已应用提示词: ${name}`)
}
</script>

<style scoped>
.settings-test-btn {
  width: 100%;
  padding: 10px 16px;
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-primary);
  font-weight: 500;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.settings-test-btn:hover:not(:disabled) {
  background-color: var(--bg-hover);
  border-color: var(--primary-color);
  color: var(--primary-color);
}

.settings-test-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
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
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 13px;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.2s ease;
  height: 38px;
}

.fetch-models-btn:hover:not(:disabled) {
  background-color: var(--primary-color);
  color: #ffffff;
  border-color: var(--primary-color);
}

/* PaddleOCR-VL 语言选择器 */
.paddleocr-vl-lang-selector {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
  padding: 10px 12px;
  background: var(--bg-secondary);
  border-radius: 6px;
  border: 1px solid var(--border-color);
}

.paddleocr-vl-lang-selector label {
  font-size: 13px;
  color: var(--text-secondary);
  white-space: nowrap;
}

.paddleocr-vl-lang-selector .custom-select {
  flex: 1;
  min-width: 150px;
}
</style>
