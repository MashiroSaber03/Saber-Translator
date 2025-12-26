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
              v-model="settings.baiduOcr.apiKey"
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
              v-model="settings.baiduOcr.secretKey"
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
            :model-value="settings.baiduOcr.version"
            :options="baiduVersionOptions"
            @change="(v: any) => settings.baiduOcr.version = v"
          />
        </div>
        <div class="settings-item">
          <label for="settingsBaiduSourceLanguage">源语言:</label>
          <CustomSelect
            :model-value="settings.baiduOcr.sourceLanguage"
            :options="baiduSourceLanguageOptions"
            @change="(v: any) => settings.baiduOcr.sourceLanguage = v"
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
            @change="(v: any) => { settings.aiVisionOcr.provider = v; handleAiVisionProviderChange() }"
          />
        </div>
        <div class="settings-item">
          <label for="settingsAiVisionApiKey">API Key:</label>
          <div class="password-input-wrapper">
            <input
              :type="showAiVisionApiKey ? 'text' : 'password'"
              id="settingsAiVisionApiKey"
              v-model="settings.aiVisionOcr.apiKey"
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
          v-model="settings.aiVisionOcr.customBaseUrl"
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
            v-model="settings.aiVisionOcr.modelName"
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
            :model-value="settings.aiVisionOcr.modelName"
            :options="aiVisionModelOptions"
            @change="(v: any) => settings.aiVisionOcr.modelName = v"
          />
          <span class="model-count">共 {{ aiVisionModels.length }} 个模型</span>
        </div>
      </div>

      <!-- OCR提示词 -->
      <div class="settings-item">
        <label for="settingsAiVisionOcrPrompt">OCR提示词:</label>
        <textarea
          id="settingsAiVisionOcrPrompt"
          v-model="settings.aiVisionOcr.prompt"
          rows="3"
          placeholder="AI视觉OCR提示词"
        ></textarea>
        <div class="prompt-format-selector">
          <CustomSelect
            :model-value="settings.aiVisionOcr.isJsonMode"
            :options="promptModeOptions"
            @change="(v: any) => { settings.aiVisionOcr.isJsonMode = v; handleAiVisionPromptModeChange() }"
          />
          <span class="input-hint">JSON格式输出更结构化</span>
        </div>
      </div>

      <!-- RPM限制 -->
      <div class="settings-item">
        <label for="settingsRpmAiVisionOcr">RPM限制 (每分钟请求数):</label>
        <input type="number" id="settingsRpmAiVisionOcr" v-model.number="settings.aiVisionOcr.rpmLimit" min="0" step="1" />
        <div class="input-hint">0 表示无限制</div>
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
import { ref, computed } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
import {
  DEFAULT_AI_VISION_OCR_PROMPT,
  DEFAULT_AI_VISION_OCR_JSON_PROMPT
} from '@/constants'
import CustomSelect from '@/components/common/CustomSelect.vue'

/** OCR引擎选项 */
const ocrEngineOptions = [
  { label: 'MangaOCR (日语专用)', value: 'manga_ocr' },
  { label: 'PaddleOCR (多语言)', value: 'paddle_ocr' },
  { label: '百度OCR', value: 'baidu_ocr' },
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

/** 提示词模式选项 */
const promptModeOptions = [
  { label: '普通提示词', value: false },
  { label: 'JSON提示词', value: true }
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
// 直接访问 settingsStore.settings 以便 v-model 可以正确工作
const settings = computed(() => settingsStore.settings)
const toast = useToast()

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
  // 保存当前服务商配置（如果需要）
  settingsStore.saveToStorage()
}

// 处理源语言切换
function handleSourceLanguageChange() {
  // 保存设置
  settingsStore.saveToStorage()
  console.log(`源语言已切换为: ${settingsStore.settings.sourceLanguage}`)
}

// 获取源语言提示信息
function getSourceLanguageHint(): string {
  const engine = settingsStore.settings.ocrEngine
  switch (engine) {
    case 'manga_ocr':
      return 'MangaOCR 专为日语漫画优化，源语言设置不影响识别'
    case 'paddle_ocr':
      return 'PaddleOCR 会根据源语言加载对应的识别模型'
    case 'baidu_ocr':
      return '百度OCR 使用独立的源语言设置（见下方）'
    case 'ai_vision':
      return 'AI视觉OCR 通过提示词指定识别语言'
    default:
      return '选择要识别的原文语言'
  }
}

// 处理AI视觉服务商切换
function handleAiVisionProviderChange() {
  // 清空模型列表
  aiVisionModels.value = []
  settingsStore.saveToStorage()
}

// 处理AI视觉提示词模式切换
function handleAiVisionPromptModeChange() {
  // 切换模式时更新默认提示词
  if (settingsStore.settings.aiVisionOcr.isJsonMode) {
    settingsStore.settings.aiVisionOcr.prompt = DEFAULT_AI_VISION_OCR_JSON_PROMPT
  } else {
    settingsStore.settings.aiVisionOcr.prompt = DEFAULT_AI_VISION_OCR_PROMPT
  }
  settingsStore.saveToStorage()
}

// 测试百度OCR连接
async function testBaiduOcr() {
  isTesting.value = true
  try {
    const result = await configApi.testBaiduOcrConnection()
    if (result.success) {
      toast.success('百度OCR连接成功')
    } else {
      toast.error(`百度OCR连接失败: ${result.error || '未知错误'}`)
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
    const aiVisionOcr = settingsStore.settings.aiVisionOcr
    const result = await configApi.testAiVisionOcrConnection({
      provider: aiVisionOcr.provider,
      apiKey: aiVisionOcr.apiKey,
      modelName: aiVisionOcr.modelName,
      customBaseUrl: aiVisionOcr.customBaseUrl
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

// 获取AI视觉模型列表
async function fetchAiVisionModels() {
  isFetchingModels.value = true
  try {
    const aiVisionOcr = settingsStore.settings.aiVisionOcr
    const result = await configApi.getModelInfo(aiVisionOcr.provider, aiVisionOcr.apiKey)
    if (result.models && result.models.length > 0) {
      aiVisionModels.value = result.models
      toast.success(`获取到 ${result.models.length} 个模型`)
    } else {
      toast.warning('未获取到可用模型')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '获取模型列表失败'
    toast.error(errorMessage)
  } finally {
    isFetchingModels.value = false
  }
}
</script>

<style scoped>
/* 使用现有CSS样式 */
</style>
