<template>
  <div class="translation-settings">
    <!-- 翻译服务配置 -->
    <div class="settings-group">
      <div class="settings-group-title">翻译服务配置</div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsModelProvider">翻译服务商:</label>
          <CustomSelect
            :model-value="localSettings.modelProvider"
            :options="providerOptions"
            @change="(v: any) => { localSettings.modelProvider = v; handleProviderChange() }"
          />
        </div>

        <!-- API Key (非本地服务显示) -->
        <div v-show="!isLocalProvider" class="settings-item">
          <label for="settingsApiKey">{{ apiKeyLabel }}:</label>
          <div class="password-input-wrapper">
            <input
              :type="showApiKey ? 'text' : 'password'"
              id="settingsApiKey"
              v-model="localSettings.apiKey"
              class="secure-input"
              :placeholder="apiKeyPlaceholder"
              autocomplete="off"
            />
            <button type="button" class="password-toggle-btn" tabindex="-1" @click="showApiKey = !showApiKey">
              <span class="eye-icon" v-if="!showApiKey">👁</span>
              <span class="eye-off-icon" v-else>👁‍🗨</span>
            </button>
          </div>
        </div>
      </div>

      <!-- 自定义Base URL -->
      <div v-show="localSettings.modelProvider === 'custom_openai'" class="settings-item">
        <label for="settingsCustomBaseUrl">Base URL:</label>
        <input
          type="text"
          id="settingsCustomBaseUrl"
          v-model="localSettings.customBaseUrl"
          placeholder="例如: https://api.example.com/v1"
        />
      </div>

      <!-- 模型名称 (非本地服务显示) -->
      <div v-show="!isLocalProvider" class="settings-item">
        <label for="settingsModelName">{{ modelNameLabel }}:</label>
        <div class="model-input-with-fetch">
          <input
            type="text"
            id="settingsModelName"
            v-model="localSettings.modelName"
            :placeholder="modelNamePlaceholder"
          />
          <button
            v-show="supportsFetchModels"
            type="button"
            class="fetch-models-btn"
            title="获取可用模型列表"
            @click="fetchModels"
            :disabled="isFetchingModels"
          >
            <span class="fetch-icon">🔍</span>
            <span class="fetch-text">{{ isFetchingModels ? '获取中...' : '获取模型' }}</span>
          </button>
        </div>
        <!-- 模型选择下拉框 -->
        <div v-if="modelList.length > 0" class="model-select-container">
          <CustomSelect
            :model-value="localSettings.modelName"
            :options="modelListOptions"
            @change="(v: any) => { localSettings.modelName = v }"
          />
          <span class="model-count">共 {{ modelList.length }} 个模型</span>
        </div>
      </div>

      <!-- 本地模型选择 (Ollama/Sakura) -->
      <div v-show="isLocalProvider" class="settings-item">
        <label>本地模型:</label>
        <div class="local-model-list">
          <div v-if="localSettings.modelProvider === 'ollama'" class="model-list-container">
            <button class="settings-test-btn" @click="fetchOllamaModels" :disabled="isFetchingModels">
              {{ isFetchingModels ? '获取中...' : '🔄 刷新模型列表' }}
            </button>
            <CustomSelect
              v-if="ollamaModels.length > 0"
              :model-value="localSettings.modelName"
              :options="ollamaModelOptions"
              @change="(v: any) => localSettings.modelName = v"
            />
            <p v-else class="model-hint">点击刷新获取可用模型</p>
          </div>
          <div v-else-if="localSettings.modelProvider === 'sakura'" class="model-list-container">
            <button class="settings-test-btn" @click="fetchSakuraModels" :disabled="isFetchingModels">
              {{ isFetchingModels ? '获取中...' : '🔄 刷新模型列表' }}
            </button>
            <CustomSelect
              v-if="sakuraModels.length > 0"
              :model-value="localSettings.modelName"
              :options="sakuraModelOptions"
              @change="(v: any) => localSettings.modelName = v"
            />
            <p v-else class="model-hint">点击刷新获取可用模型</p>
          </div>
        </div>
      </div>

      <!-- RPM限制 (云服务显示) -->
      <div v-show="showRpmLimit" class="settings-row">
        <div class="settings-item">
          <label for="settingsRpmTranslation">RPM限制:</label>
          <input type="number" id="settingsRpmTranslation" v-model.number="localSettings.rpmTranslation" min="0" step="1" />
          <div class="input-hint">每分钟请求数，0表示无限制</div>
        </div>
        <div class="settings-item">
          <label for="settingsTranslationMaxRetries">重试次数:</label>
          <input
            type="number"
            id="settingsTranslationMaxRetries"
            v-model.number="localSettings.translationMaxRetries"
            min="0"
            max="10"
            step="1"
          />
        </div>
      </div>

      <!-- 本地服务测试按钮 -->
      <div v-show="isLocalProvider" class="settings-item">
        <button class="settings-test-btn" @click="testLocalConnection" :disabled="isTesting">
          {{ isTesting ? '测试中...' : '🔗 测试连接' }}
        </button>
      </div>

      <!-- 云服务商测试按钮（复刻原版） -->
      <div v-show="!isLocalProvider" class="settings-item">
        <button class="settings-test-btn" @click="testCloudConnection" :disabled="isTesting">
          {{ isTesting ? '测试中...' : '🔗 测试连接' }}
        </button>
      </div>
    </div>

    <!-- 提示词设置 -->
    <div class="settings-group">
      <div class="settings-group-title">提示词设置</div>
      <div class="settings-item">
        <label for="settingsPromptContent">翻译提示词:</label>
        <textarea id="settingsPromptContent" v-model="localSettings.promptContent" rows="4" placeholder="翻译提示词"></textarea>
        <div class="prompt-format-selector">
          <CustomSelect
            :model-value="localSettings.translatePromptMode"
            :options="promptModeOptions"
            @change="(v: any) => { localSettings.translatePromptMode = v; handlePromptModeChange() }"
          />
          <span class="input-hint">JSON格式输出更结构化</span>
        </div>
        <!-- 快速选择提示词 -->
        <SavedPromptsPicker
          prompt-type="translate"
          @select="handleTranslatePromptSelect"
        />
        <!-- 重置为默认按钮 -->
        <button type="button" class="reset-btn" @click="resetTranslatePromptToDefault">
          重置为默认
        </button>
      </div>

      <!-- 文本框提示词 -->
      <div class="settings-item">
        <label class="checkbox-label">
          <input type="checkbox" v-model="localSettings.enableTextboxPrompt" />
          启用文本框提示词
        </label>
      </div>
      <div v-show="localSettings.enableTextboxPrompt" class="settings-item">
        <label for="settingsTextboxPromptContent">文本框提示词:</label>
        <textarea
          id="settingsTextboxPromptContent"
          v-model="localSettings.textboxPromptContent"
          rows="3"
          placeholder="文本框提示词"
        ></textarea>
        <!-- 快速选择提示词 -->
        <SavedPromptsPicker
          prompt-type="textbox"
          @select="handleTextboxPromptSelect"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 翻译服务设置组件
 * 管理翻译服务商选择和配置
 * 支持服务商配置分组存储
 */
import { ref, computed, watch } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
import { DEFAULT_TRANSLATE_PROMPT, DEFAULT_TRANSLATE_JSON_PROMPT } from '@/constants'
import type { TranslationProvider } from '@/types/settings'
import CustomSelect from '@/components/common/CustomSelect.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'

/** 翻译服务商选项 */
const providerOptions = [
  { label: 'SiliconFlow', value: 'siliconflow' },
  { label: 'DeepSeek', value: 'deepseek' },
  { label: '火山引擎', value: 'volcano' },
  { label: '彩云小译', value: 'caiyun' },
  { label: '百度翻译', value: 'baidu_translate' },
  { label: '有道翻译', value: 'youdao_translate' },
  { label: 'Google Gemini', value: 'gemini' },
  { label: 'Ollama (本地)', value: 'ollama' },
  { label: 'Sakura (本地)', value: 'sakura' },
  { label: '自定义 OpenAI 兼容服务', value: 'custom_openai' }
]

/** 提示词模式选项 */
const promptModeOptions = [
  { label: '普通提示词', value: 'normal' },
  { label: 'JSON提示词', value: 'json' }
]

// Store
const settingsStore = useSettingsStore()
const toast = useToast()

// 本地状态（双向绑定用）
const localSettings = ref({
  modelProvider: settingsStore.settings.translation.provider,
  apiKey: settingsStore.settings.translation.apiKey,
  modelName: settingsStore.settings.translation.modelName,
  customBaseUrl: settingsStore.settings.translation.customBaseUrl,
  rpmTranslation: settingsStore.settings.translation.rpmLimit,
  translationMaxRetries: settingsStore.settings.translation.maxRetries,
  promptContent: settingsStore.settings.translatePrompt,
  translatePromptMode: settingsStore.settings.translation.isJsonMode ? 'json' : 'normal',
  enableTextboxPrompt: settingsStore.settings.useTextboxPrompt,
  textboxPromptContent: settingsStore.settings.textboxPrompt
})

// 密码显示状态
const showApiKey = ref(false)

// 测试状态
const isTesting = ref(false)

// 模型获取状态
const isFetchingModels = ref(false)
const modelList = ref<string[]>([])
const ollamaModels = ref<string[]>([])
const sakuraModels = ref<string[]>([])

/** 模型列表选项（用于CustomSelect） */
const modelListOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  modelList.value.forEach(model => options.push({ label: model, value: model }))
  return options
})

/** Ollama模型选项（用于CustomSelect） */
const ollamaModelOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  ollamaModels.value.forEach(model => options.push({ label: model, value: model }))
  return options
})

/** Sakura模型选项（用于CustomSelect） */
const sakuraModelOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  sakuraModels.value.forEach(model => options.push({ label: model, value: model }))
  return options
})

// 计算属性：是否为本地服务商
const isLocalProvider = computed(() => {
  return ['ollama', 'sakura'].includes(localSettings.value.modelProvider)
})

// 计算属性：是否显示RPM限制
const showRpmLimit = computed(() => {
  return !['ollama', 'sakura', 'caiyun', 'baidu_translate', 'youdao_translate'].includes(localSettings.value.modelProvider)
})

// 计算属性：是否支持获取模型列表
const supportsFetchModels = computed(() => {
  return ['siliconflow', 'deepseek', 'volcano', 'gemini', 'custom_openai'].includes(localSettings.value.modelProvider)
})

// 计算属性：API Key 标签
const apiKeyLabel = computed(() => {
  switch (localSettings.value.modelProvider) {
    case 'baidu_translate':
      return 'App ID'
    case 'youdao_translate':
      return 'App Key'
    case 'caiyun':
      return 'API Token'
    default:
      return 'API Key'
  }
})

// 计算属性：API Key 占位符
const apiKeyPlaceholder = computed(() => {
  switch (localSettings.value.modelProvider) {
    case 'baidu_translate':
      return '请输入百度翻译App ID'
    case 'youdao_translate':
      return '请输入有道翻译应用ID'
    case 'caiyun':
      return '请输入彩云小译Token'
    default:
      return '请输入API Key'
  }
})

// 计算属性：模型名称标签
const modelNameLabel = computed(() => {
  switch (localSettings.value.modelProvider) {
    case 'baidu_translate':
      return 'App Key'
    case 'youdao_translate':
      return 'App Secret'
    case 'caiyun':
      return '源语言 (可选)'
    default:
      return '模型名称'
  }
})

// 计算属性：模型名称占位符
const modelNamePlaceholder = computed(() => {
  switch (localSettings.value.modelProvider) {
    case 'baidu_translate':
      return '请输入百度翻译App Key'
    case 'youdao_translate':
      return '请输入有道翻译应用密钥'
    case 'caiyun':
      return '可选: auto/日语/英语'
    default:
      return '请输入模型名称'
  }
})

// 处理服务商切换
function handleProviderChange() {
  const newProvider = localSettings.value.modelProvider as TranslationProvider
  
  // 使用 store 的方法切换服务商（会自动保存旧配置、恢复新配置）
  settingsStore.setTranslationProvider(newProvider)
  
  // 从 store 同步恢复的配置到本地状态
  localSettings.value.apiKey = settingsStore.settings.translation.apiKey
  localSettings.value.modelName = settingsStore.settings.translation.modelName
  localSettings.value.customBaseUrl = settingsStore.settings.translation.customBaseUrl
  localSettings.value.rpmTranslation = settingsStore.settings.translation.rpmLimit
  localSettings.value.translationMaxRetries = settingsStore.settings.translation.maxRetries
  
  // 清空模型列表
  modelList.value = []
  

}

// 处理提示词模式切换
function handlePromptModeChange() {
  const isJsonMode = localSettings.value.translatePromptMode === 'json'
  
  // 更新提示词内容
  if (isJsonMode) {
    localSettings.value.promptContent = DEFAULT_TRANSLATE_JSON_PROMPT
  } else {
    localSettings.value.promptContent = DEFAULT_TRANSLATE_PROMPT
  }
  
  // 同步到 store
  settingsStore.updateTranslationService({ isJsonMode })
  settingsStore.setTranslatePrompt(localSettings.value.promptContent)
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

watch(() => localSettings.value.translationMaxRetries, (newVal) => {
  settingsStore.updateTranslationService({ maxRetries: newVal })
})

watch(() => localSettings.value.promptContent, (newVal) => {
  settingsStore.setTranslatePrompt(newVal)
})

watch(() => localSettings.value.enableTextboxPrompt, (newVal) => {
  settingsStore.setUseTextboxPrompt(newVal)
})

watch(() => localSettings.value.textboxPromptContent, (newVal) => {
  settingsStore.setTextboxPrompt(newVal)
})

// 获取模型列表（复刻原版 doFetchModels 逻辑）
async function fetchModels() {
  const provider = localSettings.value.modelProvider
  const apiKey = localSettings.value.apiKey?.trim()
  const baseUrl = localSettings.value.customBaseUrl?.trim()

  // 验证（与原版一致）
  if (!apiKey) {
    toast.warning('请先填写 API Key')
    return
  }

  // 检查是否支持模型获取（与原版一致）
  const supportedProviders = ['siliconflow', 'deepseek', 'volcano', 'gemini', 'custom_openai']
  if (!supportedProviders.includes(provider)) {
    toast.warning(`${getProviderDisplayName(provider)} 不支持自动获取模型列表`)
    return
  }

  // 自定义服务需要 base_url（与原版一致）
  if (provider === 'custom_openai' && !baseUrl) {
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

// 获取服务商显示名称（与原版一致）
function getProviderDisplayName(provider: string): string {
  const names: Record<string, string> = {
    'siliconflow': 'SiliconFlow',
    'deepseek': 'DeepSeek',
    'volcano': '火山引擎',
    'gemini': 'Google Gemini',
    'custom_openai': '自定义OpenAI',
    'ollama': 'Ollama',
    'sakura': 'Sakura',
    'caiyun': '彩云小译',
    'baidu_translate': '百度翻译',
    'youdao_translate': '有道翻译'
  }
  return names[provider] || provider
}

// 获取Ollama模型列表
async function fetchOllamaModels() {
  isFetchingModels.value = true
  try {
    const result = await configApi.testOllamaConnection()
    if (result.success && result.models) {
      ollamaModels.value = result.models
      toast.success(`获取到 ${result.models.length} 个Ollama模型`)
    } else {
      toast.error(result.error || 'Ollama连接失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '获取Ollama模型失败'
    toast.error(errorMessage)
  } finally {
    isFetchingModels.value = false
  }
}

// 获取Sakura模型列表
async function fetchSakuraModels() {
  isFetchingModels.value = true
  try {
    const result = await configApi.testSakuraConnection()
    if (result.success && result.models) {
      sakuraModels.value = result.models
      toast.success(`获取到 ${result.models.length} 个Sakura模型`)
    } else {
      toast.error(result.error || 'Sakura连接失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '获取Sakura模型失败'
    toast.error(errorMessage)
  } finally {
    isFetchingModels.value = false
  }
}

// 测试本地服务连接
async function testLocalConnection() {
  isTesting.value = true
  try {
    let result
    if (localSettings.value.modelProvider === 'ollama') {
      result = await configApi.testOllamaConnection()
    } else {
      result = await configApi.testSakuraConnection()
    }
    if (result.success) {
      toast.success(`${localSettings.value.modelProvider === 'ollama' ? 'Ollama' : 'Sakura'} 连接成功`)
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

// 测试云服务商连接（复刻原版 testTranslationConnection 逻辑）
async function testCloudConnection() {
  const provider = localSettings.value.modelProvider
  const apiKey = localSettings.value.apiKey?.trim()
  const modelName = localSettings.value.modelName?.trim()
  const baseUrl = localSettings.value.customBaseUrl?.trim()

  // 验证必填字段（与原版一致）
  if (!apiKey) {
    toast.warning('请先填写 API Key')
    return
  }

  // 非彩云小译需要模型名称
  if (provider !== 'caiyun' && !modelName) {
    toast.warning('请填写模型名称')
    return
  }

  // 自定义服务需要 base_url
  if (provider === 'custom_openai' && !baseUrl) {
    toast.warning('自定义服务需要填写 Base URL')
    return
  }

  isTesting.value = true
  toast.info('正在测试连接...')

  try {
    let result

    // 根据服务商类型分发到不同的测试函数（与原版一致）
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
        // 其他 AI 服务商使用通用接口
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

// 处理翻译提示词选择
function handleTranslatePromptSelect(content: string, name: string) {
  localSettings.value.promptContent = content
  toast.success(`已应用提示词: ${name}`)
}

// 处理文本框提示词选择
function handleTextboxPromptSelect(content: string, name: string) {
  localSettings.value.textboxPromptContent = content
  toast.success(`已应用提示词: ${name}`)
}

// 重置翻译提示词为默认值
function resetTranslatePromptToDefault() {
  const isJsonMode = localSettings.value.translatePromptMode === 'json'
  if (isJsonMode) {
    localSettings.value.promptContent = DEFAULT_TRANSLATE_JSON_PROMPT
  } else {
    localSettings.value.promptContent = DEFAULT_TRANSLATE_PROMPT
  }
  toast.success('已重置为默认提示词')
}
</script>

<style scoped>
.model-hint {
  color: var(--text-secondary);
  font-size: 12px;
  margin-top: 5px;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.checkbox-label input[type='checkbox'] {
  width: auto;
}

/* 密码输入框包装器 */
.password-input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.password-input-wrapper input {
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
  color: var(--text-secondary, #666);
  background: transparent;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.reset-btn:hover {
  color: var(--primary-color, #4a90d9);
  border-color: var(--primary-color, #4a90d9);
  background: rgba(74, 144, 217, 0.05);
}
</style>
