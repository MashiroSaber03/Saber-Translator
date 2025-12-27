<template>
  <div class="hq-translation-settings">
    <!-- 高质量翻译服务配置 -->
    <div class="settings-group">
      <div class="settings-group-title">高质量翻译服务配置</div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsHqTranslateProvider">服务商:</label>
          <CustomSelect
            :model-value="hqSettings.provider"
            :options="providerOptions"
            @change="(v: any) => { hqSettings.provider = v; handleProviderChange() }"
          />
        </div>
        <div class="settings-item">
          <label for="settingsHqApiKey">API Key:</label>
          <div class="password-input-wrapper">
            <input
              :type="showApiKey ? 'text' : 'password'"
              id="settingsHqApiKey"
              v-model="hqSettings.apiKey"
              class="secure-input"
              placeholder="请输入API Key"
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
      <div v-show="hqSettings.provider === 'custom_openai'" class="settings-item">
        <label for="settingsHqCustomBaseUrl">Base URL:</label>
        <input
          type="text"
          id="settingsHqCustomBaseUrl"
          v-model="hqSettings.customBaseUrl"
          placeholder="例如: https://api.example.com/v1"
        />
      </div>

      <!-- 模型名称 -->
      <div class="settings-item">
        <label for="settingsHqModelName">模型名称:</label>
        <div class="model-input-with-fetch">
          <input type="text" id="settingsHqModelName" v-model="hqSettings.modelName" placeholder="请输入模型名称" />
          <button
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
            :model-value="hqSettings.modelName"
            :options="modelListOptions"
            @change="(v: any) => hqSettings.modelName = v"
          />
          <span class="model-count">共 {{ modelList.length }} 个模型</span>
        </div>
      </div>

      <!-- 测试连接按钮 -->
      <div class="settings-item">
        <button class="settings-test-btn" @click="testConnection" :disabled="isTesting">
          {{ isTesting ? '测试中...' : '🔗 测试连接' }}
        </button>
      </div>
    </div>

    <!-- 批处理设置 -->
    <div class="settings-group">
      <div class="settings-group-title">批处理设置</div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsHqBatchSize">批次大小:</label>
          <input type="number" id="settingsHqBatchSize" v-model.number="hqSettings.batchSize" min="1" max="10" step="1" />
          <div class="input-hint">每批处理的图片数量 (推荐3-5张)</div>
        </div>
        <div class="settings-item">
          <label for="settingsHqSessionReset">会话重置频率:</label>
          <input type="number" id="settingsHqSessionReset" v-model.number="hqSettings.sessionReset" min="1" step="1" />
          <div class="input-hint">多少批次后重置上下文</div>
        </div>
      </div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsHqRpmLimit">RPM限制:</label>
          <input type="number" id="settingsHqRpmLimit" v-model.number="hqSettings.rpmLimit" min="0" step="1" />
          <div class="input-hint">每分钟请求数，0表示无限制</div>
        </div>
        <div class="settings-item">
          <label for="settingsHqMaxRetries">重试次数:</label>
          <input type="number" id="settingsHqMaxRetries" v-model.number="hqSettings.maxRetries" min="0" max="10" step="1" />
        </div>
      </div>
    </div>

    <!-- 高级选项 -->
    <div class="settings-group">
      <div class="settings-group-title">高级选项</div>
      <div class="settings-row">
        <div class="settings-item">
          <label class="checkbox-label">
            <input type="checkbox" v-model="hqSettings.lowReasoning" />
            低推理模式
          </label>
          <div class="input-hint">减少模型推理深度，提高速度</div>
        </div>
        <div class="settings-item">
          <label for="settingsHqNoThinkingMethod">取消思考方法:</label>
          <CustomSelect
            :model-value="hqSettings.noThinkingMethod"
            :options="noThinkingMethodOptions"
            @change="(v: any) => hqSettings.noThinkingMethod = v"
          />
        </div>
      </div>
      <div class="settings-row">
        <div class="settings-item">
          <label class="checkbox-label">
            <input type="checkbox" v-model="hqSettings.forceJsonOutput" />
            强制JSON输出
          </label>
          <div class="input-hint">使用 response_format: json_object</div>
        </div>
        <div class="settings-item">
          <label class="checkbox-label">
            <input type="checkbox" v-model="hqSettings.useStream" />
            流式调用
          </label>
          <div class="input-hint">使用流式API调用</div>
        </div>
      </div>
    </div>

    <!-- 高质量翻译提示词 -->
    <div class="settings-group">
      <div class="settings-group-title">高质量翻译提示词</div>
      <div class="settings-item">
        <textarea id="settingsHqPrompt" v-model="hqSettings.prompt" rows="6" placeholder="高质量翻译提示词"></textarea>
        <!-- 快速选择提示词 -->
        <SavedPromptsPicker
          prompt-type="hq_translate"
          @select="handleHqPromptSelect"
        />
        <button class="btn btn-secondary btn-sm" @click="resetHqPrompt">重置为默认</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 高质量翻译设置组件
 * 管理高质量翻译服务配置
 */
import { ref, computed } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
import { DEFAULT_HQ_TRANSLATE_PROMPT } from '@/constants'
import CustomSelect from '@/components/common/CustomSelect.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'

/** 服务商选项 */
const providerOptions = [
  { label: 'SiliconFlow', value: 'siliconflow' },
  { label: 'DeepSeek', value: 'deepseek' },
  { label: '火山引擎', value: 'volcano' },
  { label: 'Google Gemini', value: 'gemini' },
  { label: '自定义 OpenAI 兼容服务', value: 'custom_openai' }
]

/** 取消思考方法选项 */
const noThinkingMethodOptions = [
  { label: 'Gemini风格 (reasoning_effort=low)', value: 'gemini' },
  { label: '火山引擎风格 (thinking=null)', value: 'volcano' }
]

// Store
const settingsStore = useSettingsStore()
// 获取高质量翻译设置的响应式引用
const hqSettings = computed(() => settingsStore.settings.hqTranslation)
const toast = useToast()

// 密码显示状态
const showApiKey = ref(false)

// 模型获取状态
const isFetchingModels = ref(false)
const modelList = ref<string[]>([])

// 测试状态
const isTesting = ref(false)

/** 模型列表选项（用于CustomSelect） */
const modelListOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  modelList.value.forEach(model => options.push({ label: model, value: model }))
  return options
})

// 处理服务商切换
function handleProviderChange() {
  // 清空模型列表
  modelList.value = []
  settingsStore.saveToStorage()
}

// 获取服务商显示名称（与原版一致）
function getProviderDisplayName(provider: string): string {
  const names: Record<string, string> = {
    'siliconflow': 'SiliconFlow',
    'deepseek': 'DeepSeek',
    'volcano': '火山引擎',
    'gemini': 'Google Gemini',
    'custom_openai': '自定义OpenAI'
  }
  return names[provider] || provider
}

// 获取模型列表（复刻原版 doFetchModels 逻辑）
async function fetchModels() {
  const provider = hqSettings.value.provider
  const apiKey = hqSettings.value.apiKey?.trim()
  const baseUrl = hqSettings.value.customBaseUrl?.trim()

  // 验证（与原版一致）
  if (!apiKey) {
    toast.warning('请先填写 API Key')
    return
  }

  // 检查是否支持模型获取
  const supportedProviders = ['siliconflow', 'deepseek', 'volcano', 'gemini', 'custom_openai']
  if (!supportedProviders.includes(provider)) {
    toast.warning(`${getProviderDisplayName(provider)} 不支持自动获取模型列表`)
    return
  }

  // 自定义服务需要 base_url
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

// 测试高质量翻译服务连接（复刻原版逻辑）
async function testConnection() {
  const provider = hqSettings.value.provider
  const apiKey = hqSettings.value.apiKey?.trim()
  const modelName = hqSettings.value.modelName?.trim()
  const baseUrl = hqSettings.value.customBaseUrl?.trim()

  // 验证必填字段
  if (!apiKey) {
    toast.warning('请先填写 API Key')
    return
  }

  if (!modelName) {
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
    const result = await configApi.testAiTranslateConnection({
      provider,
      apiKey,
      modelName,
      baseUrl
    })

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

// 重置高质量翻译提示词
function resetHqPrompt() {
  settingsStore.updateHqTranslation({ prompt: DEFAULT_HQ_TRANSLATE_PROMPT })
  toast.success('已重置为默认提示词')
}

// 处理高质量翻译提示词选择
function handleHqPromptSelect(content: string, name: string) {
  settingsStore.updateHqTranslation({ prompt: content })
  toast.success(`已应用提示词: ${name}`)
}
</script>

<style scoped>
.checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.checkbox-label input[type='checkbox'] {
  width: auto;
}

.btn-sm {
  padding: 4px 12px;
  font-size: 12px;
  margin-top: 8px;
}
</style>
