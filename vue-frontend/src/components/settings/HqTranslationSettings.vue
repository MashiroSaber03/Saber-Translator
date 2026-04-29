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
            @change="(v: any) => handleProviderChange(v)"
          />
        </div>
        <div class="settings-item">
          <label for="settingsHqApiKey">API Key:</label>
          <div class="password-input-wrapper">
            <input
              :type="showApiKey ? 'text' : 'password'"
              id="settingsHqApiKey"
              v-model="localHqSettings.apiKey"
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
      <div v-show="providerRequiresBaseUrl(hqSettings.provider)" class="settings-item">
        <label for="settingsHqCustomBaseUrl">Base URL:</label>
        <input
          type="text"
          id="settingsHqCustomBaseUrl"
          v-model="localHqSettings.customBaseUrl"
          placeholder="例如: https://api.example.com/v1"
        />
      </div>

      <!-- 模型名称 -->
      <div class="settings-item">
        <label for="settingsHqModelName">模型名称:</label>
        <div class="model-input-with-fetch">
          <input type="text" id="settingsHqModelName" v-model="localHqSettings.modelName" placeholder="请输入模型名称" />
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
            v-model="localHqSettings.modelName"
            :options="modelListOptions"
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
          <input type="number" id="settingsHqBatchSize" v-model.number="localHqSettings.batchSize" min="1" max="10" step="1" />
          <div class="input-hint">每批处理的图片数量 (推荐3-5张)</div>
        </div>
      </div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsHqRpmLimit">RPM限制:</label>
          <input type="number" id="settingsHqRpmLimit" v-model.number="localHqSettings.rpmLimit" min="0" step="1" />
          <div class="input-hint">每分钟请求数，0表示无限制</div>
        </div>
        <div class="settings-item">
          <label for="settingsHqMaxRetries">重试次数:</label>
          <input type="number" id="settingsHqMaxRetries" v-model.number="localHqSettings.maxRetries" min="0" max="10" step="1" />
        </div>
      </div>
    </div>

    <!-- 高级选项 -->
    <div class="settings-group">
      <div class="settings-group-title">高级选项</div>
      <div class="settings-row">
        <div class="settings-item">
          <label class="checkbox-label">
            <input type="checkbox" v-model="localHqSettings.lowReasoning" />
            低推理模式
          </label>
          <div class="input-hint">减少模型推理深度，提高速度</div>
        </div>
        <div class="settings-item">
          <label for="settingsHqNoThinkingMethod">取消思考方法:</label>
          <CustomSelect
            v-model="localHqSettings.noThinkingMethod"
            :options="noThinkingMethodOptions"
          />
        </div>
      </div>
      <div class="settings-row">
        <div class="settings-item">
          <label class="checkbox-label">
            <input type="checkbox" v-model="localHqSettings.forceJsonOutput" />
            强制JSON输出
          </label>
          <div class="input-hint">使用 response_format: json_object</div>
        </div>
        <div class="settings-item">
          <label class="checkbox-label">
            <input type="checkbox" v-model="localHqSettings.useStream" />
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
        <textarea id="settingsHqPrompt" v-model="localHqSettings.prompt" rows="6" placeholder="高质量翻译提示词"></textarea>
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
import { ref, computed, watch } from 'vue'
import {
  getProviderDisplayName as getProviderDisplayNameFromManifest,
  getProviderOptionsForCapability,
  providerRequiresBaseUrl,
  providerSupportsCapability
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settingsStore'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
import { DEFAULT_HQ_TRANSLATE_PROMPT } from '@/constants'
import CustomSelect from '@/components/common/CustomSelect.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'

/** 服务商选项 */
const providerOptions = getProviderOptionsForCapability('hqTranslation')

/** 取消思考方法选项 */
const noThinkingMethodOptions = [
  { label: 'Gemini风格 (reasoning_effort=low)', value: 'gemini' },
  { label: '火山引擎风格 (thinking=null)', value: 'volcano' }
]

// Store
const settingsStore = useSettingsStore()
const toast = useToast()

// 获取高质量翻译设置的响应式引用（用于显示条件判断）
const hqSettings = computed(() => settingsStore.settings.hqTranslation)

// 本地设置状态（用于双向绑定，修改后自动同步到 store）
const localHqSettings = ref({
  apiKey: settingsStore.settings.hqTranslation.apiKey,
  modelName: settingsStore.settings.hqTranslation.modelName,
  customBaseUrl: settingsStore.settings.hqTranslation.customBaseUrl,
  batchSize: settingsStore.settings.hqTranslation.batchSize,
  rpmLimit: settingsStore.settings.hqTranslation.rpmLimit,
  maxRetries: settingsStore.settings.hqTranslation.maxRetries,
  lowReasoning: settingsStore.settings.hqTranslation.lowReasoning,
  noThinkingMethod: settingsStore.settings.hqTranslation.noThinkingMethod,
  forceJsonOutput: settingsStore.settings.hqTranslation.forceJsonOutput,
  useStream: settingsStore.settings.hqTranslation.useStream,
  prompt: settingsStore.settings.hqTranslation.prompt
})

// ============================================================
// Watch 同步：本地状态变化时自动保存到 store
// ============================================================
watch(() => localHqSettings.value.apiKey, (val) => {
  settingsStore.updateHqTranslation({ apiKey: val })
})
watch(() => localHqSettings.value.modelName, (val) => {
  settingsStore.updateHqTranslation({ modelName: val })
})
watch(() => localHqSettings.value.customBaseUrl, (val) => {
  settingsStore.updateHqTranslation({ customBaseUrl: val })
})
watch(() => localHqSettings.value.batchSize, (val) => {
  settingsStore.updateHqTranslation({ batchSize: val })
})
watch(() => localHqSettings.value.rpmLimit, (val) => {
  settingsStore.updateHqTranslation({ rpmLimit: val })
})
watch(() => localHqSettings.value.maxRetries, (val) => {
  settingsStore.updateHqTranslation({ maxRetries: val })
})
watch(() => localHqSettings.value.lowReasoning, (val) => {
  settingsStore.updateHqTranslation({ lowReasoning: val })
})
watch(() => localHqSettings.value.noThinkingMethod, (val) => {
  settingsStore.updateHqTranslation({ noThinkingMethod: val })
})
watch(() => localHqSettings.value.forceJsonOutput, (val) => {
  settingsStore.updateHqTranslation({ forceJsonOutput: val })
})
watch(() => localHqSettings.value.useStream, (val) => {
  settingsStore.updateHqTranslation({ useStream: val })
})
watch(() => localHqSettings.value.prompt, (val) => {
  settingsStore.updateHqTranslation({ prompt: val })
})

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

// 处理服务商切换（复刻原版逻辑：独立保存每个服务商的配置）
function handleProviderChange(newProvider: string) {
  // 使用 store 的方法切换服务商（会自动保存旧配置、恢复新配置）
  settingsStore.setHqProvider(newProvider as import('@/types/settings').HqTranslationProvider)
  // 清空模型列表
  modelList.value = []
  // 同步本地状态（服务商切换后 store 会恢复新服务商的配置）
  syncLocalHqSettings()
}

// 同步本地高质量翻译状态
function syncLocalHqSettings() {
  const hq = settingsStore.settings.hqTranslation
  localHqSettings.value.apiKey = hq.apiKey
  localHqSettings.value.modelName = hq.modelName
  localHqSettings.value.customBaseUrl = hq.customBaseUrl
  localHqSettings.value.batchSize = hq.batchSize
  localHqSettings.value.rpmLimit = hq.rpmLimit
  localHqSettings.value.maxRetries = hq.maxRetries
  localHqSettings.value.lowReasoning = hq.lowReasoning
  localHqSettings.value.noThinkingMethod = hq.noThinkingMethod
  localHqSettings.value.forceJsonOutput = hq.forceJsonOutput
  localHqSettings.value.useStream = hq.useStream
  localHqSettings.value.prompt = hq.prompt
}

// 获取服务商显示名称（与原版一致）
function getProviderDisplayName(provider: string): string {
  return getProviderDisplayNameFromManifest(provider)
}

// 获取模型列表（复刻原版 doFetchModels 逻辑）
async function fetchModels() {
  const provider = hqSettings.value.provider
  const apiKey = localHqSettings.value.apiKey?.trim()
  const baseUrl = localHqSettings.value.customBaseUrl?.trim()

  // 验证（与原版一致）
  if (!apiKey) {
    toast.warning('请先填写 API Key')
    return
  }

  // 检查是否支持模型获取
  if (!providerSupportsCapability(provider, 'modelFetch')) {
    toast.warning(`${getProviderDisplayName(provider)} 不支持自动获取模型列表`)
    return
  }

  // 自定义服务需要 base_url
  if (providerRequiresBaseUrl(provider) && !baseUrl) {
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
  const apiKey = localHqSettings.value.apiKey?.trim()
  const modelName = localHqSettings.value.modelName?.trim()
  const baseUrl = localHqSettings.value.customBaseUrl?.trim()

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
  if (providerRequiresBaseUrl(provider) && !baseUrl) {
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
  // 同步本地状态
  localHqSettings.value.prompt = DEFAULT_HQ_TRANSLATE_PROMPT
  toast.success('已重置为默认提示词')
}

// 处理高质量翻译提示词选择
function handleHqPromptSelect(content: string, name: string) {
  settingsStore.updateHqTranslation({ prompt: content })
  // 同步本地状态
  localHqSettings.value.prompt = content
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
