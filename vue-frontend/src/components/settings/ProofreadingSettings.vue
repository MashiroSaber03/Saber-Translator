<template>
  <div class="proofreading-settings">
    <!-- AI校对启用开关 -->
    <div class="settings-group">
      <div class="settings-group-title">AI校对设置</div>
      <div class="settings-item">
        <label class="checkbox-label">
          <input type="checkbox" v-model="isProofreadingEnabled" />
          启用AI校对
        </label>
        <div class="input-hint">翻译完成后自动进行AI校对</div>
      </div>
      <div class="settings-item">
        <label for="settingsProofreadingMaxRetries">全局重试次数:</label>
        <input
          type="number"
          id="settingsProofreadingMaxRetries"
          v-model.number="proofreadingMaxRetries"
          min="0"
          max="10"
          step="1"
        />
      </div>
    </div>

    <!-- 校对轮次配置 -->
    <div v-show="isProofreadingEnabled" class="settings-group">
      <div class="settings-group-title">
        校对轮次配置
        <button class="btn btn-secondary btn-sm" @click="addRound">+ 添加轮次</button>
      </div>

      <!-- 轮次列表 -->
      <div v-for="(round, index) in proofreadingRounds" :key="index" class="proofreading-round">
        <div class="round-header">
          <span class="round-title">轮次 {{ index + 1 }}: {{ round.name || '未命名' }}</span>
          <button class="btn btn-danger btn-sm" @click="removeRound(index)" :disabled="proofreadingRounds.length <= 1">
            删除
          </button>
        </div>

        <div class="round-content">
          <!-- 轮次名称 -->
          <div class="settings-item">
            <label>轮次名称:</label>
            <input type="text" v-model="round.name" placeholder="如: 第一轮校对" />
          </div>

          <!-- 服务商选择 -->
          <div class="settings-row">
            <div class="settings-item">
              <label>服务商:</label>
              <CustomSelect
                v-model="round.provider"
                :options="providerOptions"
              />
            </div>
            <div class="settings-item">
              <label>API Key:</label>
              <div class="password-input-wrapper">
                <input
                  :type="round.showApiKey ? 'text' : 'password'"
                  v-model="round.apiKey"
                  class="secure-input"
                  placeholder="请输入API Key"
                  autocomplete="off"
                />
                <button type="button" class="password-toggle-btn" tabindex="-1" @click="round.showApiKey = !round.showApiKey">
                  <span class="eye-icon" v-if="!round.showApiKey">👁</span>
                  <span class="eye-off-icon" v-else>👁‍🗨</span>
                </button>
              </div>
            </div>
          </div>

          <!-- 自定义Base URL -->
          <div v-show="providerRequiresBaseUrl(round.provider)" class="settings-item">
            <label>Base URL:</label>
            <input type="text" v-model="round.customBaseUrl" placeholder="例如: https://api.example.com/v1" />
          </div>

          <!-- 模型名称 -->
          <div class="settings-item">
            <label>模型名称:</label>
            <div class="model-input-with-fetch">
              <input type="text" v-model="round.modelName" placeholder="请输入模型名称" />
              <button
                type="button"
                class="fetch-models-btn"
                title="获取可用模型列表"
                @click="fetchRoundModels(index)"
                :disabled="roundFetchingStates[index]"
              >
                <span class="fetch-icon">🔍</span>
                <span class="fetch-text">{{ roundFetchingStates[index] ? '获取中...' : '获取模型' }}</span>
              </button>
            </div>
            <!-- 模型选择下拉框 -->
            <div v-if="roundModelLists[index] && roundModelLists[index].length > 0" class="model-select-container">
              <CustomSelect
                v-model="round.modelName"
                :options="getRoundModelOptions(index)"
              />
              <span class="model-count">共 {{ roundModelLists[index].length }} 个模型</span>
            </div>
          </div>

          <!-- 测试连接按钮 -->
          <div class="settings-item">
            <button 
              class="settings-test-btn" 
              @click="testRoundConnection(index)" 
              :disabled="roundTestingStates[index]"
            >
              {{ roundTestingStates[index] ? '测试中...' : '🔗 测试连接' }}
            </button>
          </div>

          <!-- 批处理设置 -->
          <div class="settings-row">
            <div class="settings-item">
              <label>批次大小:</label>
              <input type="number" v-model.number="round.batchSize" min="1" max="10" step="1" />
            </div>
            <div class="settings-item">
              <label>RPM限制:</label>
              <input type="number" v-model.number="round.rpmLimit" min="0" step="1" />
            </div>
          </div>

          <!-- 高级选项 -->
          <div class="settings-row">
            <div class="settings-item">
              <label class="checkbox-label">
                <input type="checkbox" v-model="round.forceJsonOutput" />
                强制JSON输出
              </label>
              <div class="input-hint">使用 response_format: json_object</div>
            </div>
            <div class="settings-item">
              <label class="checkbox-label">
                <input type="checkbox" v-model="round.useStream" />
                流式调用
              </label>
              <div class="input-hint">使用流式API调用，避免超时</div>
            </div>
          </div>

          <!-- 校对提示词 -->
          <div class="settings-item">
            <label>校对提示词:</label>
            <textarea v-model="round.prompt" rows="4" placeholder="校对提示词"></textarea>
            <!-- 快速选择提示词 -->
            <SavedPromptsPicker
              prompt-type="proofreading"
              @select="(content, name) => handleProofreadingPromptSelect(index, content, name)"
            />
            <button class="btn btn-secondary btn-sm" @click="resetRoundPrompt(index)">重置为默认</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * AI校对设置组件
 * 管理多轮AI校对配置
 */
import { ref, computed, watch } from 'vue'
import {
  getProviderOptionsForCapability,
  providerSupportsCapability,
  providerRequiresBaseUrl
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settingsStore'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
import { DEFAULT_PROOFREADING_PROMPT } from '@/constants'
import type { ProofreadingRound } from '@/types/settings'
import CustomSelect from '@/components/common/CustomSelect.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'

/** 服务商选项 */
const providerOptions = getProviderOptionsForCapability('hqTranslation')

// Store
const settingsStore = useSettingsStore()
const toast = useToast()

// ---- 新增状态变量 ----
// 用于存储每个轮次的加载状态（使用 Record 以映射索引）
const roundFetchingStates = ref<Record<number, boolean>>({})
const roundTestingStates = ref<Record<number, boolean>>({})
const roundModelLists = ref<Record<number, string[]>>({})

// 计算属性 - 访问校对设置
const proofreadingRounds = computed(() => settingsStore.settings.proofreading.rounds)
const proofreadingMaxRetries = computed({
  get: () => settingsStore.settings.proofreading.maxRetries,
  set: (val: number) => settingsStore.setProofreadingMaxRetries(val)
})
const isProofreadingEnabled = computed({
  get: () => settingsStore.settings.proofreading.enabled,
  set: (val: boolean) => settingsStore.setProofreadingEnabled(val)
})

// ============================================================
// Watch 同步：轮次设置变化时自动保存到 localStorage
// ============================================================
watch(
  () => settingsStore.settings.proofreading.rounds,
  () => {
    // 轮次内的任何字段变化时自动保存
    settingsStore.saveToStorage()
  },
  { deep: true }
)

// ---- 新增函数 ----

/** 获取轮次模型的选项列表 */
function getRoundModelOptions(index: number) {
  const models = roundModelLists.value[index] || []
  const options = [{ label: '-- 选择模型 --', value: '' }]
  models.forEach(m => options.push({ label: m, value: m }))
  return options
}

/** 获取轮次模型列表（复刻原版逻辑） */
async function fetchRoundModels(index: number) {
  const round = proofreadingRounds.value[index]
  if (!round) return

  const provider = round.provider
  const apiKey = round.apiKey?.trim()
  const baseUrl = round.customBaseUrl?.trim()

  if (!apiKey) {
    toast.warning('请先填写 API Key')
    return
  }

  // 检查支持性
  if (!providerSupportsCapability(provider, 'modelFetch')) {
    toast.warning('当前服务商不支持获取模型列表')
    return
  }

  roundFetchingStates.value[index] = true
  try {
    const result = await configApi.fetchModels(provider, apiKey, baseUrl)
    if (result.success && result.models && result.models.length > 0) {
      roundModelLists.value[index] = result.models.map(m => m.id)
      toast.success(`轮次 ${index + 1}: 获取到 ${result.models.length} 个模型`)
    } else {
      toast.warning(result.message || '未获取到可用模型')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '获取模型列表失败'
    toast.error(errorMessage)
  } finally {
    roundFetchingStates.value[index] = false
  }
}

/** 测试轮次连接（复刻原版逻辑） */
async function testRoundConnection(index: number) {
  const round = proofreadingRounds.value[index]
  if (!round) return

  const provider = round.provider
  const apiKey = round.apiKey?.trim()
  const modelName = round.modelName?.trim()
  const baseUrl = round.customBaseUrl?.trim()

  if (!apiKey) {
    toast.warning('请先填写 API Key')
    return
  }

  if (!modelName) {
    toast.warning('请填写模型名称')
    return
  }

  roundTestingStates.value[index] = true
  toast.info(`正在测试轮次 ${index + 1} 的连接...`)

  try {
    const result = await configApi.testAiTranslateConnection({
      provider,
      apiKey,
      modelName,
      baseUrl
    })

    if (result.success) {
      toast.success(result.message || '连接成功!')
    } else {
      toast.error(result.message || result.error || '连接失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '连接测试失败'
    toast.error(errorMessage)
  } finally {
    roundTestingStates.value[index] = false
  }
}

// ---- 原有函数 ----

// 添加校对轮次
function addRound() {
  const newRound: ProofreadingRound = {
    name: `第${proofreadingRounds.value.length + 1}轮校对`,
    provider: 'siliconflow',
    apiKey: '',
    modelName: '',
    customBaseUrl: '',
    batchSize: 3,
    rpmLimit: 7,
    forceJsonOutput: false,
    useStream: true,
    prompt: DEFAULT_PROOFREADING_PROMPT,
    showApiKey: false
  }
  settingsStore.addProofreadingRound(newRound)
  toast.success('已添加新的校对轮次')
}

// 删除校对轮次
function removeRound(index: number) {
  if (proofreadingRounds.value.length <= 1) {
    toast.warning('至少需要保留一个校对轮次')
    return
  }
  settingsStore.removeProofreadingRound(index)
  toast.success('已删除校对轮次')
}

// 重置轮次提示词
function resetRoundPrompt(index: number) {
  settingsStore.updateProofreadingRound(index, { prompt: DEFAULT_PROOFREADING_PROMPT })
  toast.success('已重置为默认提示词')
}

// 处理校对提示词选择
function handleProofreadingPromptSelect(index: number, content: string, name: string) {
  settingsStore.updateProofreadingRound(index, { prompt: content })
  toast.success(`已应用提示词: ${name}`)
}
</script>

<style scoped>
.proofreading-round {
  border: 1px solid var(--border-color);
  border-radius: 8px;
  margin-bottom: 15px;
  overflow: hidden;
}

.round-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 15px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-color);
}

.round-title {
  font-weight: 500;
}

.round-content {
  padding: 15px;
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

.btn-sm {
  padding: 4px 12px;
  font-size: 12px;
}

.btn-danger {
  background: var(--danger-color, #dc3545);
  color: white;
  border: none;
}

.btn-danger:hover {
  background: var(--danger-hover-color, #c82333);
}

.btn-danger:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.settings-group-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

/* 复刻原版模型输入样式 */
.model-input-with-fetch {
  display: flex;
  gap: 10px;
  align-items: center;
}

.model-input-with-fetch input {
  flex: 1;
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
  background-color: var(--color-primary);
  color: #fff;
  border-color: var(--color-primary);
}

.fetch-models-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.model-select-container {
  margin-top: 10px;
  padding: 12px;
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.model-count {
  font-size: 12px;
  color: var(--text-secondary);
  text-align: right;
  margin-top: 4px;
}

/* 密码输入框 */
.password-input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
  width: 100%;
}

.password-input-wrapper input {
  flex: 1;
  padding-right: 40px;
}

.password-toggle-btn {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  background: none;
  border: none;
  cursor: pointer;
  color: var(--text-secondary);
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 统一测试连接按钮样式 */
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
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.settings-test-btn:active:not(:disabled) {
  background-color: var(--primary-light, #e7f3ff);
}

.settings-test-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
