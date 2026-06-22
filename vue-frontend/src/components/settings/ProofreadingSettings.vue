<template>
  <div class="proofreading-settings">
    <!-- AI校对启用开关 -->
    <UiPanel variant="settings">
      <template #title>AI校对设置</template>
      <UiField class="ui-settings-field">
        <label class="ui-checkbox-label">
          <UiInput type="checkbox" v-model="isProofreadingEnabled" />
          启用AI校对
        </label>
        <div class="ui-form-hint">翻译完成后自动进行AI校对</div>
      </UiField>
      <UiField class="ui-settings-field">
        <label for="settingsProofreadingMaxRetries">全局重试次数:</label>
        <UiInput
          type="number"
          id="settingsProofreadingMaxRetries"
          v-model.number="proofreadingMaxRetries"
          min="0"
          max="10"
          step="1"
        />
      </UiField>
    </UiPanel>

    <!-- 校对轮次配置 -->
    <UiPanel variant="settings" v-show="isProofreadingEnabled">
      <template #title>
        校对轮次配置
        <UiButton variant="secondary" @click="addRound" size="sm">+ 添加轮次</UiButton>
      </template>

      <!-- 轮次列表 -->
      <div v-for="(round, index) in proofreadingRounds" :key="index" class="proofreading-round">
        <div class="round-header">
          <span class="round-title">轮次 {{ index + 1 }}: {{ round.name || '未命名' }}</span>
          <UiButton
            variant="danger"
            class="proofreading-round__delete-btn"
            @click="removeRound(index)"
            :disabled="proofreadingRounds.length <= 1"
            size="sm"
          >
            删除
          </UiButton>
        </div>

        <div class="round-content">
          <!-- 轮次名称 -->
          <UiField class="ui-settings-field">
            <label>轮次名称:</label>
            <UiInput type="text" v-model="round.name" placeholder="如: 第一轮校对" />
          </UiField>

          <!-- 服务商选择 -->
          <div class="ui-settings-row">
            <UiField class="ui-settings-field">
              <label>服务商:</label>
              <CustomSelect
                v-model="round.provider"
                :options="providerOptions"
              />
            </UiField>
            <UiField v-show="providerRequiresApiKey(round.provider)" class="ui-settings-field">
              <label>API Key:</label>
              <div class="password-input-wrapper">
                <UiInput
                  :type="round.showApiKey ? 'text' : 'password'"
                  v-model="round.apiKey"
                  class="secure-input"
                  placeholder="请输入API Key"
                  autocomplete="off"
                />
                <UiButton variant="toolbar" type="button" class="password-toggle-btn" tabindex="-1" @click="round.showApiKey = !round.showApiKey">
                  <span class="eye-icon" v-if="!round.showApiKey">👁</span>
                  <span class="eye-off-icon" v-else>👁‍🗨</span>
                </UiButton>
              </div>
            </UiField>
          </div>

          <!-- 自定义Base URL -->
          <UiField v-show="providerRequiresBaseUrl(round.provider)" class="ui-settings-field">
            <label>Base URL:</label>
            <UiInput type="text" v-model="round.customBaseUrl" placeholder="例如: https://api.example.com/v1" />
          </UiField>

          <!-- 模型名称 -->
          <UiField class="ui-settings-field">
            <label>模型名称:</label>
            <div class="model-input-with-fetch">
              <UiInput
                type="text"
                v-model="round.modelName"
                class="proofreading-settings__model-input"
                placeholder="请输入模型名称"
              />
              <UiButton
                variant="toolbar"
                type="button"
                class="fetch-models-btn"
                title="获取可用模型列表"
                @click="fetchRoundModels(index)"
                :disabled="roundFetchingStates[index]"
              >
                <span class="fetch-icon">🔍</span>
                <span class="fetch-text">{{ roundFetchingStates[index] ? '获取中...' : '获取模型' }}</span>
              </UiButton>
            </div>
            <!-- 模型选择下拉框 -->
            <div v-if="roundModelLists[index] && roundModelLists[index].length > 0" class="model-select-container">
              <CustomSelect
                v-model="round.modelName"
                :options="getRoundModelOptions(index)"
              />
              <span class="model-count">共 {{ roundModelLists[index].length }} 个模型</span>
            </div>
          </UiField>

          <!-- 测试连接按钮 -->
          <UiField class="ui-settings-field">
            <UiButton
              variant="toolbar" 
              class="settings-test-btn" 
              @click="testRoundConnection(index)" 
              :disabled="roundTestingStates[index]"
            >
              {{ roundTestingStates[index] ? '测试中...' : '🔗 测试连接' }}
            </UiButton>
          </UiField>

          <!-- 批处理设置 -->
          <div class="ui-settings-row">
            <UiField class="ui-settings-field">
              <label>批次大小:</label>
              <UiInput type="number" v-model.number="round.batchSize" min="1" max="10" step="1" />
            </UiField>
            <UiField class="ui-settings-field">
              <label>RPM限制:</label>
              <UiInput type="number" v-model.number="round.openaiOptions.execution.rpmLimit" min="0" step="1" />
            </UiField>
          </div>

          <!-- 高级选项 -->
          <div class="ui-settings-row">
            <UiField class="ui-settings-field">
              <label>业务重试:</label>
              <UiInput type="number" v-model.number="round.openaiOptions.execution.businessRetries" min="0" max="10" step="1" />
            </UiField>
            <UiField class="ui-settings-field">
              <label>传输重试:</label>
              <UiInput type="number" v-model.number="round.openaiOptions.execution.transportRetries" min="0" max="10" step="1" />
            </UiField>
          </div>
          <div class="ui-settings-row">
            <UiField class="ui-settings-field">
              <label class="ui-checkbox-label">
                <UiInput type="checkbox" v-model="round.openaiOptions.request.forceJsonOutput" />
                强制JSON输出
              </label>
              <div class="ui-form-hint">使用 response_format: json_object</div>
            </UiField>
            <UiField class="ui-settings-field">
              <label class="ui-checkbox-label">
                <UiInput type="checkbox" v-model="round.openaiOptions.execution.useStream" />
                流式调用
              </label>
              <div class="ui-form-hint">使用流式API调用，避免超时</div>
            </UiField>
          </div>
          <UiField class="ui-settings-field">
            <OpenAIExtraBodyEditor v-model="round.openaiOptions.request.extraBody" />
          </UiField>

          <!-- 校对提示词 -->
          <UiField class="ui-settings-field">
            <label>校对提示词:</label>
            <UiTextarea v-model="round.prompt" rows="4" placeholder="校对提示词" />
            <!-- 快速选择提示词 -->
            <SavedPromptsPicker
              prompt-type="proofreading"
              @select="(content, name) => handleProofreadingPromptSelect(index, content, name)"
            />
            <UiButton variant="secondary" @click="resetRoundPrompt(index)" size="sm">重置为默认</UiButton>
          </UiField>
        </div>
      </div>
    </UiPanel>
  </div>
</template>

<script setup lang="ts">

import UiField from '@/components/ui/UiField.vue'
import UiPanel from '@/components/ui/UiPanel.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
/**
 * AI校对设置组件
 * 管理多轮AI校对配置
 */
import { ref, computed, watch } from 'vue'
import {
  getProviderOptionsForCapability,
  providerRequiresApiKey,
  providerSupportsCapability,
  providerRequiresBaseUrl
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settings'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
import { DEFAULT_PROOFREADING_PROMPT } from '@/constants'
import type { ProofreadingRound } from '@/types/settings'
import CustomSelect from '@/components/common/CustomSelect.vue'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
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

/** 获取轮次模型列表（业务逻辑） */
async function fetchRoundModels(index: number) {
  const round = proofreadingRounds.value[index]
  if (!round) return

  const provider = round.provider
  const apiKey = round.apiKey?.trim()
  const baseUrl = round.customBaseUrl?.trim()

  if (providerRequiresApiKey(provider) && !apiKey) {
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

/** 测试轮次连接（业务逻辑） */
async function testRoundConnection(index: number) {
  const round = proofreadingRounds.value[index]
  if (!round) return

  const provider = round.provider
  const apiKey = round.apiKey?.trim()
  const modelName = round.modelName?.trim()
  const baseUrl = round.customBaseUrl?.trim()

  if (providerRequiresApiKey(provider) && !apiKey) {
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
    openaiOptions: {
      request: {
        forceJsonOutput: false
      },
      execution: {
        useStream: true,
        rpmLimit: 7,
        transportRetries: 1,
        businessRetries: settingsStore.settings.proofreading.maxRetries
      }
    },
    batchSize: 3,
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
.proofreading-settings {
  --ui-button-sm-padding: 4px 12px;
  --ui-button-sm-font-size: 12px;
  --ui-button-danger-background: var(--proofreading-settings-surface-base);
  --ui-button-danger-color: white;
  --ui-button-danger-border: none;
  --ui-button-danger-shadow: none;
  --ui-button-danger-hover-background: var(--proofreading-settings-surface-raised);
  --ui-button-danger-hover-shadow: none;
  --ui-button-disabled-opacity: 0.5;
}

.proofreading-round {
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  margin-bottom: 15px;
  overflow: hidden;
}

.round-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 15px;
  background: var(--color-surface-subtle);
  border-bottom: 1px solid var(--color-border-muted);
}

.round-title {
  font-weight: 500;
}

.round-content {
  padding: 15px;
}

.ui-checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.ui-checkbox-label input[type='checkbox'] {
  width: auto;
}

.model-input-with-fetch {
  display: flex;
  gap: 10px;
  align-items: center;
}

.model-input-with-fetch .proofreading-settings__model-input {
  flex: 1;
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

.fetch-models-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.fetch-models-btn:hover:not(:disabled) {
  background-color: var(--color-action-primary);
  color: var(--color-text-inverse);
  border-color: var(--color-action-primary);
}

.model-select-container {
  margin-top: 10px;
  padding: 12px;
  background-color: var(--color-surface-subtle);
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.model-count {
  font-size: 12px;
  color: var(--color-text-supporting);
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

.password-input-wrapper .secure-input {
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
  color: var(--color-text-supporting);
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 统一测试连接按钮样式 */
.settings-test-btn {
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

.settings-test-btn:active:not(:disabled) {
  background-color: var(--proofreading-settings-surface-muted);
}
</style>
