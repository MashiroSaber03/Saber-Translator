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
          <div v-show="round.provider === 'custom_openai'" class="settings-item">
            <label>Base URL:</label>
            <input type="text" v-model="round.customBaseUrl" placeholder="例如: https://api.example.com/v1" />
          </div>

          <!-- 模型名称 -->
          <div class="settings-item">
            <label>模型名称:</label>
            <input type="text" v-model="round.modelName" placeholder="请输入模型名称" />
          </div>

          <!-- 批处理设置 -->
          <div class="settings-row">
            <div class="settings-item">
              <label>批次大小:</label>
              <input type="number" v-model.number="round.batchSize" min="1" max="10" step="1" />
            </div>
            <div class="settings-item">
              <label>会话重置频率:</label>
              <input type="number" v-model.number="round.sessionReset" min="1" step="1" />
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
                <input type="checkbox" v-model="round.lowReasoning" />
                低推理模式
              </label>
            </div>
            <div class="settings-item">
              <label>取消思考方法:</label>
              <CustomSelect
                v-model="round.noThinkingMethod"
                :options="noThinkingMethodOptions"
              />
            </div>
            <div class="settings-item">
              <label class="checkbox-label">
                <input type="checkbox" v-model="round.forceJsonOutput" />
                强制JSON输出
              </label>
            </div>
          </div>

          <!-- 校对提示词 -->
          <div class="settings-item">
            <label>校对提示词:</label>
            <textarea v-model="round.prompt" rows="4" placeholder="校对提示词"></textarea>
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
import { computed } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import { useToast } from '@/utils/toast'
import { DEFAULT_PROOFREADING_PROMPT } from '@/constants'
import type { ProofreadingRound } from '@/types/settings'
import CustomSelect from '@/components/common/CustomSelect.vue'

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
  { label: 'Gemini风格', value: 'gemini' },
  { label: '火山引擎风格', value: 'volcano' }
]

// Store
const settingsStore = useSettingsStore()
const toast = useToast()

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

// 添加校对轮次
function addRound() {
  const newRound: ProofreadingRound = {
    name: `第${proofreadingRounds.value.length + 1}轮校对`,
    provider: 'siliconflow',
    apiKey: '',
    modelName: '',
    customBaseUrl: '',
    batchSize: 3,
    sessionReset: 20,
    rpmLimit: 7,
    lowReasoning: false,
    noThinkingMethod: 'gemini',
    forceJsonOutput: true,
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
</style>
