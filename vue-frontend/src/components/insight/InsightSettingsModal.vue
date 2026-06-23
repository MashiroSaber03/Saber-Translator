<script setup lang="ts">
/**
 * 漫画分析设置模态框组件
 * 配置 VLM、LLM、Embedding、Reranker、生图和提示词参数
 */

import { ref, onMounted, onBeforeUnmount } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'

import VlmSettingsTab from './settings/VlmSettingsTab.vue'
import LlmSettingsTab from './settings/LlmSettingsTab.vue'
import BatchSettingsTab from './settings/BatchSettingsTab.vue'
import EmbeddingSettingsTab from './settings/EmbeddingSettingsTab.vue'
import RerankerSettingsTab from './settings/RerankerSettingsTab.vue'
import PromptsSettingsTab from './settings/PromptsSettingsTab.vue'
import ImageGenSettingsTab from './settings/ImageGenSettingsTab.vue'

// ============================================================
// 事件定义
// ============================================================

const emit = defineEmits<{
  (e: 'close'): void
}>()

// ============================================================
// Store
// ============================================================

const insightStore = useInsightStore()

// ============================================================
// 状态
// ============================================================

/** 当前设置选项卡 */
const activeSettingsTab = ref<'vlm' | 'llm' | 'batch' | 'embedding' | 'reranker' | 'imagegen' | 'prompts'>('vlm')

/** 是否正在保存 */
const isSaving = ref(false)

/** 测试结果消息 */
const testMessage = ref('')

/** 测试结果类型 */
const testMessageType = ref<'success' | 'error' | ''>('')
let messageTimer: ReturnType<typeof setTimeout> | null = null
let closeTimer: ReturnType<typeof setTimeout> | null = null

// ============================================================
// 子组件引用
// ============================================================

const vlmTabRef = ref<InstanceType<typeof VlmSettingsTab> | null>(null)
const llmTabRef = ref<InstanceType<typeof LlmSettingsTab> | null>(null)
const batchTabRef = ref<InstanceType<typeof BatchSettingsTab> | null>(null)
const embeddingTabRef = ref<InstanceType<typeof EmbeddingSettingsTab> | null>(null)
const rerankerTabRef = ref<InstanceType<typeof RerankerSettingsTab> | null>(null)
const promptsTabRef = ref<InstanceType<typeof PromptsSettingsTab> | null>(null)
const imageGenTabRef = ref<InstanceType<typeof ImageGenSettingsTab> | null>(null)

// ============================================================
// 方法
// ============================================================

/**
 * 切换设置选项卡
 */
function switchSettingsTab(tab: typeof activeSettingsTab.value): void {
  activeSettingsTab.value = tab
  testMessage.value = ''
  testMessageType.value = ''
}

/**
 * 关闭模态框
 */
function close(): void {
  clearMessageTimer()
  clearCloseTimer()
  emit('close')
}

function clearMessageTimer(): void {
  if (messageTimer) {
    clearTimeout(messageTimer)
    messageTimer = null
  }
}

function clearCloseTimer(): void {
  if (closeTimer) {
    clearTimeout(closeTimer)
    closeTimer = null
  }
}

/**
 * 显示消息（由子组件调用）
 */
function showMessage(message: string, type: 'success' | 'error'): void {
  clearMessageTimer()
  testMessage.value = message
  testMessageType.value = type
  messageTimer = setTimeout(() => {
    testMessage.value = ''
    testMessageType.value = ''
    messageTimer = null
  }, 3000)
}

/**
 * 保存设置到 Store 和后端
 */
async function saveSettings(): Promise<void> {
  if (isSaving.value) return
  
  isSaving.value = true
  
  try {
    // 从各子组件获取配置
    if (vlmTabRef.value) {
      insightStore.updateVlmConfig(vlmTabRef.value.getConfig())
    }
    
    if (llmTabRef.value) {
      insightStore.updateLlmConfig(llmTabRef.value.getConfig())
    }
    
    if (batchTabRef.value) {
      insightStore.updateBatchConfig(batchTabRef.value.getConfig())
    }
    
    if (embeddingTabRef.value) {
      insightStore.updateEmbeddingConfig(embeddingTabRef.value.getConfig())
    }
    
    if (rerankerTabRef.value) {
      insightStore.updateRerankerConfig(rerankerTabRef.value.getConfig())
    }
    
    if (promptsTabRef.value) {
      insightStore.updatePrompts(promptsTabRef.value.getCustomPrompts())
    }
    
    if (imageGenTabRef.value) {
      insightStore.updateImageGenConfig(imageGenTabRef.value.getConfig())
    }
    
    // 保存到后端
    const apiConfig = insightStore.getConfigForApi()
    const response = await insightApi.saveGlobalConfig(apiConfig as insightApi.AnalysisConfig)
    
    if (response.success) {
      showMessage('设置已保存', 'success')
      clearCloseTimer()
      closeTimer = setTimeout(() => {
        closeTimer = null
        close()
      }, 500)
    } else {
      showMessage('保存失败: ' + (response.error || '未知错误'), 'error')
    }
  } catch (error) {
    showMessage('保存失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isSaving.value = false
  }
}

/**
 * 加载配置
 */
async function loadConfig(): Promise<void> {
  try {
    // 先从 localStorage 加载
    insightStore.loadConfigFromStorage()
    
    // 尝试从后端加载
    const response = await insightApi.getGlobalConfig()
    if (response.success && response.config) {
      insightStore.setConfigFromApi(response.config as Record<string, unknown>)
    }
    
    // 同步到各子组件
    syncAllFromStore()
  } catch (error) {
    console.error('加载配置失败:', error)
    syncAllFromStore()
  }
}

/**
 * 同步所有子组件配置
 */
function syncAllFromStore(): void {
  vlmTabRef.value?.syncFromStore()
  llmTabRef.value?.syncFromStore()
  batchTabRef.value?.syncFromStore()
  embeddingTabRef.value?.syncFromStore()
  rerankerTabRef.value?.syncFromStore()
  promptsTabRef.value?.syncFromStore()
  imageGenTabRef.value?.syncFromStore()
}

// ============================================================
// 生命周期
// ============================================================

onMounted(async () => {
  await loadConfig()
})

onBeforeUnmount(() => {
  clearMessageTimer()
  clearCloseTimer()
})
</script>

<template>
  <BaseModal title="漫画分析设置" size="large" custom-class="insight-settings-modal" @close="close">
    <!-- 选项卡导航 -->
    <div class="settings-tabs">
      <UiButton
        variant="toolbar" 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'vlm' }"
        @click="switchSettingsTab('vlm')"
      >
        🖼️ VLM 多模态
      </UiButton>
      <UiButton
        variant="toolbar" 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'llm' }"
        @click="switchSettingsTab('llm')"
      >
        💬 LLM 对话
      </UiButton>
      <UiButton
        variant="toolbar" 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'batch' }"
        @click="switchSettingsTab('batch')"
      >
        📊 批量分析
      </UiButton>
      <UiButton
        variant="toolbar" 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'embedding' }"
        @click="switchSettingsTab('embedding')"
      >
        🔢 向量模型
      </UiButton>
      <UiButton
        variant="toolbar" 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'reranker' }"
        @click="switchSettingsTab('reranker')"
      >
        🔄 重排序
      </UiButton>
      <UiButton
        variant="toolbar" 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'imagegen' }"
        @click="switchSettingsTab('imagegen')"
      >
        🎨 生图模型
      </UiButton>
      <UiButton
        variant="toolbar" 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'prompts' }"
        @click="switchSettingsTab('prompts')"
      >
        📝 提示词
      </UiButton>
    </div>

    <!-- 测试结果消息 -->
    <div v-if="testMessage" class="test-message" :class="testMessageType">
      {{ testMessage }}
    </div>

    <!-- VLM 设置 -->
    <VlmSettingsTab 
      v-show="activeSettingsTab === 'vlm'" 
      ref="vlmTabRef"
      @show-message="showMessage"
    />

    <!-- LLM 设置 -->
    <LlmSettingsTab 
      v-show="activeSettingsTab === 'llm'" 
      ref="llmTabRef"
      @show-message="showMessage"
    />

    <!-- 批量分析设置 -->
    <BatchSettingsTab 
      v-show="activeSettingsTab === 'batch'" 
      ref="batchTabRef"
    />

    <!-- Embedding 设置 -->
    <EmbeddingSettingsTab 
      v-show="activeSettingsTab === 'embedding'" 
      ref="embeddingTabRef"
      @show-message="showMessage"
    />

    <!-- Reranker 设置 -->
    <RerankerSettingsTab 
      v-show="activeSettingsTab === 'reranker'" 
      ref="rerankerTabRef"
      @show-message="showMessage"
    />

    <!-- 提示词设置 -->
    <PromptsSettingsTab 
      v-show="activeSettingsTab === 'prompts'" 
      ref="promptsTabRef"
      @show-message="showMessage"
    />

    <!-- 生图模型设置 -->
    <ImageGenSettingsTab 
      v-show="activeSettingsTab === 'imagegen'" 
      ref="imageGenTabRef"
      @show-message="showMessage"
    />

    <!-- 底部按钮 -->
    <template #footer>
      <UiButton variant="secondary" @click="close">取消</UiButton>
      <UiButton variant="primary" :disabled="isSaving" @click="saveSettings">
        {{ isSaving ? '保存中...' : '保存' }}
      </UiButton>
    </template>
  </BaseModal>
</template>

<style scoped>
.settings-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--color-border-muted, var(--color-border-default));
  padding-bottom: 8px;
}

.settings-tab {
  padding: 8px 12px;
  border: none;
  background: none;
  cursor: pointer;
  border-radius: 4px;
  transition: all 0.2s;
  font-size: 13px;
  color: var(--color-text-default, var(--color-text-default));
}

.settings-tab:hover {
  background: var(--color-surface-muted);
}

.settings-tab.active {
  background: var(--color-surface-brand);
  color: white;
}

.test-message {
  padding: 8px 12px;
  border-radius: 4px;
  margin-bottom: 12px;
  font-size: 13px;
}

.test-message.success {
  background: var(--insight-settings-modal-surface-base);
  color: var(--insight-settings-modal-text-primary);
  border: 1px solid var(--insight-settings-modal-border-default);
}

.test-message.error {
  background: var(--insight-settings-modal-surface-raised);
  color: var(--insight-settings-modal-text-secondary);
  border: 1px solid var(--insight-settings-modal-border-strong);
}
</style>
