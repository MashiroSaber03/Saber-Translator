<script setup lang="ts">
/**
 * 漫画分析设置模态框组件（重构版）
 * 配置VLM、LLM、Embedding、Reranker等模型参数
 * 
 * 子组件已拆分到 ./settings/ 目录
 */

import { ref, onMounted } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'

// 导入拆分的子组件
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
  emit('close')
}

/**
 * 显示消息（由子组件调用）
 */
function showMessage(message: string, type: 'success' | 'error'): void {
  testMessage.value = message
  testMessageType.value = type
  setTimeout(() => {
    testMessage.value = ''
    testMessageType.value = ''
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
      setTimeout(() => {
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
</script>

<template>
  <BaseModal title="漫画分析设置" size="large" custom-class="insight-settings-modal" @close="close">
    <!-- 选项卡导航 -->
    <div class="settings-tabs">
      <button 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'vlm' }"
        @click="switchSettingsTab('vlm')"
      >
        🖼️ VLM 多模态
      </button>
      <button 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'llm' }"
        @click="switchSettingsTab('llm')"
      >
        💬 LLM 对话
      </button>
      <button 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'batch' }"
        @click="switchSettingsTab('batch')"
      >
        📊 批量分析
      </button>
      <button 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'embedding' }"
        @click="switchSettingsTab('embedding')"
      >
        🔢 向量模型
      </button>
      <button 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'reranker' }"
        @click="switchSettingsTab('reranker')"
      >
        🔄 重排序
      </button>
      <button 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'imagegen' }"
        @click="switchSettingsTab('imagegen')"
      >
        🎨 生图模型
      </button>
      <button 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'prompts' }"
        @click="switchSettingsTab('prompts')"
      >
        📝 提示词
      </button>
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
      <button class="btn btn-secondary" @click="close">取消</button>
      <button class="btn btn-primary" :disabled="isSaving" @click="saveSettings">
        {{ isSaving ? '保存中...' : '保存' }}
      </button>
    </template>
  </BaseModal>
</template>

<style>
/* 
 * InsightSettingsModal 样式
 * 注意：不使用 scoped，因为 BaseModal 使用 Teleport 将内容传送到 body
 * 样式使用 .insight-settings- 前缀避免全局污染
 */

/* 表单基础样式 */
.insight-settings-modal .form-group {
  margin-bottom: 16px;
}

.insight-settings-modal .form-group label {
  display: block;
  margin-bottom: 6px;
  font-weight: 500;
  font-size: 14px;
  color: var(--text-primary, #333);
}

.insight-settings-modal .form-group input[type="text"],
.insight-settings-modal .form-group input[type="password"],
.insight-settings-modal .form-group input[type="number"],
.insight-settings-modal .form-group select,
.insight-settings-modal .form-group textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 6px;
  font-size: 14px;
  background: var(--input-bg-color, #fff);
  color: var(--text-primary, #333);
  transition: border-color 0.2s, box-shadow 0.2s;
}

.insight-settings-modal .form-group input:focus,
.insight-settings-modal .form-group select:focus,
.insight-settings-modal .form-group textarea:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
  box-shadow: 0 0 0 3px rgb(99, 102, 241, 0.1);
}

.insight-settings-modal .form-hint {
  margin-top: 4px;
  font-size: 12px;
  color: var(--text-secondary, #666);
}

.insight-settings-modal .checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-weight: normal;
}

.insight-settings-modal .checkbox-label input[type="checkbox"] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

/* 按钮样式 */
.insight-settings-modal .btn {
  padding: 10px 16px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.insight-settings-modal .btn-primary {
  background: var(--primary, #6366f1);
  color: white;
}

.insight-settings-modal .btn-primary:hover:not(:disabled) {
  background: var(--primary-dark, #4f46e5);
}

.insight-settings-modal .btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.insight-settings-modal .btn-secondary {
  background: var(--bg-secondary, #f3f4f6);
  color: var(--text-primary, #333);
  border: 1px solid var(--border-color, #e0e0e0);
}

.insight-settings-modal .btn-secondary:hover:not(:disabled) {
  background: var(--bg-hover, #e5e7eb);
}

.insight-settings-modal .settings-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--border-color, #e0e0e0);
  padding-bottom: 8px;
}

.insight-settings-modal .settings-tab {
  padding: 8px 12px;
  border: none;
  background: none;
  cursor: pointer;
  border-radius: 4px;
  transition: all 0.2s;
  font-size: 13px;
  color: var(--text-primary, #333);
}

.insight-settings-modal .settings-tab:hover {
  background: var(--bg-hover, #f3f4f6);
}

.insight-settings-modal .settings-tab.active {
  background: var(--primary, #6366f1);
  color: white;
}

.insight-settings-modal .insight-settings-content {
  padding: 16px 0;
  min-height: 300px;
}

.insight-settings-modal .settings-hint {
  color: var(--text-secondary, #666);
  font-size: 13px;
  margin-bottom: 16px;
  padding: 8px 12px;
  background: var(--bg-secondary, #f3f4f6);
  border-radius: 4px;
}

.insight-settings-modal .form-row {
  display: flex;
  gap: 16px;
}

.insight-settings-modal .form-row .form-group {
  flex: 1;
}

.insight-settings-modal .test-message {
  padding: 8px 12px;
  border-radius: 4px;
  margin-bottom: 12px;
  font-size: 13px;
}

.insight-settings-modal .test-message.success {
  background: var(--success-bg, #d4edda);
  color: var(--success-text, #155724);
  border: 1px solid var(--success-border, #c3e6cb);
}

.insight-settings-modal .test-message.error {
  background: var(--error-bg, #f8d7da);
  color: var(--error-text, #721c24);
  border: 1px solid var(--error-border, #f5c6cb);
}

.insight-settings-modal .placeholder-text {
  color: var(--text-secondary, #666);
  text-align: center;
  padding: 40px;
}

/* 提示词编辑器样式 */
.insight-settings-modal .prompts-settings {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.insight-settings-modal .prompt-editor {
  width: 100%;
  min-height: 200px;
  font-family: Consolas, Monaco, monospace;
  font-size: 13px;
  line-height: 1.5;
  padding: 12px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 4px;
  background: var(--bg-secondary, #f3f4f6);
  color: var(--text-primary, #333);
  resize: vertical;
}

.insight-settings-modal .prompt-editor:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.insight-settings-modal .prompt-actions-bar {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.insight-settings-modal .btn-sm {
  padding: 6px 12px;
  font-size: 13px;
}

.insight-settings-modal .section-divider {
  border: none;
  border-top: 1px solid var(--border-color, #e0e0e0);
  margin: 16px 0;
}

.insight-settings-modal .prompts-library-section {
  margin-top: 8px;
}

.insight-settings-modal .library-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.insight-settings-modal .library-header h4 {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
}

.insight-settings-modal .library-actions {
  display: flex;
  gap: 8px;
}

.insight-settings-modal .saved-prompts-list {
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 4px;
  background: var(--bg-secondary, #f3f4f6);
}

.insight-settings-modal .saved-prompt-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  cursor: pointer;
  border-bottom: 1px solid var(--border-color, #e0e0e0);
  transition: background 0.2s;
}

.insight-settings-modal .saved-prompt-item:last-child {
  border-bottom: none;
}

.insight-settings-modal .saved-prompt-item:hover {
  background: var(--bg-hover, #e5e7eb);
}

.insight-settings-modal .prompt-name {
  flex: 1;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.insight-settings-modal .prompt-type-badge {
  font-size: 11px;
  padding: 2px 6px;
  background: rgb(99, 102, 241, 0.1);
  color: var(--primary, #6366f1);
  border-radius: 4px;
  white-space: nowrap;
}

.insight-settings-modal .btn-icon-sm {
  padding: 2px 6px;
  background: none;
  border: none;
  cursor: pointer;
  opacity: 0.6;
  transition: opacity 0.2s;
}

.insight-settings-modal .btn-icon-sm:hover {
  opacity: 1;
}

.insight-settings-modal .loading-text {
  text-align: center;
  padding: 20px;
  color: var(--text-secondary, #666);
}

/* 架构预览样式 */
.insight-settings-modal .batch-info-box {
  margin-top: 16px;
  padding: 12px;
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 8px;
  border: 1px solid var(--border-color, #e0e0e0);
}

.insight-settings-modal .batch-info-box h4 {
  margin: 0 0 8px;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary, #333);
}

.insight-settings-modal .layers-preview-list {
  margin: 0;
  padding-left: 20px;
  font-size: 13px;
  line-height: 1.6;
}

.insight-settings-modal .layers-preview-list li {
  margin-bottom: 4px;
}

.insight-settings-modal .align-badge {
  color: var(--primary, #6366f1);
  font-size: 12px;
}

/* 当前配置信息 */
.insight-settings-modal .batch-estimate-box {
  margin-top: 12px;
  padding: 10px 12px;
  background: linear-gradient(135deg, rgb(99, 102, 241, 0.1), rgb(99, 102, 241, 0.05));
  border-radius: 6px;
  border: 1px solid rgb(99, 102, 241, 0.2);
}

.insight-settings-modal .batch-estimate-box p {
  margin: 0;
  font-size: 13px;
  color: var(--text-primary, #333);
}

.insight-settings-modal .batch-estimate-box strong {
  color: var(--primary, #6366f1);
}

/* 模型输入行样式 */
.insight-settings-modal .model-input-row {
  display: flex;
  gap: 8px;
  align-items: center;
}

.insight-settings-modal .model-input-row input {
  flex: 1;
}

.insight-settings-modal .fetch-btn {
  white-space: nowrap;
  flex-shrink: 0;
}

/* 模型下拉选择容器 */
.insight-settings-modal .model-select-container {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  padding: 8px 12px;
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 6px;
  border: 1px solid var(--border-color, #e0e0e0);
}

.insight-settings-modal .model-select {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 4px;
  font-size: 13px;
  background: var(--input-bg-color, #fff);
  color: var(--text-primary, #333);
  cursor: pointer;
}

.insight-settings-modal .model-select:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.insight-settings-modal .model-count {
  font-size: 12px;
  color: var(--text-secondary, #666);
  white-space: nowrap;
}
</style>
