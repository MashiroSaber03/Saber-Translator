<script setup lang="ts">
import { computed, ref, onMounted, onBeforeUnmount } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import type {
  BatchConfig,
  StoreEmbeddingConfig,
  StoreImageGenConfig,
  StoreLlmConfig,
  StoreOpenAICompatibleOptions,
  StoreRerankerConfig,
  StoreVlmConfig,
} from '@/types/insight'

import VlmSettingsTab from './settings/VlmSettingsTab.vue'
import LlmSettingsTab from './settings/LlmSettingsTab.vue'
import BatchSettingsTab from './settings/BatchSettingsTab.vue'
import EmbeddingSettingsTab from './settings/EmbeddingSettingsTab.vue'
import RerankerSettingsTab from './settings/RerankerSettingsTab.vue'
import PromptsSettingsTab from './settings/PromptsSettingsTab.vue'
import ImageGenSettingsTab from './settings/ImageGenSettingsTab.vue'

const emit = defineEmits<{
  (e: 'close'): void
}>()

const insightStore = useInsightStore()

type InsightSettingsTabId = 'vlm' | 'llm' | 'batch' | 'embedding' | 'reranker' | 'imagegen' | 'prompts'

const activeSettingsTab = ref<InsightSettingsTabId>('vlm')
const isSaving = ref(false)
const testMessage = ref('')
const testMessageType = ref<'success' | 'error' | ''>('')
const messageTone = computed(() => testMessageType.value === 'error' ? 'danger' : 'success')
let messageTimer: ReturnType<typeof setTimeout> | null = null
let closeTimer: ReturnType<typeof setTimeout> | null = null

const syncRequestId = ref(0)

function cloneOpenAIOptions(options: StoreOpenAICompatibleOptions): StoreOpenAICompatibleOptions {
  return {
    request: {
      ...options.request,
      extraBody: options.request.extraBody ? { ...options.request.extraBody } : undefined,
    },
    execution: { ...options.execution },
  }
}

function cloneVlmConfig(config: StoreVlmConfig): StoreVlmConfig {
  return {
    ...config,
    openaiOptions: cloneOpenAIOptions(config.openaiOptions),
  }
}

function cloneLlmConfig(config: StoreLlmConfig): StoreLlmConfig {
  return {
    ...config,
    openaiOptions: cloneOpenAIOptions(config.openaiOptions),
  }
}

function cloneBatchConfig(config: BatchConfig): BatchConfig {
  return {
    ...config,
    customLayers: config.customLayers.map(layer => ({ ...layer })),
  }
}

const vlmDraft = ref<StoreVlmConfig>(cloneVlmConfig(insightStore.config.vlm))
const llmDraft = ref<StoreLlmConfig>(cloneLlmConfig(insightStore.config.llm))
const batchDraft = ref<BatchConfig>(cloneBatchConfig(insightStore.config.batch))
const embeddingDraft = ref<StoreEmbeddingConfig>({ ...insightStore.config.embedding })
const rerankerDraft = ref<StoreRerankerConfig>({ ...insightStore.config.reranker })
const promptsDraft = ref<Record<string, string>>({ ...insightStore.config.prompts })
const imageGenDraft = ref<StoreImageGenConfig>({ ...insightStore.config.imageGen })

const settingsTabs = [
  { id: 'vlm', label: 'VLM 多模态', iconName: 'image' },
  { id: 'llm', label: 'LLM 对话', iconName: 'message' },
  { id: 'batch', label: '批量分析', iconName: 'bar-chart' },
  { id: 'embedding', label: '向量模型', iconName: 'target' },
  { id: 'reranker', label: '重排序', iconName: 'refresh' },
  { id: 'imagegen', label: '生图模型', iconName: 'palette' },
  { id: 'prompts', label: '提示词', iconName: 'file-text' },
] satisfies Array<{ id: InsightSettingsTabId; label: string; iconName: 'image' | 'message' | 'bar-chart' | 'target' | 'refresh' | 'palette' | 'file-text' }>

function isInsightSettingsTabId(value: string): value is InsightSettingsTabId {
  return settingsTabs.some(tab => tab.id === value)
}

function switchSettingsTab(tab: InsightSettingsTabId): void {
  activeSettingsTab.value = tab
  testMessage.value = ''
  testMessageType.value = ''
}

function updateSettingsTab(tabId: string): void {
  if (isInsightSettingsTabId(tabId)) {
    switchSettingsTab(tabId)
  }
}

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

function refreshDraftsFromStore(): void {
  vlmDraft.value = cloneVlmConfig(insightStore.config.vlm)
  llmDraft.value = cloneLlmConfig(insightStore.config.llm)
  batchDraft.value = cloneBatchConfig(insightStore.config.batch)
  embeddingDraft.value = { ...insightStore.config.embedding }
  rerankerDraft.value = { ...insightStore.config.reranker }
  promptsDraft.value = { ...insightStore.config.prompts }
  imageGenDraft.value = { ...insightStore.config.imageGen }
}

function applyDraftsToStore(): void {
  insightStore.updateVlmConfig(vlmDraft.value)
  insightStore.updateLlmConfig(llmDraft.value)
  insightStore.updateBatchConfig(batchDraft.value)
  insightStore.updateEmbeddingConfig(embeddingDraft.value)
  insightStore.updateRerankerConfig(rerankerDraft.value)
  insightStore.updatePrompts(promptsDraft.value)
  insightStore.updateImageGenConfig(imageGenDraft.value)
}

async function saveSettings(): Promise<void> {
  if (isSaving.value) return

  isSaving.value = true

  try {
    applyDraftsToStore()
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

async function loadConfig(): Promise<void> {
  try {
    insightStore.loadConfigFromStorage()

    const response = await insightApi.getGlobalConfig()
    if (response.success && response.config) {
      insightStore.setConfigFromApi(response.config as Record<string, unknown>)
    }

    requestTabsSyncFromStore()
  } catch {
    showMessage('加载配置失败，将使用本地配置', 'error')
    requestTabsSyncFromStore()
  }
}

function requestTabsSyncFromStore(): void {
  refreshDraftsFromStore()
  syncRequestId.value += 1
}

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
    <ProductSegmentedTabs
      :tabs="settingsTabs"
      :active-tab="activeSettingsTab"
      aria-label="漫画分析设置分类"
      class="insight-settings-tabs"
      @update:active-tab="updateSettingsTab"
    />

    <ProductStatusBanner
      v-if="testMessage"
      class="insight-settings-message"
      :tone="messageTone"
      aria-live="polite"
    >
      {{ testMessage }}
    </ProductStatusBanner>

    <VlmSettingsTab
      v-show="activeSettingsTab === 'vlm'"
      :sync-request-id="syncRequestId"
      @update:config="vlmDraft = $event"
      @show-message="showMessage"
    />

    <LlmSettingsTab
      v-show="activeSettingsTab === 'llm'"
      :sync-request-id="syncRequestId"
      @update:config="llmDraft = $event"
      @show-message="showMessage"
    />

    <BatchSettingsTab
      v-show="activeSettingsTab === 'batch'"
      :sync-request-id="syncRequestId"
      @update:config="batchDraft = $event"
    />

    <EmbeddingSettingsTab
      v-show="activeSettingsTab === 'embedding'"
      :sync-request-id="syncRequestId"
      @update:config="embeddingDraft = $event"
      @show-message="showMessage"
    />

    <RerankerSettingsTab
      v-show="activeSettingsTab === 'reranker'"
      :sync-request-id="syncRequestId"
      @update:config="rerankerDraft = $event"
      @show-message="showMessage"
    />

    <PromptsSettingsTab
      v-show="activeSettingsTab === 'prompts'"
      :sync-request-id="syncRequestId"
      @update:prompts="promptsDraft = $event"
      @show-message="showMessage"
    />

    <ImageGenSettingsTab
      v-show="activeSettingsTab === 'imagegen'"
      :sync-request-id="syncRequestId"
      @update:config="imageGenDraft = $event"
      @show-message="showMessage"
    />

    <template #footer>
      <ProductActionRow
        aria-label="漫画分析设置操作"
        variant="dialog"
      >
        <UiButton variant="secondary" @click="close">取消</UiButton>
        <UiButton variant="primary" :disabled="isSaving" @click="saveSettings">
          {{ isSaving ? '保存中...' : '保存' }}
        </UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<style scoped>
.insight-settings-tabs {
  margin-bottom: 16px;
}

.insight-settings-message {
  margin-bottom: 12px;
}
</style>
