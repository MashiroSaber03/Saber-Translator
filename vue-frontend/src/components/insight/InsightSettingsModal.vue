<script setup lang="ts">
import { computed, ref, onMounted, onBeforeUnmount } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import { useInsightStore } from '@/stores/insightStore'
import { useSettingsAutoSave } from '@/composables/useSettingsAutoSave'
import * as insightApi from '@/api/insight'

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

type InsightSettingsTabId =
  | 'vlm'
  | 'llm'
  | 'batch'
  | 'embedding'
  | 'reranker'
  | 'imagegen'
  | 'prompts'

const activeSettingsTab = ref<InsightSettingsTabId>('vlm')
const visitedSettingsTabs = ref<Set<InsightSettingsTabId>>(new Set(['vlm']))
const isLoadingConfig = ref(true)
const backendConfigReady = ref(false)
const closeSaveFailed = ref(false)
const testMessage = ref('')
const testMessageType = ref<'success' | 'error' | ''>('')
const messageTone = computed(() => (testMessageType.value === 'error' ? 'danger' : 'success'))
let messageTimer: ReturnType<typeof setTimeout> | null = null
let savedConfig: ReturnType<typeof insightStore.getConfigForApi> | null = null
let isMounted = true
const { isSaving, persistChanges } = useSettingsAutoSave({
  source: () => insightStore.getConfigForApi(),
  ready: () => backendConfigReady.value,
  save: async () => {
    if (!savedConfig) throw new Error('请先加载后端设置')
    savedConfig = await insightApi.saveGlobalConfig(insightStore.getConfigForApi(), savedConfig)
  },
  onError: error => showMessage('保存失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error'),
})

const settingsTabs = [
  { id: 'vlm', label: 'VLM 多模态', glyph: '🖼️' },
  { id: 'llm', label: 'LLM 对话', glyph: '💬' },
  { id: 'batch', label: '批量分析', glyph: '📊' },
  { id: 'embedding', label: '向量模型', glyph: '🔢' },
  { id: 'reranker', label: '重排序', glyph: '🔄' },
  { id: 'imagegen', label: '生图模型', glyph: '🎨' },
  { id: 'prompts', label: '提示词', glyph: '📝' },
] satisfies Array<{ id: InsightSettingsTabId; label: string; glyph: string }>

function settingsTabGlyph(tabId: string): string {
  return settingsTabs.find(tab => tab.id === tabId)?.glyph ?? ''
}

function isInsightSettingsTabId(value: string): value is InsightSettingsTabId {
  return settingsTabs.some(tab => tab.id === value)
}

function switchSettingsTab(tab: InsightSettingsTabId): void {
  activeSettingsTab.value = tab
  visitedSettingsTabs.value = new Set([...visitedSettingsTabs.value, tab])
  testMessage.value = ''
  testMessageType.value = ''
}

function updateSettingsTab(tabId: string): void {
  if (isInsightSettingsTabId(tabId)) {
    switchSettingsTab(tabId)
  }
}

function closeModal(): void {
  clearMessageTimer()
  backendConfigReady.value = false
  emit('close')
}

function hasVisitedSettingsTab(tab: InsightSettingsTabId): boolean {
  return visitedSettingsTabs.value.has(tab)
}

async function handleClose(): Promise<void> {
  if (isSaving.value) return
  if (!(await persistChanges())) {
    closeSaveFailed.value = true
    return
  }
  closeModal()
}

function clearMessageTimer(): void {
  if (messageTimer) {
    clearTimeout(messageTimer)
    messageTimer = null
  }
}

function showMessage(message: string, type: 'success' | 'error'): void {
  if (!isMounted) return
  clearMessageTimer()
  testMessage.value = message
  testMessageType.value = type
  messageTimer = setTimeout(() => {
    testMessage.value = ''
    testMessageType.value = ''
    messageTimer = null
  }, 3000)
}

onMounted(async () => {
  try {
    const loaded = await insightApi.getGlobalConfig()
    if (!isMounted) return
    insightStore.setConfigFromApi(loaded)
    savedConfig = loaded
    backendConfigReady.value = true
  } catch (error) {
    showMessage(error instanceof Error ? error.message : '加载后端配置失败', 'error')
  } finally {
    if (isMounted) isLoadingConfig.value = false
  }
})

onBeforeUnmount(() => {
  isMounted = false
  clearMessageTimer()
})
</script>

<template>
  <BaseModal title="漫画分析设置" size="large" custom-class="insight-settings-modal"
    :show-close-button="!isSaving" :close-on-overlay="!isSaving" :close-on-esc="!isSaving"
    @close="handleClose">
    <ProductStatusBanner
      v-if="backendConfigReady && testMessage"
      class="insight-settings-message"
      :tone="messageTone"
      aria-live="polite"
    >
      {{ testMessage }}
    </ProductStatusBanner>

    <p v-if="isLoadingConfig" class="insight-settings-loading">正在读取后端配置…</p>
    <ProductStatusBanner v-else-if="!backendConfigReady" tone="danger" title="设置加载失败">
      {{ testMessage || '请关闭并重新打开设置后再编辑。' }}
    </ProductStatusBanner>
    <fieldset v-else class="insight-settings-fields">
      <ProductSegmentedTabs
        :tabs="settingsTabs"
        :active-tab="activeSettingsTab"
        aria-label="漫画分析设置分类"
        class="insight-settings-tabs"
        @update:active-tab="updateSettingsTab"
      >
        <template #tabIcon="{ tab }">{{ settingsTabGlyph(tab.id) }}</template>
      </ProductSegmentedTabs>

      <VlmSettingsTab
        v-if="hasVisitedSettingsTab('vlm')"
        v-show="activeSettingsTab === 'vlm'"
        @update:config="insightStore.updateVlmConfig($event)"
        @show-message="showMessage"
      />

      <LlmSettingsTab
        v-if="hasVisitedSettingsTab('llm')"
        v-show="activeSettingsTab === 'llm'"
        @update:config="insightStore.updateLlmConfig($event)"
        @show-message="showMessage"
      />

      <BatchSettingsTab
        v-if="hasVisitedSettingsTab('batch')"
        v-show="activeSettingsTab === 'batch'"
        @update:config="insightStore.updateBatchConfig($event)"
      />

      <EmbeddingSettingsTab
        v-if="hasVisitedSettingsTab('embedding')"
        v-show="activeSettingsTab === 'embedding'"
        @update:config="insightStore.updateEmbeddingConfig($event)"
        @show-message="showMessage"
      />

      <RerankerSettingsTab
        v-if="hasVisitedSettingsTab('reranker')"
        v-show="activeSettingsTab === 'reranker'"
        @update:config="insightStore.updateRerankerConfig($event)"
        @show-message="showMessage"
      />

      <PromptsSettingsTab
        v-if="hasVisitedSettingsTab('prompts')"
        v-show="activeSettingsTab === 'prompts'"
        @update:prompts="insightStore.updatePrompts($event)"
        @show-message="showMessage"
      />

      <ImageGenSettingsTab
        v-if="hasVisitedSettingsTab('imagegen')"
        v-show="activeSettingsTab === 'imagegen'"
        @update:config="insightStore.updateImageGenConfig($event)"
        @show-message="showMessage"
      />
    </fieldset>

    <template #footer>
      <div class="insight-settings-footer">
      <ProductStatusBanner v-if="closeSaveFailed" tone="danger" role="alert">
        保存失败，部分修改尚未保存。可以继续编辑并重试；仍然关闭可能丢失未保存的修改。
      </ProductStatusBanner>
      <ProductActionRow aria-label="漫画分析设置操作" variant="dialog">
        <span class="insight-settings-save-status">{{ isSaving ? '正在保存…' : closeSaveFailed ? '上次关闭时保存失败' : '修改后自动保存' }}</span>
        <UiButton v-if="closeSaveFailed" variant="secondary" :disabled="isSaving" @click="closeModal">仍然关闭</UiButton>
        <UiButton variant="primary" :disabled="isSaving" @click="handleClose">完成</UiButton>
      </ProductActionRow>
      </div>
    </template>
  </BaseModal>
</template>

<style scoped>
.insight-settings-footer {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.insight-settings-save-status {
  margin-right: auto;
  color: var(--color-text-muted);
  font-size: var(--font-size-sm);
}

.insight-settings-tabs {
  --product-segmented-tabs-active-background: var(--color-surface-brand);
  --product-segmented-tabs-active-text: var(--color-text-inverse);
  --product-segmented-tabs-active-shadow: none;
  --product-segmented-tabs-background: transparent;
  --product-segmented-tabs-padding: 0 0 8px;
  --product-segmented-tabs-radius: 0;
  --product-segmented-tabs-tab-radius: 4px;

  margin-bottom: 16px;
  border-width: 0 0 1px;
}

.insight-settings-message {
  margin-bottom: 12px;
}

.insight-settings-fields {
  min-width: 0;
  margin: 0;
  padding: 0;
  border: 0;
}

.insight-settings-loading {
  margin: 0;
  padding: 48px 24px;
  color: var(--color-text-muted);
  text-align: center;
}
</style>
