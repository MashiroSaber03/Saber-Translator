<template>
  <BaseModal
    v-model="isOpen"
    title="设置"
    size="large"
    custom-class="settings-modal-wrapper"
    mobile-presentation="fullscreen"
    width="90%"
    min-height="510px"
    max-width="900px"
    max-height="90vh"
    header-variant="brand"
    body-padding="none"
    body-display="flex"
    body-direction="column"
    :show-header="true"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="handleClose"
    @open="handleOpen"
  >
    <template #title>
      <span class="settings-modal__title">
        <UiIcon name="settings" />
        <span>设置</span>
      </span>
    </template>

    <ProductSegmentedTabs
      :tabs="tabs"
      :active-tab="activeTab"
      aria-label="设置分类"
      appearance="underline"
      class="settings-modal__tabs"
      @update:active-tab="setActiveTab"
    />

    <div class="settings-modal__tab-content">
      <div v-show="activeTab === 'ocr'" class="settings-modal__tab-pane">
        <OcrSettings />
      </div>

      <div v-show="activeTab === 'translate'" class="settings-modal__tab-pane">
        <TranslationSettings />
      </div>

      <div v-show="activeTab === 'detection'" class="settings-modal__tab-pane">
        <DetectionSettings />
      </div>

      <div v-show="activeTab === 'hq'" class="settings-modal__tab-pane">
        <HqTranslationSettings />
      </div>

      <div v-show="activeTab === 'proofreading'" class="settings-modal__tab-pane">
        <ProofreadingSettings />
      </div>

      <div v-show="activeTab === 'prompt-library'" class="settings-modal__tab-pane">
        <PromptLibrary />
      </div>

      <div v-show="activeTab === 'plugins'" class="settings-modal__tab-pane">
        <PluginManager />
      </div>

      <div v-show="activeTab === 'text-defaults'" class="settings-modal__tab-pane">
        <TextStyleDefaultsSettings
          :is-open="isOpen"
          :save-request-id="textDefaultsSaveRequestId"
          @save-complete="handleTextDefaultsSaveComplete"
        />
      </div>

      <div v-show="activeTab === 'more'" class="settings-modal__tab-pane">
        <MoreSettings />
      </div>
    </div>

    <template #footer>
      <ProductActionRow
        aria-label="应用设置操作"
        variant="dialog"
      >
        <UiButton variant="secondary" @click="handleClose">取消</UiButton>
        <UiButton variant="primary" @click="handleSave">保存设置</UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import OcrSettings from './OcrSettings.vue'
import TranslationSettings from './TranslationSettings.vue'
import DetectionSettings from './DetectionSettings.vue'
import HqTranslationSettings from './HqTranslationSettings.vue'
import ProofreadingSettings from './ProofreadingSettings.vue'
import PromptLibrary from './PromptLibrary.vue'
import PluginManager from './PluginManager.vue'
import MoreSettings from './MoreSettings.vue'
import TextStyleDefaultsSettings from './TextStyleDefaultsSettings.vue'
import { showToast } from '@/utils/toast'

interface SettingsModalSavePayload {
  textDefaultsChanged: boolean
}

interface TextDefaultsSaveResult {
  success: boolean
  changed: boolean
  error?: string
}

const props = defineProps<{
  modelValue: boolean
  initialTab?: string
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'save', payload: SettingsModalSavePayload): void
}>()

const settingsStore = useSettingsStore()

const isOpen = ref(props.modelValue)
type SettingsTabId =
  | 'ocr'
  | 'translate'
  | 'detection'
  | 'hq'
  | 'proofreading'
  | 'prompt-library'
  | 'plugins'
  | 'text-defaults'
  | 'more'

const activeTab = ref<SettingsTabId>('ocr')
const textDefaultsSaveRequestId = ref(0)
let textDefaultsSavePromise: Promise<TextDefaultsSaveResult> | null = null
let resolveTextDefaultsSave: ((result: TextDefaultsSaveResult) => void) | null = null

const tabs = [
  { id: 'ocr', label: 'OCR识别' },
  { id: 'translate', label: '翻译服务' },
  { id: 'detection', label: '检测设置' },
  { id: 'hq', label: '高质量翻译' },
  { id: 'proofreading', label: 'AI校对' },
  { id: 'prompt-library', label: '提示词管理' },
  { id: 'plugins', label: '插件管理' },
  { id: 'text-defaults', label: '文本默认值' },
  { id: 'more', label: '更多' }
] satisfies Array<{ id: SettingsTabId; label: string }>

function isSettingsTabId(value: string): value is SettingsTabId {
  return tabs.some(tab => tab.id === value)
}

function setActiveTab(tabId: string): void {
  if (isSettingsTabId(tabId)) {
    activeTab.value = tabId
  }
}

watch(
  () => props.modelValue,
  (newVal) => {
    isOpen.value = newVal
    if (newVal) {
      if (props.initialTab && isSettingsTabId(props.initialTab)) {
        activeTab.value = props.initialTab
      }
    } else {
      activeTab.value = 'ocr'
    }
  }
)

watch(isOpen, (newVal) => {
  if (!newVal && props.modelValue) {
    emit('update:modelValue', false)
  }
})

function handleOpen() {
  if (props.initialTab && isSettingsTabId(props.initialTab)) {
    activeTab.value = props.initialTab
  }
}

function handleClose() {
  isOpen.value = false
  emit('update:modelValue', false)
}

function requestTextDefaultsSave(): Promise<TextDefaultsSaveResult> {
  if (textDefaultsSavePromise) return textDefaultsSavePromise

  textDefaultsSavePromise = new Promise<TextDefaultsSaveResult>((resolve) => {
    resolveTextDefaultsSave = resolve
    textDefaultsSaveRequestId.value += 1
  })

  return textDefaultsSavePromise
}

function handleTextDefaultsSaveComplete(result: TextDefaultsSaveResult): void {
  resolveTextDefaultsSave?.(result)
  resolveTextDefaultsSave = null
  textDefaultsSavePromise = null
}

async function handleSave() {
  const textDefaultsResult = await requestTextDefaultsSave()

  if (!textDefaultsResult.success) {
    showToast(textDefaultsResult.error || '保存文本默认值失败', 'error')
    return
  }

  settingsStore.saveToStorage()

  try {
    await settingsStore.saveToBackend()
  } catch {
    showToast('设置已保存到本地，后端同步失败', 'warning')
  }

  emit('save', { textDefaultsChanged: textDefaultsResult.changed })
  handleClose()
}
</script>

<style scoped>
.settings-modal__title {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.settings-modal__tabs {
  flex: 0 0 auto;
  margin: 14px 15px 0;
}

.settings-modal__tab-content {
  flex: 1;
  overflow-y: auto;
  padding: 25px;
}

.settings-modal__tab-pane {
  display: block;
}

@media (--breakpoint-md-down) {
  .settings-modal__tabs {
    margin-inline: 10px;
  }

  .settings-modal__tab-content {
    padding: 15px;
  }
}
</style>
