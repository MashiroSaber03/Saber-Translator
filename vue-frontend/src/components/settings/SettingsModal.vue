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
    :show-close-button="!isSaving"
    :close-on-overlay="!isSaving"
    :close-on-esc="!isSaving"
    @close="handleClose"
  >
    <template #title>
      <span class="settings-modal__title">
        <span aria-hidden="true">⚙️</span>
        <span>设置</span>
      </span>
    </template>

    <ProductStatusBanner
      v-if="!settingsStore.isBackendReady"
      class="settings-modal__restricted"
      tone="danger"
      role="alert"
      title="设置加载失败"
    >
      {{ settingsStore.backendError || '正在读取后端设置。加载成功前只能查看出厂默认值，不能保存或调用 Provider。' }}
    </ProductStatusBanner>

    <fieldset
      class="settings-modal__fieldset"
      :disabled="!settingsStore.isBackendReady || isSaving"
    >
      <ProductSegmentedTabs
        :tabs="tabs"
        :active-tab="activeTab"
        aria-label="设置分类"
        appearance="underline"
        class="settings-modal__tabs"
        @update:active-tab="setActiveTab"
      />

      <div v-if="contentReady" class="settings-modal__tab-content">
        <div v-if="hasVisitedTab('ocr')" v-show="activeTab === 'ocr'" class="settings-modal__tab-pane">
          <OcrSettings />
        </div>

        <div v-if="hasVisitedTab('translate')" v-show="activeTab === 'translate'" class="settings-modal__tab-pane">
          <TranslationSettings />
        </div>

        <div v-if="hasVisitedTab('detection')" v-show="activeTab === 'detection'" class="settings-modal__tab-pane">
          <DetectionSettings />
        </div>

        <div v-if="hasVisitedTab('hq')" v-show="activeTab === 'hq'" class="settings-modal__tab-pane">
          <HqTranslationSettings />
        </div>

        <div v-if="hasVisitedTab('proofreading')" v-show="activeTab === 'proofreading'" class="settings-modal__tab-pane">
          <ProofreadingSettings />
        </div>

        <div v-if="hasVisitedTab('prompt-library')" v-show="activeTab === 'prompt-library'" class="settings-modal__tab-pane">
          <PromptLibrary />
        </div>

        <div v-if="hasVisitedTab('plugins')" v-show="activeTab === 'plugins'" class="settings-modal__tab-pane">
          <PluginManager />
        </div>

        <div v-if="hasVisitedTab('text-defaults')" v-show="activeTab === 'text-defaults'" class="settings-modal__tab-pane">
          <TextStyleDefaultsSettings />
        </div>

        <div v-if="hasVisitedTab('more')" v-show="activeTab === 'more'" class="settings-modal__tab-pane">
          <MoreSettings />
        </div>
      </div>
      <p v-else class="settings-modal__loading">正在读取后端设置…</p>
    </fieldset>

    <template #footer>
      <ProductActionRow
        aria-label="设置状态"
        variant="dialog"
      >
        <span class="settings-modal__save-status">
          {{ isSaving ? '正在保存…' : '修改后自动保存' }}
        </span>
        <UiButton variant="primary" :disabled="isSaving" @click="handleClose">完成</UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
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

const props = defineProps<{
  modelValue: boolean
  initialTab?: string
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
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
const visitedTabs = ref<Set<SettingsTabId>>(new Set(['ocr']))
const contentReady = ref(false)
const isSaving = ref(false)
let openRequestId = 0
let autoSaveTimer: ReturnType<typeof setTimeout> | null = null
let savePromise: Promise<boolean> | null = null
let applyingPersistence = false
let hasUnsavedChanges = false

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
    visitedTabs.value = new Set([...visitedTabs.value, tabId])
  }
}

function hasVisitedTab(tabId: SettingsTabId): boolean {
  return visitedTabs.value.has(tabId)
}

watch(
  () => props.modelValue,
  (newVal) => {
    if (newVal) {
      isOpen.value = true
      if (props.initialTab && isSettingsTabId(props.initialTab)) {
        setActiveTab(props.initialTab)
      }
      void handleOpen()
    } else {
      void persistChanges().finally(() => closeModal(false))
    }
  },
  { immediate: true },
)

watch(
  () => [
    settingsStore.settings,
    settingsStore.textStyleDefaults,
    settingsStore.providerConfigs,
  ],
  () => {
    if (!isOpen.value || !contentReady.value || applyingPersistence) return
    hasUnsavedChanges = true
    scheduleAutoSave()
  },
  { deep: true },
)

async function handleOpen() {
  const requestId = ++openRequestId
  contentReady.value = false
  const openingTab = props.initialTab && isSettingsTabId(props.initialTab)
    ? props.initialTab
    : activeTab.value
  activeTab.value = openingTab
  visitedTabs.value = new Set([openingTab])
  await settingsStore.loadFromBackend()
  if (requestId !== openRequestId || !isOpen.value) return
  hasUnsavedChanges = false
  contentReady.value = true
  if (props.initialTab && isSettingsTabId(props.initialTab)) {
    setActiveTab(props.initialTab)
  }
}

function closeModal(notifyParent: boolean) {
  openRequestId += 1
  contentReady.value = false
  visitedTabs.value = new Set(['ocr'])
  isOpen.value = false
  hasUnsavedChanges = false
  if (notifyParent) emit('update:modelValue', false)
}

async function handleClose(): Promise<void> {
  if (!(await persistChanges())) return
  closeModal(true)
}

onBeforeUnmount(() => {
  openRequestId += 1
  if (autoSaveTimer !== null) clearTimeout(autoSaveTimer)
  void persistChanges()
  contentReady.value = false
})

function scheduleAutoSave(): void {
  if (autoSaveTimer !== null) clearTimeout(autoSaveTimer)
  autoSaveTimer = setTimeout(() => {
    autoSaveTimer = null
    void persistChanges()
  }, 450)
}

async function persistChanges(): Promise<boolean> {
  if (autoSaveTimer !== null) {
    clearTimeout(autoSaveTimer)
    autoSaveTimer = null
  }
  if (savePromise) return savePromise
  if (!contentReady.value || !hasUnsavedChanges) return true

  hasUnsavedChanges = false
  isSaving.value = true
  applyingPersistence = true
  savePromise = (async () => {
    try {
      const saved = await settingsStore.saveToBackend()
      await nextTick()
      if (!saved) {
        hasUnsavedChanges = true
        showToast(settingsStore.backendError || '设置自动保存失败', 'error')
      }
      return saved
    } catch (error) {
      hasUnsavedChanges = true
      showToast(error instanceof Error ? error.message : '设置自动保存失败', 'error')
      return false
    } finally {
      applyingPersistence = false
      isSaving.value = false
    }
  })()
  try {
    return await savePromise
  } finally {
    savePromise = null
  }
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

.settings-modal__restricted {
  margin: 14px 15px 0;
}

.settings-modal__save-status {
  margin-right: auto;
  color: var(--color-text-muted);
  font-size: var(--font-size-sm);
}

.settings-modal__loading {
  flex: 1;
  margin: 0;
  padding: 48px 24px;
  color: var(--color-text-muted);
  text-align: center;
}

.settings-modal__fieldset {
  display: contents;
  min-width: 0;
  margin: 0;
  padding: 0;
  border: 0;
}

.settings-modal__tab-content {
  --ui-number-field-width: 100%;
  --ui-number-field-input-width: 100%;
  --ui-number-field-text-align: left;

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
