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

    <ProductStatusBanner
      v-if="!settingsStore.isBackendReady"
      class="settings-modal__restricted"
      tone="danger"
      role="alert"
      title="设置受限模式"
    >
      {{ settingsStore.backendError || '正在读取后端设置。加载成功前只能查看出厂默认值，不能保存或调用 Provider。' }}
    </ProductStatusBanner>

    <fieldset
      class="settings-modal__fieldset"
      :disabled="!settingsStore.isBackendReady"
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
          <TextStyleDefaultsSettings :is-open="isOpen" />
        </div>

        <div v-show="activeTab === 'more'" class="settings-modal__tab-pane">
          <MoreSettings />
        </div>
      </div>
      <p v-else class="settings-modal__loading">正在读取后端设置…</p>
    </fieldset>

    <template #footer>
      <ProductActionRow
        aria-label="应用设置操作"
        variant="dialog"
      >
        <UiButton variant="secondary" @click="handleClose">取消</UiButton>
        <UiButton
          variant="primary"
          :disabled="!contentReady || !settingsStore.isBackendReady"
          @click="handleSave"
        >
          保存设置
        </UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import type { ProviderConfigsCache } from '@/stores/settings'
import type {
  TextStyleSettings,
  TranslationSettings as TranslationSettingsModel,
} from '@/types/settings'
import { deepClone } from '@/utils/deepClone'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
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

interface SettingsModalSavePayload {
  textDefaultsChanged: boolean
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
const contentReady = ref(false)
let settingsSnapshot: TranslationSettingsModel | null = null
let textStyleDefaultsSnapshot: TextStyleSettings | null = null
let providerSnapshot: ProviderConfigsCache | null = null
let closeAfterSave = false
let openRequestId = 0

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

async function handleOpen() {
  const requestId = ++openRequestId
  contentReady.value = false
  await settingsStore.loadFromBackend()
  if (requestId !== openRequestId || !isOpen.value) return
  settingsSnapshot = deepClone(settingsStore.settings)
  textStyleDefaultsSnapshot = deepClone(settingsStore.textStyleDefaults)
  providerSnapshot = deepClone(settingsStore.providerConfigs)
  contentReady.value = true
  if (props.initialTab && isSettingsTabId(props.initialTab)) {
    activeTab.value = props.initialTab
  }
}

function handleClose() {
  openRequestId += 1
  if (
    !closeAfterSave
    && settingsSnapshot
    && textStyleDefaultsSnapshot
    && providerSnapshot
  ) {
    settingsStore.settings = deepClone(settingsSnapshot)
    settingsStore.textStyleDefaults = deepClone(textStyleDefaultsSnapshot)
    settingsStore.providerConfigs = deepClone(providerSnapshot)
  }
  closeAfterSave = false
  settingsSnapshot = null
  textStyleDefaultsSnapshot = null
  providerSnapshot = null
  contentReady.value = false
  isOpen.value = false
  emit('update:modelValue', false)
}

onBeforeUnmount(() => {
  openRequestId += 1
  if (
    !closeAfterSave
    && settingsSnapshot
    && textStyleDefaultsSnapshot
    && providerSnapshot
  ) {
    settingsStore.settings = deepClone(settingsSnapshot)
    settingsStore.textStyleDefaults = deepClone(textStyleDefaultsSnapshot)
    settingsStore.providerConfigs = deepClone(providerSnapshot)
  }
  settingsSnapshot = null
  textStyleDefaultsSnapshot = null
  providerSnapshot = null
  contentReady.value = false
})

async function handleSave() {
  const textDefaultsChanged = Boolean(
    textStyleDefaultsSnapshot
    && JSON.stringify(textStyleDefaultsSnapshot)
      !== JSON.stringify(settingsStore.textStyleDefaults),
  )

  const saved = await settingsStore.saveToBackend()
  if (!saved) {
    showToast(settingsStore.backendError || '设置保存失败', 'error')
    return
  }

  emit('save', { textDefaultsChanged })
  closeAfterSave = true
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

.settings-modal__restricted {
  margin: 14px 15px 0;
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
