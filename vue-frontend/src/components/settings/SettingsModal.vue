<template>
  <!-- 设置模态框 -->
  <BaseModal
    v-model="isOpen"
    title="⚙️ 设置"
    size="large"
    custom-class="settings-modal-wrapper"
    width="90%"
    min-height="510px"
    max-width="900px"
    max-height="90vh"
    header-background="linear-gradient(135deg, var(--color-action-primary) 0%, var(--color-action-primary-hover) 100%)"
    header-color="var(--color-text-inverse)"
    header-padding="20px 25px"
    title-color="var(--color-text-inverse)"
    title-font-size="1.4em"
    close-color="var(--settings-modal-text-primary)"
    close-font-size="20px"
    close-hover-color="var(--color-text-inverse)"
    close-hover-background="var(--settings-modal-surface-base)"
    body-padding="none"
    body-display="flex"
    body-direction="column"
    :show-header="true"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="handleClose"
    @open="handleOpen"
  >
    <!-- 自定义头部 -->
    <template #title>
      <span>⚙️ 设置</span>
    </template>

    <!-- Tab 导航 + 内容区域（都放在 body slot 中） -->
    <!-- Tab 导航 -->
    <div class="settings-tabs">
      <UiButton
        variant="toolbar"
        v-for="tab in tabs"
        :key="tab.id"
        class="settings-tab"
        :class="{ active: activeTab === tab.id }"
        @click="activeTab = tab.id"
      >
        {{ tab.label }}
      </UiButton>
    </div>

    <!-- Tab 内容 -->
    <div class="settings-tab-content">
      <!-- OCR设置 -->
      <div v-show="activeTab === 'ocr'" class="settings-tab-pane active">
        <OcrSettings />
      </div>

      <!-- 翻译服务设置 -->
      <div v-show="activeTab === 'translate'" class="settings-tab-pane">
        <TranslationSettings />
      </div>

      <!-- 检测设置 -->
      <div v-show="activeTab === 'detection'" class="settings-tab-pane">
        <DetectionSettings />
      </div>

      <!-- 高质量翻译设置 -->
      <div v-show="activeTab === 'hq'" class="settings-tab-pane">
        <HqTranslationSettings />
      </div>

      <!-- AI校对设置 -->
      <div v-show="activeTab === 'proofreading'" class="settings-tab-pane">
        <ProofreadingSettings />
      </div>

      <!-- 提示词管理 -->
      <div v-show="activeTab === 'prompt-library'" class="settings-tab-pane">
        <PromptLibrary />
      </div>

      <!-- 插件管理 -->
      <div v-show="activeTab === 'plugins'" class="settings-tab-pane">
        <PluginManager />
      </div>

      <!-- 文本默认值 -->
      <div v-show="activeTab === 'text-defaults'" class="settings-tab-pane">
        <TextStyleDefaultsSettings ref="textStyleDefaultsRef" :is-open="isOpen" />
      </div>

      <!-- 更多设置 -->
      <div v-show="activeTab === 'more'" class="settings-tab-pane">
        <MoreSettings />
      </div>
    </div>

    <!-- 底部按钮 -->
    <template #footer>
      <UiButton variant="secondary" @click="handleClose">取消</UiButton>
      <UiButton variant="primary" @click="handleSave">保存设置</UiButton>
    </template>
  </BaseModal>
</template>

<script setup lang="ts">
import './SettingsModal.global.styles.css'
/**
 * 设置模态框组件
 * 管理所有一次性配置的集中设置界面
 * 基于 BaseModal 实现
 */
import { ref, watch } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
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

interface TextStyleDefaultsSettingsExposed {
  saveDefaults: () => Promise<{ success: boolean; changed: boolean; error?: string }>
}

// Props
const props = defineProps<{
  modelValue: boolean
  /** 初始Tab（可选），用于直接打开到指定Tab，如'plugins' */
  initialTab?: string
}>()

// Emits
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'save', payload: SettingsModalSavePayload): void
}>()

// Store
const settingsStore = useSettingsStore()

// 本地状态
const isOpen = ref(props.modelValue)
const activeTab = ref('ocr')
const textStyleDefaultsRef = ref<TextStyleDefaultsSettingsExposed | null>(null)

// Tab 配置
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
]

// 监听 props 变化
watch(
  () => props.modelValue,
  (newVal) => {
    isOpen.value = newVal
    if (newVal) {
      // 弹窗打开时尊重调用方指定的初始 tab。
      if (props.initialTab && tabs.some(t => t.id === props.initialTab)) {
        activeTab.value = props.initialTab
      }
    } else {
      // 关闭时重置为默认Tab，以便下次打开时可以正常响应initialTab
      activeTab.value = 'ocr'
    }
  }
)

// 监听本地状态变化同步到 props
watch(isOpen, (newVal) => {
  if (!newVal && props.modelValue) {
    emit('update:modelValue', false)
  }
})

// 打开事件处理
function handleOpen() {
  if (props.initialTab && tabs.some(t => t.id === props.initialTab)) {
    activeTab.value = props.initialTab
  }
}

// 关闭模态框
function handleClose() {
  isOpen.value = false
  emit('update:modelValue', false)
}

// 保存设置
async function handleSave() {
  const textDefaultsResult = await textStyleDefaultsRef.value?.saveDefaults() ?? {
    success: true,
    changed: false
  }

  if (!textDefaultsResult.success) {
    showToast(textDefaultsResult.error || '保存文本默认值失败', 'error')
    return
  }

  // 保存设置到 localStorage
  settingsStore.saveToStorage()
  
  // 同时保存到后端（config/user_settings.json）
  try {
    await settingsStore.saveToBackend()
  } catch (error) {
    console.warn('[SettingsModal] 保存到后端失败，仅保存到 localStorage:', error)
  }
  
  emit('save', { textDefaultsChanged: textDefaultsResult.changed })
  handleClose()
}
</script>

<style scoped>
/* ===================================
   设置模态框样式 - 基于 BaseModal 定制
   =================================== */

/* 覆盖 BaseModal overlay 的 z-index 提升到最高 */

/* 自定义头部样式 — 保留业务契约的渐变背景 */

/* BaseModal body 内的 tab + content 布局 */

.settings-tabs {
  display: flex;
  border-bottom: 1px solid var(--color-border-muted);
  background-color: var(--color-surface-input);
  padding: 0 15px;
  overflow: auto hidden;
  flex-shrink: 0;
  min-height: 48px;
}

.settings-tabs > .settings-tab {
  flex: 0 1 auto;
  min-width: fit-content;
  padding: 14px 20px;
  cursor: pointer;
  border: none;
  background: none;
  color: var(--color-text-strong);
  font-size: 0.95em;
  font-weight: 500;
  position: relative;
  transition: all 0.2s;
  white-space: nowrap;
  opacity: 0.7;
}

.settings-tabs > .settings-tab:hover {
  opacity: 1;
  background-color: var(--settings-modal-surface-raised);
}

.settings-tabs > .settings-tab.active {
  opacity: 1;
  color: var(--color-action-primary);
}

.settings-tabs > .settings-tab.active::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--color-action-primary);
  border-radius: 3px 3px 0 0;
}

.settings-tab-content {
  flex: 1;
  overflow-y: auto;
  padding: 25px;
}

.settings-tab-pane {
  display: block;
}

.settings-tab-pane.active {
  display: block;
  animation: settingsModalFadeIn 0.3s;
}


@keyframes settingsModalFadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@media (--breakpoint-md-down) {
  .settings-tabs {
    padding: 0 10px;
  }
  
  .settings-tabs > .settings-tab {
    flex: 0 0 auto;
    padding: 12px 14px;
    font-size: 0.9em;
  }
  
  .settings-tab-content {
    padding: 15px;
  }
}
</style>
