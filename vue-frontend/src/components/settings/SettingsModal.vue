<template>
  <!-- 设置模态框 -->
  <BaseModal
    v-model="isOpen"
    title="⚙️ 设置"
    size="large"
    custom-class="settings-modal-wrapper"
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
      <button
        v-for="tab in tabs"
        :key="tab.id"
        class="settings-tab"
        :class="{ active: activeTab === tab.id }"
        @click="activeTab = tab.id"
      >
        {{ tab.label }}
      </button>
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
      <button class="btn btn-secondary" @click="handleClose">取消</button>
      <button class="btn btn-primary" @click="handleSave">保存设置</button>
    </template>
  </BaseModal>
</template>

<script setup lang="ts">
/**
 * 设置模态框组件
 * 管理所有一次性配置的集中设置界面
 * 基于 BaseModal 实现
 */
import { ref, watch } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import BaseModal from '@/components/common/BaseModal.vue'
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
      // 【修复问题2】如果指定了初始Tab，则跳转到该Tab
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
    console.log('[SettingsModal] 设置已保存到后端')
  } catch (error) {
    console.warn('[SettingsModal] 保存到后端失败，仅保存到 localStorage:', error)
  }
  
  emit('save', { textDefaultsChanged: textDefaultsResult.changed })
  handleClose()
}
</script>

<style>
/* 不使用 scoped，因为 BaseModal 使用 Teleport 将内容传送到 body */
/* ===================================
   设置模态框样式 - 基于 BaseModal 定制
   =================================== */

/* 覆盖 BaseModal overlay 的 z-index 提升到最高 */
.settings-modal-wrapper .modal-container {
  max-width: 900px;
  width: 90%;
  max-height: 90vh;
}

/* 自定义头部样式 — 保留原版的渐变背景 */
.settings-modal-wrapper .modal-header {
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-dark) 100%);
  color: white;
  padding: 20px 25px;
}

.settings-modal-wrapper .modal-title {
  color: white;
  font-size: 1.4em;
}

.settings-modal-wrapper .modal-close-btn {
  color: rgb(255, 255, 255, 0.8);
  font-size: 20px;
}

.settings-modal-wrapper .modal-close-btn:hover {
  color: white;
  background-color: rgb(255, 255, 255, 0.15);
}

/* BaseModal body 内的 tab + content 布局 */
.settings-modal-wrapper .modal-body {
  padding: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.settings-modal-wrapper .settings-tabs {
  display: flex;
  border-bottom: 1px solid var(--border-color);
  background-color: var(--input-bg-color);
  padding: 0 15px;
  overflow: auto hidden;
  flex-shrink: 0;
  min-height: 48px;
}

.settings-modal-wrapper .settings-tab {
  padding: 14px 20px;
  cursor: pointer;
  border: none;
  background: none;
  color: var(--text-color);
  font-size: 0.95em;
  font-weight: 500;
  position: relative;
  transition: all 0.2s;
  white-space: nowrap;
  opacity: 0.7;
}

.settings-modal-wrapper .settings-tab:hover {
  opacity: 1;
  background-color: rgb(0, 0, 0, 0.03);
}

.settings-modal-wrapper .settings-tab.active {
  opacity: 1;
  color: var(--color-primary);
}

.settings-modal-wrapper .settings-tab.active::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--color-primary);
  border-radius: 3px 3px 0 0;
}

.settings-modal-wrapper .settings-tab-content {
  flex: 1;
  overflow-y: auto;
  padding: 25px;
}

.settings-modal-wrapper .settings-tab-pane {
  display: block;
}

.settings-modal-wrapper .settings-tab-pane.active {
  display: block;
  animation: settingsModalFadeIn 0.3s;
}

/* Note: .settings-group, .settings-group-title, .settings-item, .settings-row, .settings-test-btn
   are now defined in global.css and inherited */

@keyframes settingsModalFadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@media (width <= 768px) {
  .settings-modal-wrapper .modal-container {
    margin: 0;
    max-height: 100vh;
    border-radius: 0;
    width: 100%;
  }
  
  .settings-modal-wrapper .settings-tabs {
    padding: 0 10px;
  }
  
  .settings-modal-wrapper .settings-tab {
    padding: 12px 14px;
    font-size: 0.9em;
  }
  
  .settings-modal-wrapper .settings-tab-content {
    padding: 15px;
  }
  
  .settings-modal-wrapper .settings-row {
    grid-template-columns: 1fr;
  }
}
</style>
