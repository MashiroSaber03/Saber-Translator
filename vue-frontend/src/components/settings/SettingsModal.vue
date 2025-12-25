<template>
  <!-- 设置模态框 -->
  <div v-if="isOpen" class="settings-modal" @click.self="handleClose">
    <div class="settings-modal-content">
      <!-- 头部 -->
      <div class="settings-modal-header">
        <h3>⚙️ 设置</h3>
        <span class="settings-modal-close" @click="handleClose">&times;</span>
      </div>

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

        <!-- 更多设置 -->
        <div v-show="activeTab === 'more'" class="settings-tab-pane">
          <MoreSettings />
        </div>
      </div>

      <!-- 底部按钮 -->
      <div class="settings-modal-footer">
        <button class="btn btn-secondary" @click="handleClose">取消</button>
        <button class="btn btn-primary" @click="handleSave">保存设置</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 设置模态框组件
 * 管理所有一次性配置的集中设置界面
 */
import { ref, watch, onMounted, onUnmounted } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import OcrSettings from './OcrSettings.vue'
import TranslationSettings from './TranslationSettings.vue'
import DetectionSettings from './DetectionSettings.vue'
import HqTranslationSettings from './HqTranslationSettings.vue'
import ProofreadingSettings from './ProofreadingSettings.vue'
import PromptLibrary from './PromptLibrary.vue'
import PluginManager from './PluginManager.vue'
import MoreSettings from './MoreSettings.vue'

// Props
const props = defineProps<{
  modelValue: boolean
}>()

// Emits
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'save'): void
}>()

// Store
const settingsStore = useSettingsStore()

// 本地状态
const isOpen = ref(props.modelValue)
const activeTab = ref('ocr')

// Tab 配置
const tabs = [
  { id: 'ocr', label: 'OCR识别' },
  { id: 'translate', label: '翻译服务' },
  { id: 'detection', label: '检测设置' },
  { id: 'hq', label: '高质量翻译' },
  { id: 'proofreading', label: 'AI校对' },
  { id: 'prompt-library', label: '提示词管理' },
  { id: 'plugins', label: '插件管理' },
  { id: 'more', label: '更多' }
]

// 监听 props 变化
watch(
  () => props.modelValue,
  (newVal) => {
    isOpen.value = newVal
    if (newVal) {
      // 打开时禁止背景滚动
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
  }
)

// 关闭模态框
function handleClose() {
  isOpen.value = false
  emit('update:modelValue', false)
  document.body.style.overflow = ''
}

// 保存设置
async function handleSave() {
  // 保存设置到 localStorage
  settingsStore.saveToStorage()
  
  // 同时保存到后端（config/user_settings.json）
  try {
    await settingsStore.saveToBackend()
    console.log('[SettingsModal] 设置已保存到后端')
  } catch (error) {
    console.warn('[SettingsModal] 保存到后端失败，仅保存到 localStorage:', error)
  }
  
  emit('save')
  handleClose()
}

// 键盘事件处理
function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape' && isOpen.value) {
    handleClose()
  }
}

// 生命周期
onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
  document.body.style.overflow = ''
})
</script>

<style scoped>
/* ===================================
   设置模态框样式 - 完整迁移自 settings-modal.css
   =================================== */

.settings-modal {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  position: fixed;
  z-index: 10002;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  overflow: auto;
  background-color: rgba(0,0,0,0.5);
  animation: fadeIn 0.3s;
}

.settings-modal-content {
  background-color: var(--card-bg-color);
  color: var(--text-color);
  margin: 3% auto;
  padding: 0;
  border-radius: 12px;
  box-shadow: 0 10px 40px var(--shadow-color);
  width: 90%;
  max-width: 900px;
  max-height: 90vh;
  position: relative;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.settings-modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 25px;
  border-bottom: 1px solid var(--border-color);
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-dark) 100%);
  color: white;
}

.settings-modal-header h3 {
  margin: 0;
  font-size: 1.4em;
  font-weight: 600;
}

.settings-modal-close {
  color: rgba(255,255,255,0.8);
  font-size: 28px;
  font-weight: bold;
  cursor: pointer;
  transition: color 0.2s;
  line-height: 1;
}

.settings-modal-close:hover {
  color: white;
}

.settings-tabs {
  display: flex;
  border-bottom: 1px solid var(--border-color);
  background-color: var(--input-bg-color);
  padding: 0 15px;
  overflow-x: auto;
  overflow-y: hidden;
  flex-shrink: 0;
  min-height: 48px;
}

.settings-tab {
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

.settings-tab:hover {
  opacity: 1;
  background-color: rgba(0,0,0,0.03);
}

[data-theme="dark"] .settings-tab:hover {
  background-color: rgba(255,255,255,0.05);
}

.settings-tab.active {
  opacity: 1;
  color: var(--color-primary);
}

.settings-tab.active::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--color-primary);
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
  animation: fadeIn 0.3s;
}

.settings-group {
  margin-bottom: 25px;
  padding: 20px;
  background-color: var(--input-bg-color);
  border-radius: 10px;
  border: 1px solid var(--border-color);
}

.settings-group-title {
  font-size: 1.1em;
  font-weight: 600;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border-color);
  color: var(--color-primary);
}

.settings-item {
  margin-bottom: 15px;
}

.settings-item:last-child {
  margin-bottom: 0;
}

.settings-item label {
  display: block;
  margin-bottom: 6px;
  font-weight: 500;
  font-size: 0.95em;
}

.settings-item input[type="text"],
.settings-item input[type="number"],
.settings-item input[type="password"],
.settings-item select,
.settings-item textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  background-color: var(--card-bg-color);
  color: var(--text-color);
  font-size: 0.95em;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.settings-item input:focus,
.settings-item select:focus,
.settings-item textarea:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.15);
}

.settings-item textarea {
  min-height: 100px;
  resize: vertical;
}

.settings-item .input-hint {
  font-size: 0.85em;
  color: var(--text-secondary);
  margin-top: 5px;
}

.settings-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
}

.settings-test-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  background-color: var(--color-info);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9em;
  transition: background-color 0.2s, transform 0.1s;
  margin-top: 10px;
}

.settings-test-btn:hover {
  background-color: var(--color-info-dark);
  transform: translateY(-1px);
}

.settings-test-btn:active {
  transform: translateY(0);
}

.settings-modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 15px 20px;
  border-top: 1px solid var(--border-color);
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@media (max-width: 768px) {
  .settings-modal-content {
    margin: 0;
    max-height: 100vh;
    border-radius: 0;
    width: 100%;
  }
  
  .settings-tabs {
    padding: 0 10px;
  }
  
  .settings-tab {
    padding: 12px 14px;
    font-size: 0.9em;
  }
  
  .settings-tab-content {
    padding: 15px;
  }
  
  .settings-row {
    grid-template-columns: 1fr;
  }
}
</style>
