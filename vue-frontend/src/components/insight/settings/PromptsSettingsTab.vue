<script setup lang="ts">

import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
/**
 * 提示词设置选项卡组件
 */
import { ref, watch, onMounted } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import type { PromptType, SavedPromptItem } from '@/api/insight'
import { PROMPT_TYPE_OPTIONS } from './types'

const emit = defineEmits<{
  (e: 'showMessage', message: string, type: 'success' | 'error'): void
}>()

const insightStore = useInsightStore()

const currentPromptType = ref<PromptType>('batch_analysis')
const currentPromptContent = ref('')
const customPrompts = ref<Record<string, string>>({})
const savedPromptsLibrary = ref<SavedPromptItem[]>([])
const isLoadingPrompts = ref(false)
const defaultPrompts = ref<Record<PromptType, string>>({
  batch_analysis: '',
  segment_summary: '',
  chapter_summary: '',
  qa_response: ''
})

async function loadDefaultPrompts(): Promise<void> {
  try {
    const response = await insightApi.getDefaultPrompts()
    if (response.success && response.prompts) {
      defaultPrompts.value = response.prompts
    }
  } catch (error) {
    console.error('加载默认提示词失败:', error)
  }
}

async function loadPromptsLibrary(): Promise<void> {
  isLoadingPrompts.value = true
  try {
    const response = await insightApi.getPromptsLibrary()
    if (response.success && response.library) {
      savedPromptsLibrary.value = response.library
    }
  } catch {
    savedPromptsLibrary.value = []
  } finally {
    isLoadingPrompts.value = false
  }
}

function resetCurrentPrompt(): void {
  if (confirm('确定要重置为默认提示词吗？当前编辑的内容将丢失。')) {
    currentPromptContent.value = defaultPrompts.value[currentPromptType.value] || ''
    delete customPrompts.value[currentPromptType.value]
    emit('showMessage', '已重置为默认提示词', 'success')
  }
}

async function copyPromptToClipboard(): Promise<void> {
  try {
    await navigator.clipboard.writeText(currentPromptContent.value)
    emit('showMessage', '已复制到剪贴板', 'success')
  } catch {
    emit('showMessage', '复制失败', 'error')
  }
}

async function savePromptToLibrary(): Promise<void> {
  const content = currentPromptContent.value.trim()
  if (!content) {
    emit('showMessage', '提示词内容不能为空', 'error')
    return
  }

  const name = prompt('请输入提示词名称：')
  if (!name?.trim()) return

  const newPrompt: SavedPromptItem = {
    id: Date.now().toString(),
    name: name.trim(),
    type: currentPromptType.value,
    content: content,
    created_at: new Date().toISOString()
  }

  try {
    const response = await insightApi.savePromptToLibrary(newPrompt)
    if (response.success) {
      savedPromptsLibrary.value.push(newPrompt)
      emit('showMessage', '提示词已保存到库', 'success')
    } else {
      emit('showMessage', '保存失败', 'error')
    }
  } catch {
    emit('showMessage', '保存失败', 'error')
  }
}

function loadPromptFromLibrary(promptItem: SavedPromptItem): void {
  currentPromptType.value = promptItem.type
  currentPromptContent.value = promptItem.content
  customPrompts.value[promptItem.type] = promptItem.content
  emit('showMessage', `已加载提示词: ${promptItem.name}`, 'success')
}

async function deletePromptFromLibrary(promptId: string): Promise<void> {
  if (!confirm('确定要删除这个提示词吗？')) return

  try {
    const response = await insightApi.deletePromptFromLibrary(promptId)
    if (response.success) {
      savedPromptsLibrary.value = savedPromptsLibrary.value.filter(p => p.id !== promptId)
      emit('showMessage', '提示词已删除', 'success')
    } else {
      emit('showMessage', '删除失败', 'error')
    }
  } catch {
    emit('showMessage', '删除失败', 'error')
  }
}

function exportAllPrompts(): void {
  if (currentPromptContent.value) {
    customPrompts.value[currentPromptType.value] = currentPromptContent.value
  }

  const exportData = {
    version: '1.0',
    exported_at: new Date().toISOString(),
    prompts: customPrompts.value,
    library: savedPromptsLibrary.value
  }

  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `manga-insight-prompts-${new Date().toISOString().slice(0, 10)}.json`
  a.click()
  URL.revokeObjectURL(url)

  emit('showMessage', '提示词已导出', 'success')
}

function triggerImportPrompts(): void {
  document.getElementById('promptsFileInput')?.click()
}

async function handlePromptsFileImport(event: Event): Promise<void> {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return

  try {
    const text = await file.text()
    const importData = JSON.parse(text)

    if (importData.prompts) {
      customPrompts.value = { ...customPrompts.value, ...importData.prompts }
    }

    if (importData.library && Array.isArray(importData.library)) {
      const existingIds = new Set(savedPromptsLibrary.value.map(p => p.id))
      for (const promptItem of importData.library) {
        if (!existingIds.has(promptItem.id)) {
          savedPromptsLibrary.value.push(promptItem)
        }
      }
      await insightApi.importPromptsLibrary(savedPromptsLibrary.value)
    }

    emit('showMessage', '提示词导入成功', 'success')
  } catch {
    emit('showMessage', '导入失败，请检查文件格式', 'error')
  }

  target.value = ''
}

watch(currentPromptType, (newType, oldType) => {
  if (oldType && currentPromptContent.value) {
    customPrompts.value[oldType] = currentPromptContent.value
  }
  if (newType) {
    currentPromptContent.value = customPrompts.value[newType] || defaultPrompts.value[newType] || ''
  }
})

function getCustomPrompts() {
  if (currentPromptContent.value) {
    customPrompts.value[currentPromptType.value] = currentPromptContent.value
  }
  return customPrompts.value
}

function syncFromStore(): void {
  if (insightStore.config.prompts) {
    customPrompts.value = { ...insightStore.config.prompts }
  } else {
    customPrompts.value = {}
  }
  currentPromptContent.value = customPrompts.value[currentPromptType.value] || defaultPrompts.value[currentPromptType.value] || ''
}

async function initialize(): Promise<void> {
  await loadDefaultPrompts()
  await loadPromptsLibrary()
}

onMounted(initialize)

defineExpose({ getCustomPrompts, syncFromStore, initialize })
</script>

<template>
  <div class="insight-settings-content prompts-settings">
    <p class="settings-hint">自定义分析过程中使用的提示词模板。</p>

    <div class="insight-settings-field">
      <label>提示词类型</label>
      <CustomSelect v-model="currentPromptType" :options="PROMPT_TYPE_OPTIONS" />
      <p class="form-hint">{{ insightApi.PROMPT_METADATA[currentPromptType]?.hint }}</p>
    </div>

    <div class="insight-settings-field">
      <label>提示词内容</label>
      <UiTextarea v-model="currentPromptContent" class="prompt-editor" rows="12" placeholder="输入提示词内容..." />
    </div>

    <div class="prompt-actions-bar">
      <UiButton variant="secondary" @click="resetCurrentPrompt" title="重置为默认" size="sm">🔄 重置</UiButton>
      <UiButton variant="secondary" @click="copyPromptToClipboard" title="复制到剪贴板" size="sm">📋 复制</UiButton>
      <UiButton variant="primary" @click="savePromptToLibrary" title="保存到库" size="sm">💾 保存到库</UiButton>
    </div>

    <hr class="section-divider">

    <div class="prompts-library-section">
      <div class="library-header">
        <h4>📚 提示词库</h4>
        <div class="library-actions">
          <UiButton variant="secondary" @click="exportAllPrompts" title="导出所有提示词" size="sm">📤 导出</UiButton>
          <UiButton variant="secondary" @click="triggerImportPrompts" title="导入提示词" size="sm">📥 导入</UiButton>
          <UiFileInput id="promptsFileInput" accept=".json" style="display: none" @change="handlePromptsFileImport" />
        </div>
      </div>

      <div class="saved-prompts-list">
        <div v-if="isLoadingPrompts" class="loading-text">加载中...</div>
        <div v-else-if="savedPromptsLibrary.length === 0" class="placeholder-text">暂无保存的提示词</div>
        <div v-else v-for="promptItem in savedPromptsLibrary" :key="promptItem.id" class="saved-prompt-item">
          <UiButton
            variant="toolbar"
            type="button"
            class="saved-prompt-load"
            :aria-label="`加载提示词：${promptItem.name}`"
            @click="loadPromptFromLibrary(promptItem)"
          >
            <span class="prompt-name">{{ promptItem.name }}</span>
            <span class="prompt-type-badge">{{ insightApi.PROMPT_METADATA[promptItem.type]?.label || promptItem.type }}</span>
          </UiButton>
          <UiButton
            variant="toolbar"
            type="button"
            class="button-icon-sm saved-prompt-delete"
            :aria-label="`删除提示词：${promptItem.name}`"
            title="删除"
            @click="deletePromptFromLibrary(promptItem.id)"
          >
            🗑️
          </UiButton>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.insight-settings-content {
  --prompts-settings-tab-border-default: rgba(99, 102, 241, .2);
  --prompts-settings-tab-surface-base: rgba(99, 102, 241, .05);
}

.insight-settings-content {
  padding: 16px 0;
  min-height: 300px;
}

.insight-settings-content .settings-hint {
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 13px;
  margin-bottom: 16px;
  padding: 8px 12px;
  background: var(--color-surface-muted);
  border-radius: 4px;
}

.insight-settings-content .insight-settings-field {
  margin-bottom: 16px;
}

.insight-settings-content .insight-settings-field label {
  display: block;
  margin-bottom: 6px;
  font-weight: 500;
  font-size: 14px;
  color: var(--color-text-default, var(--color-text-default));
}

.insight-settings-content .insight-settings-field input[type="text"],
.insight-settings-content .insight-settings-field input[type="password"],
.insight-settings-content .insight-settings-field input[type="number"],
.insight-settings-content .insight-settings-field select,
.insight-settings-content .insight-settings-field textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 6px;
  font-size: 14px;
  background: var(--color-surface-input, var(--color-surface-base));
  color: var(--color-text-default, var(--color-text-default));
  transition: border-color 0.2s, box-shadow 0.2s;
}

.insight-settings-content .insight-settings-field input:focus,
.insight-settings-content .insight-settings-field select:focus,
.insight-settings-content .insight-settings-field textarea:focus {
  outline: none;
  border-color: var(--color-border-brand);
  box-shadow: 0 0 0 3px var(--color-focus-brand-soft);
}

.insight-settings-content .form-hint {
  margin-top: 4px;
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.insight-settings-content .ui-checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-weight: normal;
}

.insight-settings-content .ui-checkbox-label input[type="checkbox"] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

.insight-settings-content {
  --ui-button-padding: 10px 16px;
  --ui-button-radius: 6px;
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--color-surface-brand);
  --ui-button-primary-hover-background: var(--color-surface-brand-strong);
  --ui-button-secondary-background: var(--color-surface-muted);
  --ui-button-secondary-color: var(--color-text-default, var(--color-text-default));
  --ui-button-secondary-border: 1px solid var(--color-border-muted, var(--color-border-default));
  --ui-button-secondary-hover-background: var(--color-surface-hover);
  --ui-button-sm-padding: 6px 12px;
  --ui-button-sm-font-size: 13px;
  --ui-button-disabled-opacity: 0.6;
}

.insight-settings-content .form-row {
  display: flex;
  gap: 16px;
}

.insight-settings-content .form-row .insight-settings-field {
  flex: 1;
}

.insight-settings-content .placeholder-text {
  color: var(--color-text-supporting, var(--color-text-secondary));
  text-align: center;
  padding: 40px;
}

.insight-settings-content .prompts-settings {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.insight-settings-content .prompt-editor {
  width: 100%;
  min-height: 200px;
  font-family: Consolas, Monaco, monospace;
  font-size: 13px;
  line-height: 1.5;
  padding: 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 4px;
  background: var(--color-surface-muted);
  color: var(--color-text-default, var(--color-text-default));
  resize: vertical;
}

.insight-settings-content .prompt-editor:focus {
  outline: none;
  border-color: var(--color-border-brand);
}

.insight-settings-content .prompt-actions-bar {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}


.insight-settings-content .section-divider {
  border: none;
  border-top: 1px solid var(--color-border-muted, var(--color-border-default));
  margin: 16px 0;
}

.insight-settings-content .prompts-library-section {
  margin-top: 8px;
}

.insight-settings-content .library-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.insight-settings-content .library-header h4 {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
}

.insight-settings-content .library-actions {
  display: flex;
  gap: 8px;
}

.insight-settings-content .saved-prompts-list {
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 4px;
  background: var(--color-surface-muted);
}

.insight-settings-content .saved-prompt-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 12px 0 0;
  border-bottom: 1px solid var(--color-border-muted, var(--color-border-default));
  transition: background 0.2s;
}

.insight-settings-content .saved-prompt-item:last-child {
  border-bottom: none;
}

.insight-settings-content .saved-prompt-item:hover {
  background: var(--color-surface-hover);
}

.insight-settings-content .saved-prompt-load {
  display: flex;
  flex: 1;
  align-items: center;
  gap: 8px;
  min-width: 0;
  padding: 8px 0 8px 12px;
  color: inherit;
  text-align: left;
}

.insight-settings-content .prompt-name {
  flex: 1;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.insight-settings-content .prompt-type-badge {
  font-size: 11px;
  padding: 2px 6px;
  background: var(--color-focus-brand-soft);
  color: var(--color-text-brand);
  border-radius: 4px;
  white-space: nowrap;
}

.insight-settings-content .button-icon-sm {
  padding: 2px 6px;
  background: none;
  border: none;
  cursor: pointer;
  opacity: 0.6;
  transition: opacity 0.2s;
}

.insight-settings-content .button-icon-sm:hover {
  opacity: 1;
}

.insight-settings-content .loading-text {
  text-align: center;
  padding: 20px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.insight-settings-content .batch-info-box {
  margin-top: 16px;
  padding: 12px;
  background: var(--color-surface-subtle);
  border-radius: 8px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
}

.insight-settings-content .batch-info-box h4 {
  margin: 0 0 8px;
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-default, var(--color-text-default));
}

.insight-settings-content .layers-preview-list {
  margin: 0;
  padding-left: 20px;
  font-size: 13px;
  line-height: 1.6;
}

.insight-settings-content .layers-preview-list li {
  margin-bottom: 4px;
}

.insight-settings-content .align-badge {
  color: var(--color-text-brand);
  font-size: 12px;
}

.insight-settings-content .batch-estimate-box {
  margin-top: 12px;
  padding: 10px 12px;
  background: linear-gradient(135deg, var(--color-focus-brand-soft), var(--prompts-settings-tab-surface-base));
  border-radius: 6px;
  border: 1px solid var(--prompts-settings-tab-border-default);
}

.insight-settings-content .batch-estimate-box p {
  margin: 0;
  font-size: 13px;
  color: var(--color-text-default, var(--color-text-default));
}

.insight-settings-content .batch-estimate-box strong {
  color: var(--color-text-brand);
}

.insight-settings-content .model-input-row {
  display: flex;
  gap: 8px;
  align-items: center;
}

.insight-settings-content .model-input-row input {
  flex: 1;
}

.insight-settings-content .fetch-btn {
  white-space: nowrap;
  flex-shrink: 0;
}

.insight-settings-content .model-select-container {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  padding: 8px 12px;
  background: var(--color-surface-subtle);
  border-radius: 6px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
}

.insight-settings-content .model-select {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 4px;
  font-size: 13px;
  background: var(--color-surface-input, var(--color-surface-base));
  color: var(--color-text-default, var(--color-text-default));
  cursor: pointer;
}

.insight-settings-content .model-select:focus {
  outline: none;
  border-color: var(--color-border-brand);
}

.insight-settings-content .model-count {
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  white-space: nowrap;
}
</style>
