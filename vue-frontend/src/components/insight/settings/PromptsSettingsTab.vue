<script setup lang="ts">
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
    
    <div class="form-group">
      <label>提示词类型</label>
      <CustomSelect v-model="currentPromptType" :options="PROMPT_TYPE_OPTIONS" />
      <p class="form-hint">{{ insightApi.PROMPT_METADATA[currentPromptType]?.hint }}</p>
    </div>
    
    <div class="form-group">
      <label>提示词内容</label>
      <textarea v-model="currentPromptContent" class="prompt-editor" rows="12" placeholder="输入提示词内容..."></textarea>
    </div>
    
    <div class="prompt-actions-bar">
      <button class="btn btn-secondary btn-sm" @click="resetCurrentPrompt" title="重置为默认">🔄 重置</button>
      <button class="btn btn-secondary btn-sm" @click="copyPromptToClipboard" title="复制到剪贴板">📋 复制</button>
      <button class="btn btn-primary btn-sm" @click="savePromptToLibrary" title="保存到库">💾 保存到库</button>
    </div>
    
    <hr class="section-divider">
    
    <div class="prompts-library-section">
      <div class="library-header">
        <h4>📚 提示词库</h4>
        <div class="library-actions">
          <button class="btn btn-secondary btn-sm" @click="exportAllPrompts" title="导出所有提示词">📤 导出</button>
          <button class="btn btn-secondary btn-sm" @click="triggerImportPrompts" title="导入提示词">📥 导入</button>
          <input id="promptsFileInput" type="file" accept=".json" style="display: none" @change="handlePromptsFileImport">
        </div>
      </div>
      
      <div class="saved-prompts-list">
        <div v-if="isLoadingPrompts" class="loading-text">加载中...</div>
        <div v-else-if="savedPromptsLibrary.length === 0" class="placeholder-text">暂无保存的提示词</div>
        <div v-else v-for="promptItem in savedPromptsLibrary" :key="promptItem.id" class="saved-prompt-item" @click="loadPromptFromLibrary(promptItem)">
          <span class="prompt-name">{{ promptItem.name }}</span>
          <span class="prompt-type-badge">{{ insightApi.PROMPT_METADATA[promptItem.type]?.label || promptItem.type }}</span>
          <button class="btn-icon-sm" @click.stop="deletePromptFromLibrary(promptItem.id)" title="删除">🗑️</button>
        </div>
      </div>
    </div>
  </div>
</template>
