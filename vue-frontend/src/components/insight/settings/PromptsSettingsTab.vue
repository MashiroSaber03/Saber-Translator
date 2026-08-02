<script setup lang="ts">

import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList, { type ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import { ref, watch, onMounted } from 'vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiField from '@/components/ui/UiField.vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import type { PromptType, SavedPromptItem } from '@/api/insight'
import { confirmProductAction } from '@/composables/useProductConfirm'
import { requestProductTextInput } from '@/composables/useProductTextInput'
import { triggerBlobDownload } from '@/utils/browserDownload'
import { copyTextToClipboard } from '@/utils/clipboard'
import InsightSettingsPanel from './InsightSettingsPanel.vue'
import { PROMPT_TYPE_OPTIONS } from './types'

const emit = defineEmits<{
  (e: 'showMessage', message: string, type: 'success' | 'error'): void
  (e: 'update:prompts', prompts: Record<string, string>): void
}>()

const props = defineProps<{
  syncRequestId?: number
}>()

const insightStore = useInsightStore()

const currentPromptType = ref<PromptType>('batch_analysis')
const currentPromptContent = ref('')
const customPrompts = ref<Record<string, string>>({})
const savedPromptsLibrary = ref<SavedPromptItem[]>([])
const isLoadingPrompts = ref(false)
const promptsImportInput = ref<InstanceType<typeof UiFileInput> | null>(null)
const defaultPrompts = ref<Record<PromptType, string>>({
  batch_analysis: '',
  segment_summary: '',
  chapter_summary: '',
  qa_response: ''
})

async function loadDefaultPrompts(): Promise<void> {
  try {
    defaultPrompts.value = await insightApi.getDefaultPrompts()
  } catch {
    emit('showMessage', '默认提示词加载失败', 'error')
  }
}

async function loadPromptsLibrary(): Promise<void> {
  isLoadingPrompts.value = true
  try {
    savedPromptsLibrary.value = await insightApi.getPromptsLibrary()
  } catch {
    savedPromptsLibrary.value = []
  } finally {
    isLoadingPrompts.value = false
  }
}

async function resetCurrentPrompt(): Promise<void> {
  const confirmed = await confirmProductAction({
    title: '重置提示词',
    message: '确定要重置为默认提示词吗？当前编辑的内容将丢失。',
    confirmText: '重置',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed) return
  currentPromptContent.value = defaultPrompts.value[currentPromptType.value] || ''
  delete customPrompts.value[currentPromptType.value]
  emit('showMessage', '已重置为默认提示词', 'success')
}

async function copyPromptToClipboard(): Promise<void> {
  const copied = await copyTextToClipboard(currentPromptContent.value)
  emit('showMessage', copied ? '已复制到剪贴板' : '复制失败', copied ? 'success' : 'error')
}

async function savePromptToLibrary(): Promise<void> {
  const content = currentPromptContent.value.trim()
  if (!content) {
    emit('showMessage', '提示词内容不能为空', 'error')
    return
  }

  const name = await requestProductTextInput({
    title: '保存提示词',
    message: '请输入提示词名称：',
    placeholder: '提示词名称',
    confirmText: '保存',
    cancelText: '取消',
  })
  if (!name?.trim()) return

  const newPrompt: SavedPromptItem = {
    id: Date.now().toString(),
    name: name.trim(),
    type: currentPromptType.value,
    content: content,
    created_at: new Date().toISOString()
  }

  try {
    const saved = await insightApi.savePromptToLibrary(newPrompt)
    savedPromptsLibrary.value.push(saved)
    emit('showMessage', '提示词已保存到库', 'success')
  } catch {
    emit('showMessage', '保存失败', 'error')
  }
}

function loadPromptFromLibrary(promptItem: SavedPromptItem): void {
  currentPromptType.value = promptItem.type
  currentPromptContent.value = promptItem.content
  customPrompts.value[promptItem.type] = promptItem.content
  emitPrompts()
  emit('showMessage', `已加载提示词: ${promptItem.name}`, 'success')
}

function promptTypeChip(promptItem: SavedPromptItem): ProductChipItem[] {
  return [
    {
      id: promptItem.id,
      label: insightApi.PROMPT_METADATA[promptItem.type]?.label || promptItem.type,
      tone: 'primary',
    },
  ]
}

async function deletePromptFromLibrary(promptId: string): Promise<void> {
  const confirmed = await confirmProductAction({
    title: '删除提示词',
    message: '确定要删除这个提示词吗？',
    confirmText: '删除',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed) return

  try {
    await insightApi.deletePromptFromLibrary(promptId)
    savedPromptsLibrary.value = savedPromptsLibrary.value.filter(p => p.id !== promptId)
    emit('showMessage', '提示词已删除', 'success')
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
  triggerBlobDownload(blob, `manga-insight-prompts-${new Date().toISOString().slice(0, 10)}.json`)

  emit('showMessage', '提示词已导出', 'success')
}

function triggerImportPrompts(): void {
  promptsImportInput.value?.click()
}

async function handlePromptsFileImport(files: File[]): Promise<void> {
  const file = files[0]
  if (!file) return

  try {
    const text = await file.text()
    const importData = JSON.parse(text)

    if (importData.prompts) {
      customPrompts.value = { ...customPrompts.value, ...importData.prompts }
      emitPrompts()
    }

    if (importData.library && Array.isArray(importData.library)) {
      const existingIds = new Set(savedPromptsLibrary.value.map(p => p.id))
      for (const promptItem of importData.library) {
        if (!existingIds.has(promptItem.id)) {
          savedPromptsLibrary.value.push(promptItem)
        }
      }
      savedPromptsLibrary.value = await insightApi.importPromptsLibrary(
        savedPromptsLibrary.value
      )
    }

    emit('showMessage', '提示词导入成功', 'success')
  } catch {
    emit('showMessage', '导入失败，请检查文件格式', 'error')
  }

  promptsImportInput.value?.clear()
}

watch(currentPromptType, (newType, previousType) => {
  if (previousType && currentPromptContent.value) {
    customPrompts.value[previousType] = currentPromptContent.value
  }
  if (newType) {
    currentPromptContent.value = customPrompts.value[newType] || defaultPrompts.value[newType] || ''
  }
})

watch([currentPromptType, currentPromptContent], () => {
  emitPrompts()
}, { immediate: true })

function collectDraftPrompts(): Record<string, string> {
  if (currentPromptContent.value) {
    customPrompts.value[currentPromptType.value] = currentPromptContent.value
  }
  return customPrompts.value
}

function emitPrompts(): void {
  emit('update:prompts', { ...collectDraftPrompts() })
}

function refreshDraftFromStore(): void {
  if (insightStore.config.prompts) {
    customPrompts.value = { ...insightStore.config.prompts }
  } else {
    customPrompts.value = {}
  }
  currentPromptContent.value = customPrompts.value[currentPromptType.value] || defaultPrompts.value[currentPromptType.value] || ''
  emitPrompts()
}

watch(() => props.syncRequestId, () => {
  refreshDraftFromStore()
})

async function initialize(): Promise<void> {
  await loadDefaultPrompts()
  await loadPromptsLibrary()
  refreshDraftFromStore()
}

onMounted(initialize)
</script>

<template>
  <InsightSettingsPanel class="prompts-settings-tab" description="自定义分析过程中使用的提示词模板。">
    <UiField variant="settings" label="提示词类型" :hint="insightApi.PROMPT_METADATA[currentPromptType]?.hint">
      <UiSelect v-model="currentPromptType" :options="PROMPT_TYPE_OPTIONS" />
    </UiField>

    <UiField variant="settings" label="提示词内容">
      <UiTextarea v-model="currentPromptContent" class="prompts-settings-tab__editor" variant="panel" size="lg" rows="12" placeholder="输入提示词内容..." />
    </UiField>

    <ProductActionRow aria-label="提示词编辑操作">
      <UiButton variant="secondary" @click="resetCurrentPrompt" title="重置为默认" size="sm">
        <UiIcon name="refresh" size="14" />
        <span>重置</span>
      </UiButton>
      <UiButton variant="secondary" @click="copyPromptToClipboard" title="复制到剪贴板" size="sm">
        <UiIcon name="copy" size="14" />
        <span>复制</span>
      </UiButton>
      <UiButton variant="primary" @click="savePromptToLibrary" title="保存到库" size="sm">
        <UiIcon name="save" size="14" />
        <span>保存到库</span>
      </UiButton>
    </ProductActionRow>

    <hr class="prompts-settings-tab__divider">

    <div class="prompts-settings-tab__library">
      <div class="prompts-settings-tab__library-header">
        <h4 class="prompts-settings-tab__library-title">
          <UiIcon name="book-open" size="15" />
          <span>提示词库</span>
        </h4>
        <ProductActionRow aria-label="提示词库导入导出操作">
          <UiButton variant="secondary" @click="exportAllPrompts" title="导出所有提示词" size="sm">
            <UiIcon name="download" size="14" />
            <span>导出</span>
          </UiButton>
          <UiButton variant="secondary" @click="triggerImportPrompts" title="导入提示词" size="sm">
            <UiIcon name="upload" size="14" />
            <span>导入</span>
          </UiButton>
          <UiFileInput
            ref="promptsImportInput"
            accept=".json"
            hidden
            @files-change="handlePromptsFileImport"
          />
        </ProductActionRow>
      </div>

      <div class="prompts-settings-tab__saved-list">
        <ProductStatusBanner
          v-if="isLoadingPrompts"
          tone="neutral"
          icon-name="refresh"
          title="正在加载提示词库"
          aria-live="polite"
        >
          正在同步已保存的 Insight 提示词。
        </ProductStatusBanner>
        <ProductStatusBanner
          v-else-if="savedPromptsLibrary.length === 0"
          tone="neutral"
          icon-name="file-text"
          title="暂无保存的提示词"
        >
          保存后的提示词会显示在这里。
        </ProductStatusBanner>
        <div v-else v-for="promptItem in savedPromptsLibrary" :key="promptItem.id" class="prompts-settings-tab__saved-item">
          <ProductRecordCard class="prompts-settings-tab__saved-card">
            <template #actions>
              <UiIconButton
                type="button"
                variant="danger"
                size="sm"
                :label="`删除提示词：${promptItem.name}`"
                @click="deletePromptFromLibrary(promptItem.id)"
              >
                <UiIcon name="trash" size="14" />
              </UiIconButton>
            </template>

            <UiButton
              variant="toolbar"
              type="button"
              class="prompts-settings-tab__saved-load"
              :aria-label="`加载提示词：${promptItem.name}`"
              @click="loadPromptFromLibrary(promptItem)"
            >
              <span class="prompts-settings-tab__prompt-name">{{ promptItem.name }}</span>
              <ProductChipList :items="promptTypeChip(promptItem)" />
            </UiButton>
          </ProductRecordCard>
        </div>
      </div>
    </div>
  </InsightSettingsPanel>
</template>

<style scoped>
.prompts-settings-tab {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.prompts-settings-tab__editor {
  width: 100%;
  font-family: var(--font-mono);
}

.prompts-settings-tab__divider {
  border: none;
  border-top: 1px solid var(--color-border-muted);
  margin: 16px 0;
}

.prompts-settings-tab__library {
  margin-top: 8px;
}

.prompts-settings-tab__library-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px 12px;
  margin-bottom: 12px;
  min-width: 0;
}

.prompts-settings-tab__library-title {
  display: inline-flex;
  flex: 1 1 180px;
  align-items: center;
  gap: 6px;
  min-width: 0;
  margin: 0;
  font-size: 14px;
  font-weight: 500;
  overflow-wrap: anywhere;
}

.prompts-settings-tab__saved-list {
  max-height: 200px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.prompts-settings-tab__saved-item {
  min-width: 0;
}

.prompts-settings-tab__saved-card {
  --product-record-card-padding: 8px 10px;
  --product-record-card-background: var(--color-surface-card);
  --product-record-card-shadow-hover: none;
}

.prompts-settings-tab__saved-load {
  display: flex;
  width: 100%;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
  padding: 0;
  color: inherit;
  text-align: left;
}

.prompts-settings-tab__prompt-name {
  flex: 1;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

</style>
