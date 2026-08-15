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
import type { PromptType, SavedPromptInput, SavedPromptItem } from '@/api/insight'
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
const customPrompts = ref<Record<string, string>>({ ...insightStore.config.prompts })
const currentPromptContent = ref(customPrompts.value.batch_analysis ?? '')
const savedPromptsLibrary = ref<SavedPromptItem[]>([])
const isLoadingPrompts = ref(false)
const isResettingPrompt = ref(false)
const isSavingPrompt = ref(false)
const isImportingPrompts = ref(false)
const deletingPromptIds = ref<Set<string>>(new Set())
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
    for (const type of insightApi.INSIGHT_PROMPT_TYPES) {
      if (customPrompts.value[type] === undefined) {
        customPrompts.value[type] = defaultPrompts.value[type]
      }
    }
    currentPromptContent.value = customPrompts.value[currentPromptType.value] ?? ''
    emitPrompts()
  } catch (error) {
    emit(
      'showMessage',
      '默认提示词加载失败: ' + (error instanceof Error ? error.message : '网络错误'),
      'error',
    )
  }
}

async function loadPromptsLibrary(): Promise<void> {
  isLoadingPrompts.value = true
  try {
    savedPromptsLibrary.value = await insightApi.getPromptsLibrary()
  } catch (error) {
    savedPromptsLibrary.value = []
    emit(
      'showMessage',
      '提示词库加载失败: ' + (error instanceof Error ? error.message : '网络错误'),
      'error',
    )
  } finally {
    isLoadingPrompts.value = false
  }
}

async function resetCurrentPrompt(): Promise<void> {
  if (isResettingPrompt.value) return
  const promptType = currentPromptType.value
  isResettingPrompt.value = true
  try {
    const confirmed = await confirmProductAction({
      title: '重置提示词',
      message: '确定要重置为默认提示词吗？当前编辑的内容将丢失。',
      confirmText: '重置',
      cancelText: '取消',
      tone: 'danger',
    })
    if (!confirmed) return
    const content = await insightApi.resetDefaultPrompt(promptType)
    defaultPrompts.value[promptType] = content
    customPrompts.value[promptType] = content
    if (currentPromptType.value === promptType) currentPromptContent.value = content
    emitPrompts()
    emit('showMessage', '已重置为默认提示词', 'success')
  } catch (error) {
    emit(
      'showMessage',
      '重置失败: ' + (error instanceof Error ? error.message : '网络错误'),
      'error',
    )
  } finally {
    isResettingPrompt.value = false
  }
}

async function copyPromptToClipboard(): Promise<void> {
  const copied = await copyTextToClipboard(currentPromptContent.value)
  emit('showMessage', copied ? '已复制到剪贴板' : '复制失败', copied ? 'success' : 'error')
}

async function savePromptToLibrary(): Promise<void> {
  if (isSavingPrompt.value) return
  const content = currentPromptContent.value
  const promptType = currentPromptType.value
  if (!content.trim()) {
    emit('showMessage', '提示词内容不能为空', 'error')
    return
  }

  isSavingPrompt.value = true
  try {
    const name = await requestProductTextInput({
      title: '保存提示词',
      message: '请输入提示词名称：',
      placeholder: '提示词名称',
      confirmText: '保存',
      cancelText: '取消',
    })
    if (!name?.trim()) return

    const saved = await insightApi.savePromptToLibrary({
      name: name.trim(),
      type: promptType,
      content,
    })
    savedPromptsLibrary.value.push(saved)
    emit('showMessage', '提示词已保存到库', 'success')
  } catch (error) {
    emit(
      'showMessage',
      '保存失败: ' + (error instanceof Error ? error.message : '网络错误'),
      'error',
    )
  } finally {
    isSavingPrompt.value = false
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
  if (deletingPromptIds.value.has(promptId)) return
  deletingPromptIds.value = new Set([...deletingPromptIds.value, promptId])

  try {
    const confirmed = await confirmProductAction({
      title: '删除提示词',
      message: '确定要删除这个提示词吗？',
      confirmText: '删除',
      cancelText: '取消',
      tone: 'danger',
    })
    if (!confirmed) return

    await insightApi.deletePromptFromLibrary(promptId)
    savedPromptsLibrary.value = savedPromptsLibrary.value.filter(p => p.id !== promptId)
    emit('showMessage', '提示词已删除', 'success')
  } catch (error) {
    emit(
      'showMessage',
      '删除失败: ' + (error instanceof Error ? error.message : '网络错误'),
      'error',
    )
  } finally {
    const remaining = new Set(deletingPromptIds.value)
    remaining.delete(promptId)
    deletingPromptIds.value = remaining
  }
}

function exportAllPrompts(): void {
  customPrompts.value[currentPromptType.value] = currentPromptContent.value

  const exportData = {
    version: 2,
    exportedAt: new Date().toISOString(),
    prompts: Object.fromEntries(
      insightApi.INSIGHT_PROMPT_TYPES.map(type => [type, customPrompts.value[type] ?? '']),
    ),
    library: savedPromptsLibrary.value.map(({ name, type, content }) => ({
      name,
      type,
      content,
    })),
  }

  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
  triggerBlobDownload(blob, `manga-insight-prompts-${new Date().toISOString().slice(0, 10)}.json`)

  emit('showMessage', '提示词已导出', 'success')
}

function triggerImportPrompts(): void {
  promptsImportInput.value?.click()
}

async function handlePromptsFileImport(files: File[]): Promise<void> {
  if (isImportingPrompts.value) return
  const file = files[0]
  if (!file) return

  isImportingPrompts.value = true
  try {
    const importData = parsePromptsImport(await file.text())
    const library = importData.library.length > 0
      ? await insightApi.importPromptsLibrary(importData.library)
      : savedPromptsLibrary.value

    customPrompts.value = { ...customPrompts.value, ...importData.prompts }
    currentPromptContent.value = customPrompts.value[currentPromptType.value] ?? ''
    savedPromptsLibrary.value = library
    emitPrompts()

    emit('showMessage', '提示词导入成功', 'success')
  } catch (error) {
    emit(
      'showMessage',
      '导入失败: ' + (error instanceof Error ? error.message : '请检查文件格式'),
      'error',
    )
  } finally {
    isImportingPrompts.value = false
    promptsImportInput.value?.clear()
  }
}

function parsePromptsImport(text: string): {
  prompts: Record<PromptType, string>
  library: SavedPromptInput[]
} {
  const value: unknown = JSON.parse(text)
  if (!isObject(value)) throw new Error('文件根节点必须是对象')
  requireExactKeys(value, ['version', 'exportedAt', 'prompts', 'library'], '文件')
  if (value.version !== 2) throw new Error('提示词文件版本必须为 2')
  if (typeof value.exportedAt !== 'string') throw new Error('exportedAt 必须是字符串')
  if (!isObject(value.prompts)) throw new Error('prompts 必须是对象')
  const promptValues = value.prompts
  requireExactKeys(promptValues, [...insightApi.INSIGHT_PROMPT_TYPES], 'prompts')

  const prompts = Object.fromEntries(
    insightApi.INSIGHT_PROMPT_TYPES.map(type => {
      const content = promptValues[type]
      if (typeof content !== 'string') throw new Error(`prompts.${type} 必须是字符串`)
      return [type, content]
    }),
  ) as Record<PromptType, string>

  if (!Array.isArray(value.library)) throw new Error('library 必须是数组')
  const names = new Set<string>()
  const library = value.library.map((item, index): SavedPromptInput => {
    if (!isObject(item)) throw new Error(`library[${index}] 必须是对象`)
    requireExactKeys(item, ['name', 'type', 'content'], `library[${index}]`)
    if (typeof item.name !== 'string' || !item.name.trim() || item.name.trim().length > 200) {
      throw new Error(`library[${index}].name 必须包含 1-200 个字符`)
    }
    if (!insightApi.isInsightPromptType(item.type)) {
      throw new Error(`library[${index}].type 不是可用的 Insight 提示词类型`)
    }
    if (typeof item.content !== 'string' || !item.content.trim()) {
      throw new Error(`library[${index}].content 不能为空`)
    }
    const identity = `${item.type}\u0000${item.name.trim()}`
    if (names.has(identity)) throw new Error(`library[${index}] 与前面的提示词重名`)
    names.add(identity)
    return { name: item.name.trim(), type: item.type, content: item.content }
  })
  return { prompts, library }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function requireExactKeys(
  value: Record<string, unknown>,
  expected: string[],
  label: string,
): void {
  const actual = Object.keys(value).sort()
  const required = [...expected].sort()
  if (actual.length !== required.length || actual.some((key, index) => key !== required[index])) {
    throw new Error(`${label}字段必须为：${expected.join('、')}`)
  }
}

watch(currentPromptType, (newType, previousType) => {
  if (previousType) {
    customPrompts.value[previousType] = currentPromptContent.value
  }
  if (newType) {
    currentPromptContent.value = customPrompts.value[newType] ?? defaultPrompts.value[newType] ?? ''
  }
})

watch([currentPromptType, currentPromptContent], () => {
  emitPrompts()
})

function collectDraftPrompts(): Record<string, string> {
  customPrompts.value[currentPromptType.value] = currentPromptContent.value
  return customPrompts.value
}

function emitPrompts(): void {
  emit('update:prompts', { ...collectDraftPrompts() })
}

function refreshDraftFromStore(): void {
  customPrompts.value = { ...insightStore.config.prompts }
  currentPromptContent.value = customPrompts.value[currentPromptType.value]
    ?? defaultPrompts.value[currentPromptType.value]
    ?? ''
  emitPrompts()
}

watch(() => props.syncRequestId, () => {
  refreshDraftFromStore()
})

async function initialize(): Promise<void> {
  await Promise.all([loadDefaultPrompts(), loadPromptsLibrary()])
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
      <UiButton
        variant="secondary"
        :disabled="isResettingPrompt"
        @click="resetCurrentPrompt"
        title="重置为默认"
        size="sm"
      >
        <UiIcon v-if="!isResettingPrompt" name="refresh" size="14" />
        <span>{{ isResettingPrompt ? '重置中...' : '重置' }}</span>
      </UiButton>
      <UiButton variant="secondary" @click="copyPromptToClipboard" title="复制到剪贴板" size="sm">
        <UiIcon name="copy" size="14" />
        <span>复制</span>
      </UiButton>
      <UiButton
        variant="primary"
        :disabled="isSavingPrompt"
        @click="savePromptToLibrary"
        title="保存到库"
        size="sm"
      >
        <UiIcon name="save" size="14" />
        <span>{{ isSavingPrompt ? '保存中...' : '保存到库' }}</span>
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
          <UiButton
            variant="secondary"
            :disabled="isLoadingPrompts || isImportingPrompts"
            @click="triggerImportPrompts"
            title="导入提示词"
            size="sm"
          >
            <UiIcon name="upload" size="14" />
            <span>{{ isImportingPrompts ? '导入中...' : '导入' }}</span>
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
                :disabled="deletingPromptIds.has(promptItem.id) || isImportingPrompts"
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
              :disabled="deletingPromptIds.has(promptItem.id) || isImportingPrompts"
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
