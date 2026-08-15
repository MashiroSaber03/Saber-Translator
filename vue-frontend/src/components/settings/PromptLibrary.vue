<template>
  <div class="prompt-library">
    <ProductFormSection>
      <template #title>提示词管理</template>
      <UiField variant="settings" label="提示词类型" control-id="promptType">
        <UiSelect
          id="promptType"
          :model-value="selectedType"
          :options="promptTypeOptions"
          :disabled="isMutating"
          @change="handleTypeSelect"
        />
      </UiField>

      <UiField
        v-if="supportsModeSwitch"
        variant="settings"
        label="提示词模式"
        control-id="promptMode"
        :hint="modeHint"
      >
        <UiSelect
          id="promptMode"
          :model-value="selectedMode"
          :options="availablePromptModeOptions"
          :disabled="isMutating"
          @change="handleModeSelect"
        />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>已保存的提示词</template>
      <ProductStatusBanner
        v-if="isLoading"
        tone="neutral"
        icon-name="refresh"
        title="正在加载提示词"
        aria-live="polite"
      >
        正在同步已保存的提示词列表。
      </ProductStatusBanner>
      <ProductStatusBanner
        v-else-if="loadError"
        tone="danger"
        icon-name="alert-triangle"
        title="提示词加载失败"
        aria-live="polite"
      >
        {{ loadError }}
        <template #actions>
          <UiButton variant="secondary" size="sm" @click="loadPromptList">
            重试
          </UiButton>
        </template>
      </ProductStatusBanner>
      <ProductStatusBanner
        v-else-if="promptList.length === 0"
        tone="neutral"
        icon-name="file-text"
        title="暂无保存的提示词"
      >
        保存后的提示词会出现在这里。
      </ProductStatusBanner>
      <div v-else class="prompt-library__list">
        <div
          v-for="prompt in promptList"
          :key="prompt.id"
          class="prompt-library__item"
          :class="{ 'prompt-library__item--active': selectedPromptId === prompt.id }"
        >
          <UiButton
            variant="toolbar"
            type="button"
            class="prompt-library__select-action"
            :aria-label="`选择提示词：${prompt.name}`"
            :aria-pressed="selectedPromptId === prompt.id"
            :disabled="isMutating"
            @click="selectPrompt(prompt)"
          >
            <span class="prompt-library__name">{{ prompt.name }}</span>
          </UiButton>
          <div class="prompt-library__actions">
            <UiIconButton
              class="prompt-library__load-action"
              :label="`加载提示词：${prompt.name}`"
              variant="soft"
              size="sm"
              :disabled="isMutating"
              @click="loadPrompt(prompt)"
            >
              <UiIcon name="download" />
            </UiIconButton>
            <UiIconButton
              variant="danger"
              class="prompt-library__delete-action"
              :label="`删除提示词：${prompt.name}`"
              size="sm"
              @click="deletePrompt(prompt)"
              :disabled="isMutating || prompt.isFactoryDefault"
            >
              <UiIcon name="trash" />
            </UiIconButton>
          </div>
        </div>
      </div>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>提示词编辑</template>
      <UiField variant="settings" label="提示词名称" control-id="promptName">
        <UiInput
          type="text"
          id="promptName"
          v-model="editingName"
          placeholder="请输入提示词名称"
          :disabled="isMutating"
        />
      </UiField>
      <UiField variant="settings" label="提示词内容" control-id="promptContent">
        <UiTextarea
          id="promptContent"
          v-model="editingContent"
          rows="8"
          variant="panel"
          placeholder="请输入提示词内容"
          :disabled="isMutating"
        />
      </UiField>
      <ProductActionRow aria-label="提示词编辑操作" justify="start">
        <UiButton
          variant="primary"
          @click="savePrompt"
          :disabled="isMutating || !editingName.trim()"
        >
          {{ isMutating ? '处理中…' : '保存提示词' }}
        </UiButton>
      </ProductActionRow>
    </ProductFormSection>
  </div>
</template>

<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import { ref, computed, onBeforeUnmount, onMounted } from 'vue'
import {
  createV2Prompt,
  deleteV2Prompt,
  listV2Prompts,
  updateV2Prompt,
  type V2Prompt,
} from '@/api/v2/settings'
import { useSettingsStore } from '@/stores/settings'
import { useToast } from '@/utils/toast'
import UiSelect from '@/components/ui/UiSelect.vue'
import { confirmProductAction } from '@/composables/useProductConfirm'

const promptTypeOptions = [
  { label: '翻译提示词', value: 'translate' },
  { label: '文本框提示词', value: 'textbox' },
  { label: 'AI视觉OCR提示词', value: 'ai_vision_ocr' },
  { label: '高质量翻译提示词', value: 'hq_translate' },
  { label: '校对提示词', value: 'proofreading' },
] satisfies Array<{ label: string; value: V2Prompt['type'] }>

const translatePromptModeOptions = [
  { label: '普通模式', value: 'normal' },
  { label: 'JSON格式模式', value: 'json' },
]

const aiVisionPromptModeOptions = [
  { label: '普通模式', value: 'normal' },
  { label: 'JSON格式模式', value: 'json' },
  { label: 'OCR模型提示词', value: 'paddleocr_vl' },
]

const toast = useToast()
const settingsStore = useSettingsStore()

const selectedType = ref<V2Prompt['type']>('translate')
const promptList = ref<V2Prompt[]>([])
const selectedPromptId = ref('')
const editingName = ref('')
const editingContent = ref('')
const isLoading = ref(false)
const loadError = ref('')
const isMutating = ref(false)
const selectedMode = ref<'normal' | 'json' | 'paddleocr_vl'>('normal')
let promptListRequestId = 0
let isMounted = true

const supportsModeSwitch = computed(() => {
  return selectedType.value === 'translate' || selectedType.value === 'ai_vision_ocr'
})

const availablePromptModeOptions = computed(() => {
  return selectedType.value === 'ai_vision_ocr'
    ? aiVisionPromptModeOptions
    : translatePromptModeOptions
})

const modeHint = computed(() => {
  if (selectedType.value === 'ai_vision_ocr' && selectedMode.value === 'paddleocr_vl') {
    return '适用于 PaddleOCR-VL、GLM-OCR 等专用 OCR 模型'
  }
  if (selectedMode.value === 'json') {
    return '适用于需要结构化输出的场景'
  }
  return '适用于普通翻译场景'
})

function getTranslationPromptMode(): 'normal' | 'json' {
  const translationSettings = settingsStore.settings.translation
  const forceJsonOutput = translationSettings.openaiOptions.request.forceJsonOutput

  return forceJsonOutput ? 'json' : 'normal'
}

async function loadPromptList() {
  const requestId = ++promptListRequestId
  const promptType = selectedType.value
  isLoading.value = true
  loadError.value = ''
  try {
    const result = await listV2Prompts(promptType)
    if (!isMounted || requestId !== promptListRequestId || selectedType.value !== promptType) {
      return
    }
    promptList.value = result
  } catch (error: unknown) {
    if (!isMounted || requestId !== promptListRequestId || selectedType.value !== promptType) {
      return
    }
    const errorMessage = error instanceof Error ? error.message : '加载提示词列表失败'
    loadError.value = errorMessage
    toast.error(errorMessage)
  } finally {
    if (isMounted && requestId === promptListRequestId && selectedType.value === promptType) {
      isLoading.value = false
    }
  }
}

function upsertPrompt(prompt: V2Prompt) {
  const next = promptList.value.filter(item => item.id !== prompt.id)
  next.push(prompt)
  promptList.value = next.sort((left, right) => left.name.localeCompare(right.name, 'zh-CN'))
}

function selectPrompt(prompt: V2Prompt) {
  loadPrompt(prompt)
}

function loadPrompt(prompt: V2Prompt) {
  editingName.value = prompt.name
  editingContent.value = prompt.content
  selectedPromptId.value = prompt.id
  toast.success('已加载提示词')
}

async function savePrompt() {
  if (isMutating.value) return
  if (!editingName.value.trim()) {
    toast.warning('请输入提示词名称')
    return
  }
  isMutating.value = true
  try {
    const selected = promptList.value.find(prompt => prompt.id === selectedPromptId.value)
    let saved: V2Prompt
    if (selected) {
      saved = await updateV2Prompt({
        ...selected,
        name: editingName.value,
        content: editingContent.value,
      })
    } else {
      saved = await createV2Prompt(
        selectedType.value,
        editingName.value,
        editingContent.value,
      )
    }
    if (!loadError.value) upsertPrompt(saved)
    toast.success('提示词保存成功')
    selectedPromptId.value = ''
    editingName.value = ''
    editingContent.value = ''
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '保存提示词失败'
    toast.error(errorMessage)
  } finally {
    isMutating.value = false
  }
}

async function deletePrompt(prompt: V2Prompt) {
  if (isMutating.value) return
  if (prompt.isFactoryDefault) {
    toast.warning('默认提示词不能删除')
    return
  }

  isMutating.value = true
  try {
    const confirmed = await confirmProductAction({
      title: '删除提示词',
      message: `确定要删除提示词“${prompt.name}”吗？此操作无法撤销。`,
      confirmText: '删除',
      cancelText: '取消',
      tone: 'danger',
    })
    if (!confirmed) return

    await deleteV2Prompt(prompt.id)
    promptList.value = promptList.value.filter(item => item.id !== prompt.id)
    toast.success('提示词删除成功')
    if (selectedPromptId.value === prompt.id) {
      selectedPromptId.value = ''
      editingName.value = ''
      editingContent.value = ''
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '删除提示词失败'
    toast.error(errorMessage)
  } finally {
    isMutating.value = false
  }
}

function handleTypeChange() {
  promptList.value = []
  loadError.value = ''
  selectedPromptId.value = ''
  editingName.value = ''
  editingContent.value = ''
  if (selectedType.value === 'translate') {
    selectedMode.value = getTranslationPromptMode()
  } else if (selectedType.value === 'ai_vision_ocr') {
    selectedMode.value = settingsStore.settings.aiVisionOcr.promptMode
  } else {
    selectedMode.value = 'normal'
  }

  loadPromptList()
}

function handleTypeSelect(value: string | number) {
  if (typeof value !== 'string') return
  const option = promptTypeOptions.find(candidate => candidate.value === value)
  if (!option) return
  selectedType.value = option.value
  handleTypeChange()
}

function handleModeChange() {
  if (selectedType.value === 'translate') {
    if (selectedMode.value === 'paddleocr_vl') return
    settingsStore.setTranslatePromptMode(selectedMode.value === 'json')
  } else if (selectedType.value === 'ai_vision_ocr') {
    settingsStore.setAiVisionOcrPromptMode(selectedMode.value)
  }

  const modeLabel =
    selectedMode.value === 'json'
      ? 'JSON格式'
      : selectedMode.value === 'paddleocr_vl'
        ? 'OCR模型提示词'
        : '普通'
  toast.info(`已切换到${modeLabel}模式`)
}

function handleModeSelect(value: string | number) {
  if (value === 'normal' || value === 'json' || value === 'paddleocr_vl') {
    selectedMode.value = value
    handleModeChange()
  }
}

onMounted(() => {
  selectedMode.value = getTranslationPromptMode()
  loadPromptList()
})

onBeforeUnmount(() => {
  isMounted = false
  promptListRequestId += 1
})
</script>

<style scoped>
.prompt-library__list {
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid var(--color-border-muted);
  border-radius: 4px;
}

.prompt-library__item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  border-bottom: 1px solid var(--color-border-muted);
}

.prompt-library__item:last-child {
  border-bottom: none;
}

.prompt-library__item:hover {
  background: var(--color-surface-hover);
}

.prompt-library__item--active {
  background: var(--color-surface-subtle);
}

.prompt-library__name {
  display: block;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.prompt-library__select-action {
  flex: 1;
  min-width: 0;
  justify-content: flex-start;
  padding: 0;
  color: var(--color-text-default);
  text-align: left;
}

.prompt-library__actions {
  display: flex;
  gap: 4px;
}
</style>
