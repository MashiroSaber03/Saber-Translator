<template>
  <div class="prompt-library">
    <ProductFormSection>
      <template #title>提示词管理</template>
      <UiField variant="settings" label="提示词类型" control-id="promptType">
        <UiSelect
          id="promptType"
          :model-value="selectedType"
          :options="promptTypeOptions"
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
          :key="prompt.name"
          class="prompt-library__item"
          :class="{ 'prompt-library__item--active': selectedPrompt === prompt.name }"
        >
          <UiButton
            variant="toolbar"
            type="button"
            class="prompt-library__select-action"
            :aria-label="`选择提示词：${prompt.name}`"
            :aria-pressed="String(selectedPrompt === prompt.name)"
            @click="selectPrompt(prompt.name)"
          >
            <span class="prompt-library__name">{{ prompt.name }}</span>
          </UiButton>
          <div class="prompt-library__actions">
            <UiIconButton
              class="prompt-library__load-action"
              :label="`加载提示词：${prompt.name}`"
              variant="soft"
              size="sm"
              @click="loadPrompt(prompt.name)"
            >
              <UiIcon name="download" />
            </UiIconButton>
            <UiIconButton
              variant="danger"
              class="prompt-library__delete-action"
              :label="`删除提示词：${prompt.name}`"
              size="sm"
              @click="deletePrompt(prompt.name)"
              :disabled="prompt.name === 'default'"
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
        <UiInput type="text" id="promptName" v-model="editingName" placeholder="请输入提示词名称" />
      </UiField>
      <UiField variant="settings" label="提示词内容" control-id="promptContent">
        <UiTextarea
          id="promptContent"
          v-model="editingContent"
          rows="8"
          variant="panel"
          placeholder="请输入提示词内容"
        />
      </UiField>
      <ProductActionRow aria-label="提示词编辑操作" justify="start">
        <UiButton variant="primary" @click="savePrompt" :disabled="!editingName || !editingContent">
          保存提示词
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
import { configApi, type PromptContentResponse } from '@/api/config'
import type { PromptListResponse } from '@/types'
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
]

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

const selectedType = ref('translate')
const promptList = ref<{ name: string }[]>([])
const selectedPrompt = ref('')
const editingName = ref('')
const editingContent = ref('')
const isLoading = ref(false)
const selectedMode = ref<'normal' | 'json' | 'paddleocr_vl'>('normal')
let promptListRequestId = 0
let promptContentRequestId = 0
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
  try {
    const result: PromptListResponse =
      promptType === 'textbox'
        ? await configApi.getTextboxPrompts()
        : await configApi.getPrompts(promptType)
    if (!isMounted || requestId !== promptListRequestId || selectedType.value !== promptType) {
      return
    }
    const names = result.prompt_names || []
    promptList.value = names.map(name => ({ name }))
  } catch (error: unknown) {
    if (!isMounted || requestId !== promptListRequestId || selectedType.value !== promptType) {
      return
    }
    const errorMessage = error instanceof Error ? error.message : '加载提示词列表失败'
    toast.error(errorMessage)
  } finally {
    if (isMounted && requestId === promptListRequestId && selectedType.value === promptType) {
      isLoading.value = false
    }
  }
}

async function selectPrompt(name: string) {
  selectedPrompt.value = name
  editingName.value = name
  await loadPrompt(name)
}

async function loadPrompt(name: string) {
  const requestId = ++promptContentRequestId
  const promptType = selectedType.value
  try {
    const result: PromptContentResponse =
      promptType === 'textbox'
        ? await configApi.getTextboxPromptContent(name)
        : await configApi.getPromptContent(promptType, name)
    if (!isMounted || requestId !== promptContentRequestId || selectedType.value !== promptType) {
      return
    }
    editingName.value = name
    editingContent.value = result.prompt_content || ''
    selectedPrompt.value = name
    toast.success('已加载提示词')
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '加载提示词内容失败'
    toast.error(errorMessage)
  }
}

async function savePrompt() {
  if (!editingName.value || !editingContent.value) {
    toast.warning('请输入提示词名称和内容')
    return
  }
  try {
    if (selectedType.value === 'textbox') {
      await configApi.saveTextboxPrompt(editingName.value, editingContent.value)
    } else {
      await configApi.savePrompt(selectedType.value, editingName.value, editingContent.value)
    }
    toast.success('提示词保存成功')
    editingName.value = ''
    editingContent.value = ''
    await loadPromptList()
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '保存提示词失败'
    toast.error(errorMessage)
  }
}

async function deletePrompt(name: string) {
  if (name === 'default') {
    toast.warning('默认提示词不能删除')
    return
  }

  const confirmed = await confirmProductAction({
    title: '删除提示词',
    message: `确定要删除提示词“${name}”吗？此操作无法撤销。`,
    confirmText: '删除',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed) {
    return
  }

  try {
    if (selectedType.value === 'textbox') {
      await configApi.deleteTextboxPrompt(name)
    } else {
      await configApi.deletePrompt(selectedType.value, name)
    }
    toast.success('提示词删除成功')
    if (selectedPrompt.value === name) {
      selectedPrompt.value = ''
      editingName.value = ''
      editingContent.value = ''
    }
    await loadPromptList()
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '删除提示词失败'
    toast.error(errorMessage)
  }
}

function handleTypeChange() {
  selectedPrompt.value = ''
  editingName.value = ''
  editingContent.value = ''
  if (selectedType.value === 'translate') {
    selectedMode.value = getTranslationPromptMode()
  } else if (selectedType.value === 'ai_vision_ocr') {
    selectedMode.value = settingsStore.settings.aiVisionOcr.promptMode || 'normal'
  } else {
    selectedMode.value = 'normal'
  }

  loadPromptList()
}

function handleTypeSelect(value: string | number) {
  selectedType.value = String(value)
  handleTypeChange()
}

function handleModeChange() {
  if (selectedType.value === 'translate') {
    settingsStore.updateTranslationService({ forceJsonOutput: selectedMode.value === 'json' })
  } else if (selectedType.value === 'ai_vision_ocr') {
    settingsStore.updateAiVisionOcr({
      forceJsonOutput: selectedMode.value === 'json',
      promptMode: selectedMode.value,
    })
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
  const nextMode = String(value)
  if (nextMode === 'normal' || nextMode === 'json' || nextMode === 'paddleocr_vl') {
    selectedMode.value = nextMode
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
  promptContentRequestId += 1
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
