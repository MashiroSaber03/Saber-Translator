<template>
  <div class="prompt-library">
    <UiPanel variant="settings">
      <template #title>提示词管理</template>
      <UiField class="ui-settings-field">
        <label for="promptType">提示词类型:</label>
        <CustomSelect
          v-model="selectedType"
          :options="promptTypeOptions"
          @change="handleTypeChange"
        />
      </UiField>

      <UiField v-if="supportsModeSwitch" class="ui-settings-field">
        <label for="promptMode">提示词模式:</label>
        <CustomSelect
          :model-value="selectedMode"
          :options="availablePromptModeOptions"
          @change="(v: string | number) => { selectedMode = String(v); handleModeChange() }"
        />
        <span class="mode-hint">{{ modeHint }}</span>
      </UiField>
    </UiPanel>

    <UiPanel variant="settings">
      <template #title>已保存的提示词</template>
      <div v-if="isLoading" class="loading-hint">加载中...</div>
      <div v-else-if="promptList.length === 0" class="empty-hint">暂无保存的提示词</div>
      <div v-else class="prompt-list">
        <div v-for="prompt in promptList" :key="prompt.name" class="prompt-item" :class="{ active: selectedPrompt === prompt.name }">
          <UiButton
            variant="toolbar"
            type="button"
            class="prompt-select"
            :aria-label="`选择提示词：${prompt.name}`"
            @click="selectPrompt(prompt.name)"
          >
            <span class="prompt-name">{{ prompt.name }}</span>
          </UiButton>
          <div class="prompt-actions">
            <UiButton
              variant="secondary"
              class="prompt-actions__load"
              :aria-label="`加载提示词：${prompt.name}`"
              @click="loadPrompt(prompt.name)"
              title="加载到编辑器"
              size="sm"
            >
              📥
            </UiButton>
            <UiButton
              variant="danger"
              class="prompt-actions__delete"
              :aria-label="`删除提示词：${prompt.name}`"
              @click="deletePrompt(prompt.name)"
              title="删除"
              :disabled="prompt.name === 'default'"
              size="sm"
            >
              🗑️
            </UiButton>
          </div>
        </div>
      </div>
    </UiPanel>

    <UiPanel variant="settings">
      <template #title>提示词编辑</template>
      <UiField class="ui-settings-field">
        <label for="promptName">提示词名称:</label>
        <UiInput type="text" id="promptName" v-model="editingName" placeholder="请输入提示词名称" />
      </UiField>
      <UiField class="ui-settings-field">
        <label for="promptContent">提示词内容:</label>
        <UiTextarea id="promptContent" v-model="editingContent" rows="8" placeholder="请输入提示词内容" />
      </UiField>
      <div class="prompt-editor-actions">
        <UiButton variant="primary" @click="savePrompt" :disabled="!editingName || !editingContent">保存提示词</UiButton>
      </div>
    </UiPanel>
  </div>
</template>

<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiPanel from '@/components/ui/UiPanel.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { ref, computed, onMounted } from 'vue'
import { configApi } from '@/api/config'
import { useSettingsStore } from '@/stores/settings'
import { useToast } from '@/utils/toast'
import CustomSelect from '@/components/common/CustomSelect.vue'

const promptTypeOptions = [
  { label: '翻译提示词', value: 'translate' },
  { label: '文本框提示词', value: 'textbox' },
  { label: 'AI视觉OCR提示词', value: 'ai_vision_ocr' },
  { label: '高质量翻译提示词', value: 'hq_translate' },
  { label: '校对提示词', value: 'proofreading' }
]

const translatePromptModeOptions = [
  { label: '普通模式', value: 'normal' },
  { label: 'JSON格式模式', value: 'json' }
]

const aiVisionPromptModeOptions = [
  { label: '普通模式', value: 'normal' },
  { label: 'JSON格式模式', value: 'json' },
  { label: 'OCR模型提示词', value: 'paddleocr_vl' }
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
  isLoading.value = true
  try {
    let result
    if (selectedType.value === 'textbox') {
      result = await configApi.getTextboxPrompts()
    } else {
      result = await configApi.getPrompts(selectedType.value)
    }
    const names = result.prompt_names || []
    promptList.value = names.map(name => ({ name }))
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '加载提示词列表失败'
    toast.error(errorMessage)
  } finally {
    isLoading.value = false
  }
}

async function selectPrompt(name: string) {
  selectedPrompt.value = name
  editingName.value = name
  await loadPrompt(name)
}

async function loadPrompt(name: string) {
  try {
    let result
    if (selectedType.value === 'textbox') {
      result = await configApi.getTextboxPromptContent(name)
    } else {
      result = await configApi.getPromptContent(selectedType.value, name)
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

function handleModeChange() {
  if (selectedType.value === 'translate') {
    settingsStore.updateTranslationService({ forceJsonOutput: selectedMode.value === 'json' })
  } else if (selectedType.value === 'ai_vision_ocr') {
    settingsStore.updateAiVisionOcr({
      forceJsonOutput: selectedMode.value === 'json',
      promptMode: selectedMode.value
    })
  }
  
  const modeLabel = selectedMode.value === 'json'
    ? 'JSON格式'
    : selectedMode.value === 'paddleocr_vl'
      ? 'OCR模型提示词'
      : '普通'
  toast.info(`已切换到${modeLabel}模式`)
}

onMounted(() => {
  selectedMode.value = getTranslationPromptMode()
  loadPromptList()
})
</script>

<style scoped>
.prompt-library {
  --ui-button-sm-padding: 4px 8px;
  --ui-button-sm-font-size: 12px;
  --ui-button-danger-background: transparent;
  --ui-button-danger-border: none;
  --ui-button-danger-shadow: none;
  --ui-button-danger-hover-background: transparent;
  --ui-button-danger-hover-shadow: none;
}

.prompt-list {
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid var(--color-border-muted);
  border-radius: 4px;
}

.prompt-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  border-bottom: 1px solid var(--color-border-muted);
}

.prompt-item:last-child {
  border-bottom: none;
}

.prompt-item:hover {
  background: var(--color-surface-hover);
}

.prompt-item.active {
  background: var(--color-surface-subtle);
}

.prompt-name {
  display: block;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.prompt-select {
  flex: 1;
  min-width: 0;
  justify-content: flex-start;
  padding: 0;
  color: var(--color-text-default);
  text-align: left;
}

.prompt-actions {
  display: flex;
  gap: 4px;
}

.prompt-editor-actions {
  display: flex;
  gap: 10px;
  margin-top: 10px;
}

.loading-hint,
.empty-hint {
  padding: 20px;
  text-align: center;
  color: var(--color-text-supporting);
}

.prompt-actions__delete:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.mode-hint {
  font-size: 12px;
  color: var(--color-text-supporting);
  margin-left: 10px;
}
</style>
