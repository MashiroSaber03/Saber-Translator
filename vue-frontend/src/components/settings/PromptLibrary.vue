<template>
  <div class="prompt-library">
    <!-- 提示词类型选择 -->
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

      <!-- 提示词模式切换（仅翻译和AI视觉OCR支持） -->
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

    <!-- 已保存的提示词列表 -->
    <UiPanel variant="settings">
      <template #title>已保存的提示词</template>
      <div v-if="isLoading" class="loading-hint">加载中...</div>
      <div v-else-if="promptList.length === 0" class="empty-hint">暂无保存的提示词</div>
      <div v-else class="prompt-list">
        <div v-for="prompt in promptList" :key="prompt.name" class="prompt-item" :class="{ active: selectedPrompt === prompt.name }" @click="selectPrompt(prompt.name)">
          <span class="prompt-name">{{ prompt.name }}</span>
          <div class="prompt-actions">
            <UiButton variant="secondary" @click.stop="loadPrompt(prompt.name)" title="加载到编辑器" size="sm">📥</UiButton>
            <UiButton variant="danger" @click.stop="deletePrompt(prompt.name)" title="删除" :disabled="prompt.name === 'default'" size="sm">
              🗑️
            </UiButton>
          </div>
        </div>
      </div>
    </UiPanel>

    <!-- 提示词编辑器 -->
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
/**
 * 提示词管理组件
 * 管理各类提示词的保存、加载和删除
 * 支持提示词模式切换（翻译: 普通/JSON；AI视觉OCR: 普通/JSON/OCR模型）
 */
import { ref, computed, onMounted } from 'vue'
import { configApi } from '@/api/config'
import { useSettingsStore } from '@/stores/settingsStore'
import { useToast } from '@/utils/toast'
import CustomSelect from '@/components/common/CustomSelect.vue'

/** 提示词类型选项 */
const promptTypeOptions = [
  { label: '翻译提示词', value: 'translate' },
  { label: '文本框提示词', value: 'textbox' },
  { label: 'AI视觉OCR提示词', value: 'ai_vision_ocr' },
  { label: '高质量翻译提示词', value: 'hq_translate' },
  { label: '校对提示词', value: 'proofreading' }
]

/** 提示词模式选项 */
const translatePromptModeOptions = [
  { label: '普通模式', value: 'normal' },
  { label: 'JSON格式模式', value: 'json' }
]

const aiVisionPromptModeOptions = [
  { label: '普通模式', value: 'normal' },
  { label: 'JSON格式模式', value: 'json' },
  { label: 'OCR模型提示词', value: 'paddleocr_vl' }
]

// Toast 和 Store
const toast = useToast()
const settingsStore = useSettingsStore()

// 状态
const selectedType = ref('translate')
const promptList = ref<{ name: string }[]>([])
const selectedPrompt = ref('')
const editingName = ref('')
const editingContent = ref('')
const isLoading = ref(false)
const selectedMode = ref<'normal' | 'json' | 'paddleocr_vl'>('normal')

// ============================================================
// 计算属性
// ============================================================

/** 是否支持模式切换（仅翻译和AI视觉OCR支持） */
const supportsModeSwitch = computed(() => {
  return selectedType.value === 'translate' || selectedType.value === 'ai_vision_ocr'
})

const availablePromptModeOptions = computed(() => {
  return selectedType.value === 'ai_vision_ocr'
    ? aiVisionPromptModeOptions
    : translatePromptModeOptions
})

/** 模式提示文字 */
const modeHint = computed(() => {
  if (selectedType.value === 'ai_vision_ocr' && selectedMode.value === 'paddleocr_vl') {
    return '适用于 PaddleOCR-VL、GLM-OCR 等专用 OCR 模型'
  }
  if (selectedMode.value === 'json') {
    return '适用于需要结构化输出的场景'
  }
  return '适用于普通翻译场景'
})

// ============================================================
// 提示词列表操作
// ============================================================

function getTranslationPromptMode(): 'normal' | 'json' {
  const translationSettings = settingsStore.settings.translation
  const forceJsonOutput =
    translationSettings.openaiOptions?.request?.forceJsonOutput ?? translationSettings.isJsonMode

  return forceJsonOutput ? 'json' : 'normal'
}

/** 加载提示词列表 */
async function loadPromptList() {
  isLoading.value = true
  try {
    let result
    if (selectedType.value === 'textbox') {
      result = await configApi.getTextboxPrompts()
    } else {
      result = await configApi.getPrompts(selectedType.value)
    }
    // 后端返回的是字符串数组，需要转换为对象数组以匹配类型定义
    const names = result.prompt_names || []
    promptList.value = (names as unknown as string[]).map(name => ({ name }))
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '加载提示词列表失败'
    toast.error(errorMessage)
  } finally {
    isLoading.value = false
  }
}

/** 选择提示词（同时加载内容） */
async function selectPrompt(name: string) {
  selectedPrompt.value = name
  editingName.value = name
  // 自动加载提示词内容
  await loadPrompt(name)
}

/** 加载提示词内容 */
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

/** 保存提示词 */
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
    
    // 清空编辑器内容
    editingName.value = ''
    editingContent.value = ''
    
    // 强制刷新列表
    await loadPromptList()
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '保存提示词失败'
    toast.error(errorMessage)
  }
}

/** 删除提示词 */
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

/** 处理类型变化 */
function handleTypeChange() {
  selectedPrompt.value = ''
  editingName.value = ''
  editingContent.value = ''
  
  // 同步模式状态
  if (selectedType.value === 'translate') {
    selectedMode.value = getTranslationPromptMode()
  } else if (selectedType.value === 'ai_vision_ocr') {
    selectedMode.value = settingsStore.settings.aiVisionOcr.promptMode || 'normal'
  } else {
    selectedMode.value = 'normal'
  }
  
  loadPromptList()
}

/** 处理模式切换 */
function handleModeChange() {
  // 更新 store 中的模式状态
  if (selectedType.value === 'translate') {
    settingsStore.updateTranslationService({ isJsonMode: selectedMode.value === 'json' })
  } else if (selectedType.value === 'ai_vision_ocr') {
    settingsStore.updateAiVisionOcr({
      isJsonMode: selectedMode.value === 'json',
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

// ============================================================
// 监听和初始化
// ============================================================

// 初始化
onMounted(() => {
  // 同步初始模式状态
  selectedMode.value = getTranslationPromptMode()
  loadPromptList()
})
</script>

<style scoped>
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
  cursor: pointer;
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
  flex: 1;
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

.ui-button--sm {
  padding: 4px 8px;
  font-size: 12px;
}

.ui-button--danger {
  background: transparent;
  border: none;
}

.ui-button--danger:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.mode-hint {
  font-size: 12px;
  color: var(--color-text-supporting);
  margin-left: 10px;
}
</style>
