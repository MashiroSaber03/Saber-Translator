<template>
  <div class="saved-prompts-picker">
    <span class="picker-label">📑 快速选择:</span>
    <div class="prompts-chips-container">
      <span v-if="isLoading" class="empty-hint">加载中...</span>
      <span v-else-if="promptList.length === 0" class="empty-hint">暂无保存的提示词</span>
      <UiButton
        variant="toolbar"
        v-else
        v-for="prompt in promptList"
        :key="prompt.name"
        type="button"
        class="prompt-chip"
        :title="prompt.name"
        @click="handleSelect(prompt.name)"
      >
        <span class="chip-icon">📝</span>
        {{ prompt.name }}
      </UiButton>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
/**
 * 提示词快速选择器组件
 * 显示用户保存的提示词列表，点击可快速应用到对应的输入框
 * 
 * 使用方式:
 * <SavedPromptsPicker prompt-type="translate" @select="handlePromptSelect" />
 */
import { ref, onMounted, watch } from 'vue'
import { configApi } from '@/api/config'

// Props
const props = defineProps<{
  /** 提示词类型: translate | textbox | ai_vision_ocr | hq_translate | proofreading */
  promptType: string
}>()

// Emits
const emit = defineEmits<{
  /** 选择提示词时触发，返回提示词内容 */
  (e: 'select', content: string, name: string): void
}>()

// 状态
const promptList = ref<{ name: string }[]>([])
const isLoading = ref(false)

/** 加载提示词列表 */
async function loadPromptList() {
  isLoading.value = true
  try {
    let result
    if (props.promptType === 'textbox') {
      result = await configApi.getTextboxPrompts()
    } else {
      result = await configApi.getPrompts(props.promptType)
    }
    // API 返回的可能是 { name: string }[] 或 string[]，统一处理
    // 后端返回的是字符串数组 prompt_names
    const names = result.prompt_names || []
    promptList.value = (names as unknown as string[]).map(name => ({ name }))
  } catch (error) {
    console.error('加载提示词列表失败:', error)
    promptList.value = []
  } finally {
    isLoading.value = false
  }
}

/** 处理选择提示词 */
async function handleSelect(name: string) {
  try {
    let result
    if (props.promptType === 'textbox') {
      result = await configApi.getTextboxPromptContent(name)
    } else {
      result = await configApi.getPromptContent(props.promptType, name)
    }
    if (result.prompt_content) {
      emit('select', result.prompt_content, name)
    }
  } catch (error) {
    console.error('加载提示词内容失败:', error)
  }
}



// 监听 promptType 变化时刷新列表
watch(() => props.promptType, () => {
  loadPromptList()
})

// 初始化加载
onMounted(() => {
  loadPromptList()
})

// 暴露刷新方法
defineExpose({ refresh: loadPromptList })
</script>

<style scoped>
.saved-prompts-picker {
  --saved-prompts-picker-surface-base: #4a90d9;
}

.saved-prompts-picker {
  margin-top: 10px;
  padding: 10px 12px;
  background: var(--color-surface-input, var(--color-surface-subtle));
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 6px;
}

.picker-label {
  font-size: 0.85em;
  color: var(--color-text-supporting, var(--color-text-secondary));
  margin-right: 10px;
  white-space: nowrap;
}

.prompts-chips-container {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
  min-height: 32px;
  align-items: center;
}

.prompt-chip {
  padding: 5px 12px;
  background: var(--color-surface-card, var(--color-surface-base));
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 16px;
  cursor: pointer;
  font-size: 0.85em;
  color: var(--color-text-strong, var(--color-text-default));
  transition: all 0.2s;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.prompt-chip:hover {
  background: var(--color-action-primary, var(--saved-prompts-picker-surface-base));
  color: white;
  border-color: var(--color-action-primary, var(--color-border-info));
}

.chip-icon {
  font-size: 0.9em;
}

.empty-hint {
  font-size: 0.85em;
  color: var(--color-text-supporting, var(--color-text-muted));
  font-style: italic;
}
</style>
