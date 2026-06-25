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
import { ref, onBeforeUnmount, onMounted, watch } from 'vue'
import { configApi, type PromptContentResponse } from '@/api/config'
import type { PromptListResponse } from '@/types'

const props = defineProps<{
  promptType: string
}>()

const emit = defineEmits<{
  (e: 'select', content: string, name: string): void
}>()

const promptList = ref<{ name: string }[]>([])
const isLoading = ref(false)
let promptListRequestId = 0
let promptContentRequestId = 0
let isMounted = true

async function loadPromptList() {
  const requestId = ++promptListRequestId
  const promptType = props.promptType
  isLoading.value = true
  try {
    const result: PromptListResponse = promptType === 'textbox'
      ? await configApi.getTextboxPrompts()
      : await configApi.getPrompts(promptType)
    if (!isMounted || requestId !== promptListRequestId || props.promptType !== promptType) {
      return
    }
    const names = result.prompt_names || []
    promptList.value = names.map(name => ({ name }))
  } catch {
    if (!isMounted || requestId !== promptListRequestId || props.promptType !== promptType) {
      return
    }
    promptList.value = []
  } finally {
    if (isMounted && requestId === promptListRequestId && props.promptType === promptType) {
      isLoading.value = false
    }
  }
}

async function handleSelect(name: string) {
  const requestId = ++promptContentRequestId
  const promptType = props.promptType
  try {
    const result: PromptContentResponse = promptType === 'textbox'
      ? await configApi.getTextboxPromptContent(name)
      : await configApi.getPromptContent(promptType, name)
    if (!isMounted || requestId !== promptContentRequestId || props.promptType !== promptType) {
      return
    }
    if (result.prompt_content) {
      emit('select', result.prompt_content, name)
    }
  } catch {
    // Prompt selection is optional; the picker remains available for another choice.
  }
}

watch(() => props.promptType, () => {
  loadPromptList()
})

onMounted(() => {
  loadPromptList()
})

onBeforeUnmount(() => {
  isMounted = false
  promptListRequestId += 1
  promptContentRequestId += 1
})

defineExpose({ refresh: loadPromptList })
</script>

<style scoped>
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
  background: var(--color-action-primary);
  color: var(--color-text-inverse);
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
