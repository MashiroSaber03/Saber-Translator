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
import { ref, onMounted, watch } from 'vue'
import { configApi } from '@/api/config'

const props = defineProps<{
  promptType: string
}>()

const emit = defineEmits<{
  (e: 'select', content: string, name: string): void
}>()

const promptList = ref<{ name: string }[]>([])
const isLoading = ref(false)

async function loadPromptList() {
  isLoading.value = true
  try {
    let result
    if (props.promptType === 'textbox') {
      result = await configApi.getTextboxPrompts()
    } else {
      result = await configApi.getPrompts(props.promptType)
    }
    const names = result.prompt_names || []
    promptList.value = (names as unknown as string[]).map(name => ({ name }))
  } catch {
    promptList.value = []
  } finally {
    isLoading.value = false
  }
}

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
