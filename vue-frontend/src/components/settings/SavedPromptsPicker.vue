<template>
  <div class="saved-prompts-picker">
    <ProductChipList
      label="快速选择"
      aria-label="已保存提示词"
      :items="promptChipItems"
      @select="(id) => handleSelect(String(id))"
    />
  </div>
</template>

<script setup lang="ts">
import ProductChipList, { type ProductChipItem } from '@/components/product/ProductChipList.vue'
import { computed, ref, onBeforeUnmount, onMounted, watch } from 'vue'
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
const promptChipItems = computed<ProductChipItem[]>(() => {
  if (isLoading.value) {
    return [{
      id: 'loading',
      label: '加载中...',
      iconName: 'refresh',
      interactive: false,
      tone: 'neutral',
    }]
  }

  if (promptList.value.length === 0) {
    return [{
      id: 'empty',
      label: '暂无保存的提示词',
      iconName: 'file-text',
      interactive: false,
      tone: 'neutral',
    }]
  }

  return promptList.value.map(prompt => ({
    id: prompt.name,
    label: prompt.name,
    iconName: 'file-text',
    interactive: true,
    tone: 'neutral',
  }))
})
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
</script>

<style scoped>
.saved-prompts-picker {
  margin-top: 10px;
  padding: 10px 12px;
  background: var(--color-surface-input);
  border: 1px solid var(--color-border-muted);
  border-radius: 6px;
}

</style>
