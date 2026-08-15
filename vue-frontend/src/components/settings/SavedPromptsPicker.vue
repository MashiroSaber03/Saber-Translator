<template>
  <div class="saved-prompts-picker">
    <ProductChipList
      label="快速选择"
      aria-label="已保存提示词"
      :items="promptChipItems"
      @select="handleSelect"
    />
  </div>
</template>

<script setup lang="ts">
import ProductChipList, { type ProductChipItem } from '@/components/product/ProductChipList.vue'
import { computed, ref, onBeforeUnmount, onMounted, watch } from 'vue'
import { listV2Prompts, type V2Prompt } from '@/api/v2/settings'

const props = defineProps<{
  promptType: V2Prompt['type']
}>()

const emit = defineEmits<{
  (e: 'select', content: string, name: string): void
}>()

const promptList = ref<V2Prompt[]>([])
const isLoading = ref(false)
const loadError = ref(false)
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

  if (loadError.value) {
    return [{
      id: 'error',
      label: '加载失败，点击重试',
      iconName: 'alert-triangle',
      interactive: true,
      tone: 'danger',
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
    id: prompt.id,
    label: prompt.name,
    iconName: 'file-text',
    interactive: true,
    tone: 'neutral',
  }))
})
let promptListRequestId = 0
let isMounted = true

async function loadPromptList() {
  const requestId = ++promptListRequestId
  const promptType = props.promptType
  isLoading.value = true
  loadError.value = false
  try {
    const result = await listV2Prompts(promptType)
    if (!isMounted || requestId !== promptListRequestId || props.promptType !== promptType) {
      return
    }
    promptList.value = result
  } catch {
    if (!isMounted || requestId !== promptListRequestId || props.promptType !== promptType) {
      return
    }
    loadError.value = true
  } finally {
    if (isMounted && requestId === promptListRequestId && props.promptType === promptType) {
      isLoading.value = false
    }
  }
}

function handleSelect(promptId: string | number) {
  if (typeof promptId !== 'string') return
  if (promptId === 'error') {
    void loadPromptList()
    return
  }
  const prompt = promptList.value.find(item => item.id === promptId)
  if (prompt) emit('select', prompt.content, prompt.name)
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
