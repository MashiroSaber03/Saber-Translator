<template>
  <div class="script-panel">
    <h3>📝 生成续写脚本</h3>

    <div class="script-editor" v-if="script">
      <div class="script-header">
        <h4>{{ script.chapter_title }}</h4>
        <span class="script-meta">共 {{ script.page_count }} 页 · {{ script.generated_at }}</span>
      </div>

      <UiTextarea
        v-model="scriptText"
        class="script-textarea"
        rows="15"
        placeholder="脚本将在此显示..."
        @update:model-value="handleScriptInput"
      />

      <div class="script-actions">
        <UiButton variant="secondary" @click="$emit('reset-script')" size="sm">↺ 重置</UiButton>
        <UiButton variant="secondary" :disabled="!script || isSaving" @click="handleSave" size="sm">
          {{ isSaving ? '保存中...' : '💾 保存' }}
        </UiButton>
      </div>
    </div>

    <div v-else class="no-script">
      <p>点击下方按钮生成续写脚本</p>
    </div>

    <div class="reference-config">
      <div class="config-row">
        <label for="script-reference-count">VLM参考图数:</label>
        <UiInput
          id="script-reference-count"
          type="number"
          v-model.number="refCount"
          min="1"
          max="10"
          class="ref-count-input"
        />
        <UiButton
          variant="secondary"
          class="ref-btn"
          @click="openReferenceSelector" size="sm"
        >
          📷 参考图 ({{ getDisplayRefCount() }})
        </UiButton>
      </div>
    </div>

    <UiButton
      variant="primary"
      block
      :disabled="isGenerating"
      @click="handleGenerate"
    >
      {{ isGenerating ? '生成中...' : '🎯 生成脚本' }}
    </UiButton>

    <ReferenceImageSelector
      v-model:visible="selectorVisible"
      mode="script"
      :max-count="refCount"
      :original-images="availableOriginalImages"
      :continuation-images="[]"
      :character-forms="[]"
      :initial-selection="selectedReferenceTokens"
      :book-id="bookId"
      @confirm="handleSelectorConfirm"
    />
  </div>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { onBeforeUnmount, ref, watch } from 'vue'
import type { ChapterScript, MangaImageInfo } from '@/api/continuation'
import { getAvailableImages } from '@/api/continuation'
import ReferenceImageSelector from './ReferenceImageSelector.vue'

const props = defineProps<{
  script: ChapterScript | null
  isGenerating: boolean
  isSaving?: boolean
  bookId: string
}>()

const emit = defineEmits<{
  'generate': [payload: { referenceTokens: string[] | null; referenceImageCount: number }]
  'update-script': [scriptText: string]
  'save-script': []
  'reset-script': []
}>()

const scriptText = ref('')
const refCount = ref(5)
const selectorVisible = ref(false)
const selectedReferenceTokens = ref<string[]>([])
const availableOriginalImages = ref<MangaImageInfo[]>([])
let imageRequestSeq = 0
let isMounted = true

watch(() => props.script?.script_text, (newScriptText) => {
  scriptText.value = newScriptText || ''
}, { immediate: true })

watch(() => props.script, (newScript) => {
  if (!newScript) {
    selectedReferenceTokens.value = []
    selectorVisible.value = false
  }
})

function invalidateAvailableImages(): void {
  imageRequestSeq += 1
}

async function loadAvailableImages(bookId = props.bookId) {
  if (!bookId) return

  const requestId = ++imageRequestSeq

  try {
    const response = await getAvailableImages(bookId, 'script')
    if (!isMounted || requestId !== imageRequestSeq || props.bookId !== bookId) return
    if (response.success && response.original_images) {
      availableOriginalImages.value = response.original_images
    }
  } catch {
    if (!isMounted || requestId !== imageRequestSeq || props.bookId !== bookId) return
    availableOriginalImages.value = []
  }
}

function openReferenceSelector() {
  if (availableOriginalImages.value.length === 0) {
    loadAvailableImages()
  }
  selectorVisible.value = true
}

function handleSelectorConfirm(tokens: string[]) {
  selectedReferenceTokens.value = tokens
}

function getDisplayRefCount(): number {
  if (selectedReferenceTokens.value.length > 0) {
    return selectedReferenceTokens.value.length
  }
  return refCount.value
}

function handleGenerate() {
  const refs = selectedReferenceTokens.value.length > 0 ? selectedReferenceTokens.value : null
  emit('generate', {
    referenceTokens: refs,
    referenceImageCount: refCount.value,
  })
}

function handleScriptInput(value: string) {
  emit('update-script', value)
}

function handleSave() {
  emit('save-script')
}

watch(() => props.bookId, (newBookId) => {
  if (newBookId) {
    loadAvailableImages(newBookId)
    selectedReferenceTokens.value = []
    refCount.value = 5
  } else {
    invalidateAvailableImages()
    refCount.value = 5
    availableOriginalImages.value = []
    selectedReferenceTokens.value = []
    selectorVisible.value = false
  }
}, { immediate: true })

onBeforeUnmount(() => {
  isMounted = false
  invalidateAvailableImages()
})
</script>

<style scoped>
.script-panel {
  --ui-input-padding: 6px 10px;
  --ui-input-border: 1px solid var(--color-border-muted, var(--color-border-default));
  --ui-input-radius: 6px;
  --ui-input-font-size: 14px;
  --ui-input-background: var(--color-surface-input, var(--color-surface-base));
  --ui-input-color: var(--color-text-default);
  --ui-input-focus-border: var(--color-border-brand);
  --ui-input-focus-shadow: var(--color-focus-brand-soft);
  --ui-textarea-padding: 16px;
  --ui-textarea-border: 1px solid var(--color-border-muted, var(--color-border-default));
  --ui-textarea-radius: 8px;
  --ui-textarea-background: var(--color-surface-input, var(--color-surface-base));
  --ui-textarea-color: var(--color-text-default);
  --ui-textarea-font-size: 14px;
  --ui-textarea-line-height: 1.6;
  --ui-textarea-focus-border: var(--color-border-brand);
  --ui-textarea-focus-shadow: var(--color-focus-brand-soft);
  --ui-button-padding: 10px 20px;
  --ui-button-radius: 8px;
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--color-surface-brand);
  --ui-button-primary-color: var(--color-text-inverse);
  --ui-button-primary-shadow: none;
  --ui-button-primary-hover-background: var(--color-surface-brand-strong);
  --ui-button-primary-hover-transform: none;
  --ui-button-primary-hover-shadow: none;
  --ui-button-secondary-background: var(--color-surface-muted);
  --ui-button-secondary-color: var(--color-text-default);
  --ui-button-secondary-border: 1px solid var(--color-border-muted, var(--color-border-default));
  --ui-button-secondary-hover-background: var(--color-surface-hover);
  --ui-button-secondary-hover-border-color: var(--color-border-muted, var(--color-border-default));
  --ui-button-sm-padding: 6px 12px;
  --ui-button-sm-font-size: 13px;
  --ui-button-disabled-opacity: 0.5;

  padding: 24px;
}

.script-panel h3 {
  margin: 0 0 20px;
  font-size: 18px;
  font-weight: 600;
}

.script-editor {
  margin-bottom: 20px;
}

.script-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.script-header h4 {
  margin: 0;
  font-size: 16px;
}

.script-meta {
  font-size: 13px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.script-textarea {
  width: 100%;
}

.script-actions {
  margin-top: 12px;
}

.no-script {
  text-align: center;
  padding: 40px 20px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.no-script p {
  margin: 0;
}

.reference-config {
  margin-bottom: 16px;
  padding: 12px 16px;
  background: var(--color-surface-subtle);
  border-radius: 8px;
}

.config-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.config-row label {
  font-size: 14px;
  color: var(--color-text-default);
  white-space: nowrap;
}

.ref-count-input {
  width: 60px;
  text-align: center;
}

.ref-btn {
  margin-left: auto;
}
</style>
