<template>
  <div class="script-generation-panel">
    <ProductSectionHeader title="生成续写脚本" icon-name="file-text" />

    <div class="script-generation-panel__editor" v-if="script">
      <div class="script-generation-panel__header">
        <h4 class="script-generation-panel__title">{{ script.chapter_title }}</h4>
        <span class="script-generation-panel__meta">共 {{ script.page_count }} 页 · {{ script.generated_at }}</span>
      </div>

      <UiTextarea
        v-model="scriptText"
        class="script-generation-panel__textarea"
        rows="15"
        variant="panel"
        size="lg"
        placeholder="脚本将在此显示..."
        @update:model-value="handleScriptInput"
      />

      <ProductActionRow class="script-generation-panel__actions" aria-label="续写脚本编辑操作">
        <UiButton variant="secondary" @click="$emit('reset-script')" size="sm">
          <UiIcon name="refresh" size="14" />
          <span>重置</span>
        </UiButton>
        <UiButton variant="secondary" :disabled="!script || isSaving" @click="handleSave" size="sm">
          <UiIcon v-if="!isSaving" name="save" size="14" />
          <span>{{ isSaving ? '保存中...' : '保存' }}</span>
        </UiButton>
      </ProductActionRow>
    </div>

    <ProductStatusBanner
      v-else
      class="script-generation-panel__empty-status"
      tone="neutral"
      role="note"
      icon-name="file-text"
      title="暂无脚本"
    >
      点击下方按钮生成续写脚本
    </ProductStatusBanner>

    <div class="script-generation-panel__reference-config">
      <div class="script-generation-panel__reference-row">
        <UiField
          class="script-generation-panel__reference-count-field"
          variant="settings"
          label="VLM参考图数"
          control-id="script-reference-count"
        >
          <UiNumberField
            input-id="script-reference-count"
            v-model="refCount"
            :min="1"
            :max="10"
            size="xs"
            aria-label="VLM参考图数"
          />
        </UiField>
        <ProductActionRow class="script-generation-panel__reference-actions" aria-label="脚本参考图操作">
          <UiButton
            variant="secondary"
            @click="openReferenceSelector" size="sm"
          >
            <UiIcon name="camera" size="14" />
            <span>参考图 ({{ getDisplayRefCount() }})</span>
          </UiButton>
        </ProductActionRow>
      </div>
    </div>

    <UiButton
      variant="primary"
      block
      :disabled="isGenerating"
      @click="handleGenerate"
    >
      <UiIcon v-if="!isGenerating" name="target" size="16" />
      <span>{{ isGenerating ? '生成中...' : '生成脚本' }}</span>
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
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
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
    const response = await getAvailableImages(bookId)
    if (!isMounted || requestId !== imageRequestSeq || props.bookId !== bookId) return
    availableOriginalImages.value = response.original_images
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
.script-generation-panel {
  min-width: 0;
}

.script-generation-panel__editor {
  margin-bottom: 20px;
}

.script-generation-panel__header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 6px 12px;
  min-width: 0;
  margin-bottom: 12px;
}

.script-generation-panel__title {
  margin: 0;
  min-width: 0;
  font-size: 16px;
  overflow-wrap: anywhere;
}

.script-generation-panel__meta {
  min-width: 0;
  font-size: 13px;
  color: var(--color-text-supporting);
  overflow-wrap: anywhere;
}

.script-generation-panel__textarea {
  width: 100%;
}

.script-generation-panel__actions {
  margin-top: 12px;
}

.script-generation-panel__empty-status {
  margin-bottom: 20px;
}

.script-generation-panel__reference-config {
  margin-bottom: 16px;
  padding: 12px 16px;
  background: var(--color-surface-subtle);
  border-radius: 8px;
}

.script-generation-panel__reference-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
}

.script-generation-panel__reference-count-field {
  flex: 1 1 150px;
  max-width: 180px;
  min-width: min(100%, 140px);
  margin-bottom: 0;
}

.script-generation-panel__reference-actions {
  flex: 1 1 220px;
  justify-content: flex-end;
  min-width: min(100%, 180px);
}

@media (--breakpoint-sm-down) {
  .script-generation-panel__reference-actions {
    justify-content: stretch;
  }
}
</style>
