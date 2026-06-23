<template>
  <div class="image-generation-panel">
    <h3>🎨 图片生成与导出</h3>

    <div class="generation-controls">
      <div class="batch-config">
        <div class="config-row">
          <label>画风参考图数量:</label>
          <UiInput
            type="number"
            v-model.number="refCount"
            min="1"
            max="10"
            class="ref-count-input"
          />
          <UiButton
            variant="secondary"
           
            @click="openBatchReferenceSelector"
          >
            📷 选择初始参考图 ({{ getInitialRefCount() }})
          </UiButton>
        </div>
      </div>

      <UiButton
        variant="primary"
        block
       
        :disabled="isGenerating || pages.length === 0"
        @click="handleBatchGenerate" size="lg"
      >
        {{ isGenerating ? '生成中...' : '🚀 批量生成图片' }}
      </UiButton>

      <div v-if="isGenerating" class="progress-bar">
        <div class="progress-fill" :style="{ width: progress + '%' }"></div>
        <span class="progress-text">{{ progress }}%</span>
      </div>
    </div>

    <div class="generated-images">
      <div v-for="page in pages" :key="page.page_number" class="image-card">
        <div class="image-header">
          <h4>页面 {{ page.page_number }}</h4>
          <span class="image-status" :class="page.status">{{ getStatusText(page.status) }}</span>
        </div>

        <div class="image-preview">
          <img
            v-if="page.image_url"
            :src="getImageUrl(page.image_url)"
            :alt="`页面 ${page.page_number}`"
          >
          <div v-else class="no-image">
            <span>{{ page.status === 'generating' ? '⏳' : '📷' }}</span>
            <p>{{ page.status === 'generating' ? '生成中...' : '未生成' }}</p>
          </div>
        </div>

        <div class="story-context">
          <div class="context-block">
            <div class="context-header">
              <label>上一页剧情</label>
              <UiButton
                variant="toolbar"
                v-if="shouldShowStoryToggle(page.page_number, 'continuity')"
                class="context-toggle"
                @click="toggleStorySection(page.page_number, 'continuity')"
              >
                {{ isStorySectionExpanded(page.page_number, 'continuity') ? '收起' : '展开' }}
              </UiButton>
            </div>
            <p
              class="context-text"
              :class="getStoryTextClass(page.page_number, 'continuity', 3)"
            >
              {{ page.continuity_text || '（无）' }}
            </p>
          </div>
          <div class="context-block">
            <div class="context-header">
              <label>本页剧情</label>
              <UiButton
                variant="toolbar"
                v-if="shouldShowStoryToggle(page.page_number, 'story')"
                class="context-toggle"
                @click="toggleStorySection(page.page_number, 'story')"
              >
                {{ isStorySectionExpanded(page.page_number, 'story') ? '收起' : '展开' }}
              </UiButton>
            </div>
            <p
              class="context-text"
              :class="getStoryTextClass(page.page_number, 'story', 3)"
            >
              {{ page.story_text || '（无）' }}
            </p>
          </div>
          <div class="context-block">
            <div class="context-header">
              <label>关键对白</label>
              <UiButton
                variant="toolbar"
                v-if="shouldShowStoryToggle(page.page_number, 'dialogue')"
                class="context-toggle"
                @click="toggleStorySection(page.page_number, 'dialogue')"
              >
                {{ isStorySectionExpanded(page.page_number, 'dialogue') ? '收起' : '展开' }}
              </UiButton>
            </div>
            <p
              class="context-text"
              :class="getStoryTextClass(page.page_number, 'dialogue', 2)"
            >
              {{ page.dialogue_text || '（无）' }}
            </p>
          </div>
        </div>

        <div class="prompt-section">
          <div class="prompt-header">
            <label>📝 最终生图提示词</label>
            <UiButton
              variant="toolbar"
              class="btn-mini"
              @click="togglePromptEdit(page.page_number)"
            >
              {{ editingPromptPage === page.page_number ? '收起' : '编辑' }}
            </UiButton>
          </div>
          <div v-if="editingPromptPage === page.page_number" class="prompt-edit">
            <UiTextarea
              v-model="page.final_prompt"
              rows="8"
              class="prompt-input"
              placeholder="输入最终生图提示词..."
              @input="$emit('prompt-change', page.page_number)"
            />
          </div>
          <div v-else class="prompt-collapsed">
            <p v-if="page.final_prompt" class="prompt-collapsed-hint">默认已折叠，点击“编辑”查看或修改</p>
            <p v-else class="prompt-empty">暂无最终提示词</p>
          </div>
        </div>

        <div class="image-actions">
          <UiButton
            variant="secondary"
           
            :disabled="page.status === 'generating'"
            @click="$emit('regenerate', page.page_number)" size="sm"
          >
            ↺ 重新生成
          </UiButton>
          <UiButton
            variant="secondary"
            v-if="page.previous_url"
           
            @click="$emit('use-previous', page.page_number)" size="sm"
          >
            ◀ 上一版本
          </UiButton>
        </div>
      </div>
    </div>

    <ReferenceImageSelector
      v-model:visible="selectorVisible"
      mode="image"
      :max-count="refCount"
      :original-images="availableOriginalImages"
      :continuation-images="availableContinuationImages"
      :character-forms="availableCharacterForms"
      :initial-selection="batchInitialReferenceTokens"
      :book-id="bookId"
      @confirm="handleSelectorConfirm"
      @cancel="handleSelectorCancel"
    />
  </div>
</template>

<script setup lang="ts">

import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
import { ref, watch, onMounted } from 'vue'
import type { PageContent, MangaImageInfo, CharacterFormInfo } from '@/api/continuation'
import { getAvailableImages } from '@/api/continuation'
import type { ContinuationState } from '@/composables/continuation/useContinuationState'
import ReferenceImageSelector from './ReferenceImageSelector.vue'

type StorySectionKey = 'continuity' | 'story' | 'dialogue'

const props = defineProps<{
  pages: PageContent[]
  isGenerating: boolean
  progress: number
  bookId: string
  state: ContinuationState
}>()

const emit = defineEmits<{
  'batch-generate': [initialStyleReferenceTokens: string[] | null]
  'regenerate': [pageNumber: number]
  'use-previous': [pageNumber: number]
  'prompt-change': [pageNumber: number]
}>()

const state = props.state
const editingPromptPage = ref<number | null>(null)
const expandedStorySections = ref<Record<string, boolean>>({})
const refCount = ref(state.styleRefPages?.value || 3)
const batchInitialReferenceTokens = ref<string[]>([])
const selectorVisible = ref(false)
const availableOriginalImages = ref<MangaImageInfo[]>([])
const availableContinuationImages = ref<MangaImageInfo[]>([])
const availableCharacterForms = ref<CharacterFormInfo[]>([])

function togglePromptEdit(pageNumber: number) {
  if (editingPromptPage.value === pageNumber) {
    editingPromptPage.value = null
  } else {
    editingPromptPage.value = pageNumber
  }
}

function getImageUrl(imagePath: string): string {
  return state.getGeneratedImageUrl(imagePath)
}

function getStatusText(status: string): string {
  const map: Record<string, string> = {
    'pending': '待生成',
    'generating': '生成中',
    'generated': '已生成',
    'failed': '失败'
  }
  return map[status] || status
}

function getStorySectionStateKey(pageNumber: number, section: StorySectionKey): string {
  return `${pageNumber}:${section}`
}

function isStorySectionExpanded(pageNumber: number, section: StorySectionKey): boolean {
  return Boolean(expandedStorySections.value[getStorySectionStateKey(pageNumber, section)])
}

function toggleStorySection(pageNumber: number, section: StorySectionKey): void {
  const key = getStorySectionStateKey(pageNumber, section)
  expandedStorySections.value = {
    ...expandedStorySections.value,
    [key]: !expandedStorySections.value[key],
  }
}

function shouldShowStoryToggle(pageNumber: number, section: StorySectionKey): boolean {
  const page = props.pages.find(item => item.page_number === pageNumber)
  if (!page) return false

  const contentMap: Record<StorySectionKey, string> = {
    continuity: page.continuity_text || '',
    story: page.story_text || '',
    dialogue: page.dialogue_text || '',
  }

  const text = contentMap[section].trim()
  const thresholdMap: Record<StorySectionKey, number> = {
    continuity: 24,
    story: 24,
    dialogue: 18,
  }

  return text.length > thresholdMap[section]
}

function getStoryTextClass(pageNumber: number, section: StorySectionKey, maxLines: number): string[] {
  if (isStorySectionExpanded(pageNumber, section)) {
    return ['is-expanded']
  }

  return ['is-clamped', `lines-${maxLines}`]
}

function getInitialRefCount(): number {
  if (batchInitialReferenceTokens.value.length > 0) {
    return batchInitialReferenceTokens.value.length
  }
  return refCount.value
}

async function openBatchReferenceSelector() {
  try {
    const response = await getAvailableImages(
      props.bookId,
      'image'
    )
    if (response.success) {
      availableOriginalImages.value = response.original_images || []
      availableContinuationImages.value = response.continuation_images || []
      availableCharacterForms.value = response.character_forms || []
    }
  } catch (error) {
    console.error('加载可用图片失败:', error)
  }

  selectorVisible.value = true
}

function handleSelectorConfirm(tokens: string[]) {
  batchInitialReferenceTokens.value = tokens
}

function handleSelectorCancel() {
  // noop
}

function handleBatchGenerate() {
  const tokens = batchInitialReferenceTokens.value.length > 0 ? batchInitialReferenceTokens.value : null
  emit('batch-generate', tokens)
}

onMounted(() => {
  if (state.styleRefPages?.value) {
    refCount.value = state.styleRefPages.value
  }
})

watch(refCount, (newValue) => {
  if (state.styleRefPages && newValue > 0) {
    state.styleRefPages.value = newValue
  }
})

watch(() => state.styleRefPages?.value, (newValue) => {
  if (newValue && newValue !== refCount.value) {
    refCount.value = newValue
  }
})

watch(() => props.bookId, () => {
  batchInitialReferenceTokens.value = []
  availableOriginalImages.value = []
  availableContinuationImages.value = []
  availableCharacterForms.value = []
})

watch(() => props.pages.length, (pageCount) => {
  if (pageCount === 0) {
    batchInitialReferenceTokens.value = []
    expandedStorySections.value = {}
  }
})
</script>

<style scoped>.image-generation-panel {
  --image-generation-panel-border-default: rgba(99, 102, 241, .25);
  --image-generation-panel-surface-base: #f7f7f7;
  --image-generation-panel-text-primary: #92400e;
  --image-generation-panel-text-secondary: #1e40af;
  --image-generation-panel-text-muted: #065f46;
  --image-generation-panel-text-subtle: #991b1b;

  padding: 24px;
}

.image-generation-panel h3 {
  margin: 0 0 20px;
  font-size: 18px;
  font-weight: 600;
}

.image-generation-panel .generation-controls {
  margin-bottom: 24px;
}

.image-generation-panel .batch-config {
  margin-bottom: 16px;
  padding: 16px;
  background: var(--color-surface-subtle);
  border-radius: 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
}

.image-generation-panel .config-row {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.image-generation-panel .config-row label {
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-default, var(--color-text-default));
}

.image-generation-panel .ref-count-input {
  width: 60px;
  padding: 8px 10px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 6px;
  font-size: 14px;
  text-align: center;
}

.image-generation-panel .generated-images {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 20px;
}

.image-generation-panel .image-card {
  background: var(--color-surface-subtle);
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
}

.image-generation-panel .image-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: var(--color-surface-base);
  border-bottom: 1px solid var(--color-border-muted, var(--color-border-default));
}

.image-generation-panel .image-header h4 {
  margin: 0;
  font-size: 15px;
}

.image-generation-panel .image-preview {
  min-height: 320px;
  padding: 16px;
  background: var(--color-surface-base);
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-generation-panel .image-preview img {
  display: block;
  width: 100%;
  max-width: 100%;
  max-height: 720px;
  object-fit: contain;
  border-radius: 8px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
}

.image-generation-panel .no-image {
  min-height: 280px;
  width: 100%;
  border: 1px dashed var(--color-border-muted, var(--color-border-subtle));
  border-radius: 8px;
  background: var(--image-generation-panel-surface-base);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.image-generation-panel .no-image span {
  font-size: 40px;
  margin-bottom: 10px;
}

.image-generation-panel .no-image p {
  margin: 0;
}

.image-generation-panel .story-context {
  display: grid;
  gap: 12px;
  margin: 0;
  padding: 16px;
}

.image-generation-panel .context-block {
  background: var(--color-surface-base);
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 8px;
  padding: 10px 12px;
}

.image-generation-panel .context-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.image-generation-panel .context-block label {
  display: block;
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 4px;
}

.image-generation-panel .context-text {
  margin: 0;
  white-space: pre-wrap;
  line-height: 1.55;
  color: var(--color-text-default, var(--color-text-default));
  font-size: 13px;
}

.image-generation-panel .context-text.is-clamped {
  display: -webkit-box;
  overflow: hidden;
  -webkit-box-orient: vertical;
}

.image-generation-panel .context-text.lines-2 {
  -webkit-line-clamp: 2;
}

.image-generation-panel .context-text.lines-3 {
  -webkit-line-clamp: 3;
}

.image-generation-panel .context-text.is-expanded {
  display: block;
}

.image-generation-panel .context-toggle {
  border: none;
  background: none;
  padding: 0;
  color: var(--color-text-brand);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  flex-shrink: 0;
}

.image-generation-panel .prompt-section {
  padding: 0 16px 16px;
}

.image-generation-panel .prompt-input {
  width: 100%;
  white-space: pre-wrap;
  line-height: 1.6;
  padding: 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 8px;
  font-family: inherit;
}

.image-generation-panel .btn-mini {
  padding: 4px 10px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 6px;
  background: var(--color-surface-base);
  color: var(--color-text-brand);
  font-size: 12px;
  font-weight: 500;
  line-height: 1.2;
  cursor: pointer;
  transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
}

.image-generation-panel .btn-mini:hover {
  background: var(--color-surface-subtle);
  border-color: var(--color-border-brand);
}

.image-generation-panel .btn-mini:focus-visible {
  outline: 2px solid var(--image-generation-panel-border-default);
  outline-offset: 1px;
}

.image-generation-panel .prompt-collapsed {
  background: var(--color-surface-base);
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 8px;
  padding: 10px 12px;
}

.image-generation-panel .prompt-empty {
  margin: 0;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.image-generation-panel .prompt-collapsed-hint {
  margin: 0;
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 12px;
}


.image-generation-panel .image-actions {
  display: flex;
  gap: 8px;
  padding: 0 16px 16px;
}

.image-generation-panel .image-actions > * {
  flex: 1;
}


.image-generation-panel .progress-bar {
  height: 10px;
  background: var(--color-surface-hover);
  border-radius: 999px;
  overflow: hidden;
  margin-top: 16px;
  position: relative;
}

.image-generation-panel .progress-fill {
  height: 100%;
  background: var(--color-surface-brand);
}

.image-generation-panel .progress-text {
  position: absolute;
  right: 10px;
  top: -24px;
  font-size: 12px;
}

.image-generation-panel .image-status.pending {
  color: var(--image-generation-panel-text-primary);
}

.image-generation-panel .image-status.generating {
  color: var(--image-generation-panel-text-secondary);
}

.image-generation-panel .image-status.generated {
  color: var(--image-generation-panel-text-muted);
}

.image-generation-panel .image-status.failed {
  color: var(--image-generation-panel-text-subtle);
}

@media (--breakpoint-xl-down) {
  .image-generation-panel .generated-images {
    grid-template-columns: 1fr;
  }
}

@media (--breakpoint-sm-down) {
  .image-generation-panel {
    padding: 16px;
  }

  .image-generation-panel .image-preview {
    min-height: 240px;
    padding: 12px;
  }

  .image-generation-panel .story-context,
  .image-generation-panel .prompt-section,
  .image-generation-panel .image-actions {
    padding-left: 12px;
    padding-right: 12px;
  }
}
</style>
