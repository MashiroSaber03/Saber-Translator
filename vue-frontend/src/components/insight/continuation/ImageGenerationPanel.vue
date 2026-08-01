<template>
  <div class="image-generation-panel">
    <ProductSectionHeader title="图片生成与导出" icon-name="palette" />

    <div class="image-generation-panel__controls">
      <div class="image-generation-panel__reference-config">
        <div class="image-generation-panel__reference-row">
          <UiField
            class="image-generation-panel__reference-count-field"
            variant="settings"
            label="画风参考图数量"
            control-id="continuation-style-reference-count"
          >
            <UiNumberField
              input-id="continuation-style-reference-count"
              v-model="refCount"
              :min="1"
              :max="10"
              size="xs"
              aria-label="画风参考图数量"
            />
          </UiField>
          <UiButton
            variant="secondary"
            @click="openBatchReferenceSelector"
          >
            <UiIcon name="camera" size="15" />
            <span>选择初始参考图 ({{ getInitialRefCount() }})</span>
          </UiButton>
        </div>
      </div>

      <UiButton
        variant="primary"
        block
        :disabled="isGenerating || pages.length === 0"
        size="lg"
        @click="handleBatchGenerate"
      >
        <UiIcon v-if="!isGenerating" name="sparkles" size="18" />
        <span>{{ isGenerating ? '生成中...' : '批量生成图片' }}</span>
      </UiButton>

      <UiProgressBar
        v-if="isGenerating"
        class="image-generation-panel__generation-progress"
        :value="boundedProgress"
        label="图片生成进度"
      >
        <span class="image-generation-panel__progress-text">{{ boundedProgress }}%</span>
      </UiProgressBar>
    </div>

    <div class="image-generation-panel__images">
      <ProductRecordCard v-for="page in pages" :key="page.page_number" class="image-generation-panel__image-card">
        <div class="image-generation-panel__image-header">
          <h4 class="image-generation-panel__image-title">页面 {{ page.page_number }}</h4>
          <ProductChipList
            class="image-generation-panel__status"
            :aria-label="`页面 ${page.page_number} 生成状态`"
            :items="[getStatusChipItem(page.status)]"
          />
        </div>

        <div class="image-generation-panel__preview">
          <img
            v-if="page.image_url"
            class="image-generation-panel__image"
            :src="getImageUrl(page.image_url)"
            :alt="`页面 ${page.page_number}`"
          >
          <div v-else class="image-generation-panel__empty-preview">
            <UiIcon
              class="image-generation-panel__empty-preview-icon"
              :class="{ 'image-generation-panel__empty-preview-icon--loading': page.status === 'generating' }"
              :name="page.status === 'generating' ? 'loading' : 'camera'"
              size="40"
              stroke-width="1.5"
            />
            <p class="image-generation-panel__empty-preview-text">{{ page.status === 'generating' ? '生成中...' : '未生成' }}</p>
          </div>
        </div>

        <div class="image-generation-panel__story-context">
          <ProductDetailSection label="上一页剧情">
            <template #label-actions>
              <UiButton
                variant="link"
                size="xs"
                v-if="shouldShowStoryToggle(page.page_number, 'continuity')"
                @click="toggleStorySection(page.page_number, 'continuity')"
              >
                {{ isStorySectionExpanded(page.page_number, 'continuity') ? '收起' : '展开' }}
              </UiButton>
            </template>
            <p
              class="image-generation-panel__context-text"
              :class="getStoryTextClass(page.page_number, 'continuity', 3)"
            >
              {{ page.continuity_text || '（无）' }}
            </p>
          </ProductDetailSection>
          <ProductDetailSection label="本页剧情">
            <template #label-actions>
              <UiButton
                variant="link"
                size="xs"
                v-if="shouldShowStoryToggle(page.page_number, 'story')"
                @click="toggleStorySection(page.page_number, 'story')"
              >
                {{ isStorySectionExpanded(page.page_number, 'story') ? '收起' : '展开' }}
              </UiButton>
            </template>
            <p
              class="image-generation-panel__context-text"
              :class="getStoryTextClass(page.page_number, 'story', 3)"
            >
              {{ page.story_text || '（无）' }}
            </p>
          </ProductDetailSection>
          <ProductDetailSection label="关键对白">
            <template #label-actions>
              <UiButton
                variant="link"
                size="xs"
                v-if="shouldShowStoryToggle(page.page_number, 'dialogue')"
                @click="toggleStorySection(page.page_number, 'dialogue')"
              >
                {{ isStorySectionExpanded(page.page_number, 'dialogue') ? '收起' : '展开' }}
              </UiButton>
            </template>
            <p
              class="image-generation-panel__context-text"
              :class="getStoryTextClass(page.page_number, 'dialogue', 2)"
            >
              {{ page.dialogue_text || '（无）' }}
            </p>
          </ProductDetailSection>
        </div>

        <div class="image-generation-panel__prompt-section">
          <ProductDetailSection label="最终生图提示词">
            <template #label-actions>
              <UiButton
                variant="secondary"
                size="xs"
                @click="togglePromptEdit(page.page_number)"
              >
                {{ editingPromptPage === page.page_number ? '收起' : '编辑' }}
              </UiButton>
            </template>
            <div v-if="editingPromptPage === page.page_number" class="image-generation-panel__prompt-edit">
              <UiTextarea
                :model-value="page.final_prompt"
                rows="8"
                class="image-generation-panel__prompt-input"
                variant="panel"
                size="md"
                placeholder="输入最终生图提示词..."
                @update:model-value="handlePromptInput(page.page_number, $event)"
              />
            </div>
            <template v-else>
              <p v-if="page.final_prompt" class="image-generation-panel__prompt-collapsed-hint">默认已折叠，点击“编辑”查看或修改</p>
              <ProductStatusBanner
                v-else
                icon-name="message"
                role="note"
                tone="neutral"
                title="暂无最终提示词"
              >
                生成图片前会在这里显示最终生图提示词。
              </ProductStatusBanner>
            </template>
          </ProductDetailSection>
        </div>

        <ProductActionRow
          class="image-generation-panel__image-actions"
          justify="between"
          :aria-label="`页面 ${page.page_number} 图片操作`"
        >
          <UiButton
            variant="secondary"
            :disabled="page.status === 'generating'"
            size="sm"
            @click="$emit('regenerate', page.page_number)"
          >
            <UiIcon name="refresh" size="14" />
            <span>重新生成</span>
          </UiButton>
          <UiButton
            variant="secondary"
            v-if="page.previous_url"
            size="sm"
            @click="$emit('use-previous', page.page_number)"
          >
            <UiIcon class="image-generation-panel__previous-icon" name="chevron-right" size="14" />
            <span>上一版本</span>
          </UiButton>
        </ProductActionRow>
      </ProductRecordCard>
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
    />
  </div>
</template>

<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductDetailSection from '@/components/product/ProductDetailSection.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
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
  'prompt-change': [pageNumber: number, prompt: string]
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
const boundedProgress = computed(() => Math.min(100, Math.max(0, Number(props.progress) || 0)))
let imageRequestSeq = 0
let isMounted = true

function togglePromptEdit(pageNumber: number) {
  if (editingPromptPage.value === pageNumber) {
    editingPromptPage.value = null
  } else {
    editingPromptPage.value = pageNumber
  }
}

function handlePromptInput(pageNumber: number, prompt: string): void {
  emit('prompt-change', pageNumber, prompt)
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

function getStatusTone(status: string): ProductChipItem['tone'] {
  const map: Record<string, ProductChipItem['tone']> = {
    pending: 'warning',
    generating: 'primary',
    generated: 'success',
    failed: 'danger',
  }

  return map[status] || 'neutral'
}

function getStatusChipItem(status: string): ProductChipItem {
  return {
    id: status,
    label: getStatusText(status),
    tone: getStatusTone(status),
  }
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
    return ['image-generation-panel__context-text--expanded']
  }

  return [
    'image-generation-panel__context-text--clamped',
    `image-generation-panel__context-text--lines-${maxLines}`,
  ]
}

function getInitialRefCount(): number {
  if (batchInitialReferenceTokens.value.length > 0) {
    return batchInitialReferenceTokens.value.length
  }
  return refCount.value
}

function invalidateAvailableImages(): void {
  imageRequestSeq += 1
}

async function openBatchReferenceSelector() {
  const bookId = props.bookId
  const requestId = ++imageRequestSeq

  try {
    const response = await getAvailableImages(bookId)
    if (!isMounted || requestId !== imageRequestSeq || props.bookId !== bookId) return
    availableOriginalImages.value = response.original_images
    availableContinuationImages.value = response.continuation_images
    availableCharacterForms.value = response.character_forms
  } catch {
    if (!isMounted || requestId !== imageRequestSeq || props.bookId !== bookId) return
    availableOriginalImages.value = []
    availableContinuationImages.value = []
    availableCharacterForms.value = []
    state.showMessage('加载可用参考图失败', 'error')
  }

  selectorVisible.value = true
}

function handleSelectorConfirm(tokens: string[]) {
  batchInitialReferenceTokens.value = tokens
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
  invalidateAvailableImages()
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

onBeforeUnmount(() => {
  isMounted = false
  invalidateAvailableImages()
})
</script>

<style scoped>
.image-generation-panel {
  --image-generation-panel-empty-preview-background: var(--color-surface-muted);

  min-width: 0;
  container: continuation-image-generation / inline-size;
}

.image-generation-panel__controls {
  margin-bottom: 24px;
}

.image-generation-panel__reference-config {
  margin-bottom: 16px;
  padding: 16px;
  background: var(--color-surface-subtle);
  border-radius: 12px;
  border: 1px solid var(--color-border-muted);
}

.image-generation-panel__reference-row {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.image-generation-panel__reference-count-field {
  width: min(100%, 150px);
  margin-bottom: 0;
}

.image-generation-panel__images {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 360px), 1fr));
  gap: 20px;
}

.image-generation-panel__image-card {
  --product-record-card-background: var(--color-surface-subtle);
  --product-record-card-border: var(--color-border-muted);
  --product-record-card-radius: 12px;
  --product-record-card-padding: 0;
  --product-record-card-gap: 0;
  --product-record-card-shadow-hover: none;

  overflow: hidden;
}

.image-generation-panel__image-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: var(--color-surface-base);
  border-bottom: 1px solid var(--color-border-muted);
}

.image-generation-panel__image-title {
  margin: 0;
  font-size: 15px;
}

.image-generation-panel__preview {
  min-height: 320px;
  padding: 16px;
  background: var(--color-surface-base);
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-generation-panel__image {
  display: block;
  width: 100%;
  max-width: 100%;
  max-height: 720px;
  object-fit: contain;
  border-radius: 8px;
  border: 1px solid var(--color-border-muted);
}

.image-generation-panel__empty-preview {
  min-height: 280px;
  width: 100%;
  border: 1px dashed var(--color-border-muted);
  border-radius: 8px;
  background: var(--image-generation-panel-empty-preview-background);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--color-text-supporting);
}

.image-generation-panel__empty-preview-icon {
  margin-bottom: 10px;
}

.image-generation-panel__empty-preview-icon--loading {
  animation: spin 1s linear infinite;
}

.image-generation-panel__empty-preview-text {
  margin: 0;
}

.image-generation-panel__story-context {
  display: grid;
  gap: 12px;
  margin: 0;
  padding: 16px;
}

.image-generation-panel__context-text {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  line-height: 1.55;
  color: var(--color-text-default);
  font-size: 13px;
}

.image-generation-panel__context-text--clamped {
  display: -webkit-box;
  overflow: hidden;
  -webkit-box-orient: vertical;
}

.image-generation-panel__context-text--lines-2 {
  -webkit-line-clamp: 2;
}

.image-generation-panel__context-text--lines-3 {
  -webkit-line-clamp: 3;
}

.image-generation-panel__context-text--expanded {
  display: block;
}

.image-generation-panel__prompt-section {
  padding: 0 16px 16px;
}

.image-generation-panel__prompt-input {
  width: 100%;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.image-generation-panel__prompt-collapsed-hint {
  margin: 0;
  color: var(--color-text-supporting);
  font-size: 12px;
  overflow-wrap: anywhere;
}


.image-generation-panel__image-actions {
  padding: 0 16px 16px;
}

.image-generation-panel__image-actions > * {
  flex: 1;
}

.image-generation-panel__previous-icon {
  transform: rotate(180deg);
}

.image-generation-panel__generation-progress {
  margin-top: 16px;
}

.image-generation-panel__progress-text {
  font-size: 12px;
}

@container continuation-image-generation (max-width: 520px) {
  .image-generation-panel__preview {
    min-height: 240px;
    padding: 12px;
  }

  .image-generation-panel__story-context,
  .image-generation-panel__prompt-section,
  .image-generation-panel__image-actions {
    padding-left: 12px;
    padding-right: 12px;
  }
}
</style>
