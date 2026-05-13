<template>
  <div class="image-generation-panel">
    <h3>🎨 图片生成与导出</h3>

    <div class="generation-controls">
      <div class="batch-config">
        <div class="config-row">
          <label>画风参考图数量:</label>
          <input
            type="number"
            v-model.number="refCount"
            min="1"
            max="10"
            class="ref-count-input"
          />
          <button
            class="btn secondary"
            @click="openBatchReferenceSelector"
          >
            📷 选择初始参考图 ({{ getInitialRefCount() }})
          </button>
        </div>
      </div>

      <button
        class="btn primary large"
        :disabled="isGenerating || pages.length === 0"
        @click="handleBatchGenerate"
      >
        {{ isGenerating ? '生成中...' : '🚀 批量生成图片' }}
      </button>

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
              <button
                v-if="shouldShowStoryToggle(page.page_number, 'continuity')"
                class="context-toggle"
                @click="toggleStorySection(page.page_number, 'continuity')"
              >
                {{ isStorySectionExpanded(page.page_number, 'continuity') ? '收起' : '展开' }}
              </button>
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
              <button
                v-if="shouldShowStoryToggle(page.page_number, 'story')"
                class="context-toggle"
                @click="toggleStorySection(page.page_number, 'story')"
              >
                {{ isStorySectionExpanded(page.page_number, 'story') ? '收起' : '展开' }}
              </button>
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
              <button
                v-if="shouldShowStoryToggle(page.page_number, 'dialogue')"
                class="context-toggle"
                @click="toggleStorySection(page.page_number, 'dialogue')"
              >
                {{ isStorySectionExpanded(page.page_number, 'dialogue') ? '收起' : '展开' }}
              </button>
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
            <button
              class="btn-mini"
              @click="togglePromptEdit(page.page_number)"
            >
              {{ editingPromptPage === page.page_number ? '收起' : '编辑' }}
            </button>
          </div>
          <div v-if="editingPromptPage === page.page_number" class="prompt-edit">
            <textarea
              v-model="page.final_prompt"
              rows="8"
              class="prompt-input"
              placeholder="输入最终生图提示词..."
              @input="$emit('prompt-change', page.page_number)"
            ></textarea>
          </div>
          <div v-else class="prompt-collapsed">
            <p v-if="page.final_prompt" class="prompt-collapsed-hint">默认已折叠，点击“编辑”查看或修改</p>
            <p v-else class="prompt-empty">暂无最终提示词</p>
          </div>
        </div>

        <div class="image-actions">
          <button
            class="btn secondary small"
            :disabled="page.status === 'generating'"
            @click="$emit('regenerate', page.page_number)"
          >
            ↺ 重新生成
          </button>
          <button
            v-if="page.previous_url"
            class="btn secondary small"
            @click="$emit('use-previous', page.page_number)"
          >
            ◀ 上一版本
          </button>
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
import { ref, watch, onMounted } from 'vue'
import type { PageContent, MangaImageInfo, CharacterFormInfo } from '@/api/continuation'
import { getAvailableImages } from '@/api/continuation'
import { useContinuationStateInject } from '@/composables/continuation/useContinuationState'
import ReferenceImageSelector from './ReferenceImageSelector.vue'

type StorySectionKey = 'continuity' | 'story' | 'dialogue'

const props = defineProps<{
  pages: PageContent[]
  isGenerating: boolean
  progress: number
  bookId: string
}>()

const emit = defineEmits<{
  'batch-generate': [initialStyleReferenceTokens: string[] | null]
  'regenerate': [pageNumber: number]
  'use-previous': [pageNumber: number]
  'prompt-change': [pageNumber: number]
}>()

const state = useContinuationStateInject()
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

<style scoped>
.image-generation-panel {
  padding: 24px;
}

.image-generation-panel h3 {
  margin: 0 0 20px;
  font-size: 18px;
  font-weight: 600;
}

.generation-controls {
  margin-bottom: 24px;
}

.batch-config {
  margin-bottom: 16px;
  padding: 16px;
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 12px;
  border: 1px solid var(--border-color, #e0e0e0);
}

.config-row {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.config-row label {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary, #333);
}

.ref-count-input {
  width: 60px;
  padding: 8px 10px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 6px;
  font-size: 14px;
  text-align: center;
}

.generated-images {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 20px;
}

.image-card {
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--border-color, #e0e0e0);
}

.image-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: var(--bg-primary, #fff);
  border-bottom: 1px solid var(--border-color, #e0e0e0);
}

.image-header h4 {
  margin: 0;
  font-size: 15px;
}

.image-preview {
  min-height: 320px;
  padding: 16px;
  background: var(--bg-primary, #fff);
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-preview img {
  display: block;
  width: 100%;
  max-width: 100%;
  max-height: 720px;
  object-fit: contain;
  border-radius: 8px;
  border: 1px solid var(--border-color, #ddd);
}

.no-image {
  min-height: 280px;
  width: 100%;
  border: 1px dashed var(--border-color, #ddd);
  border-radius: 8px;
  background: var(--bg-secondary, #f7f7f7);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--text-secondary, #666);
}

.no-image span {
  font-size: 40px;
  margin-bottom: 10px;
}

.no-image p {
  margin: 0;
}

.story-context {
  display: grid;
  gap: 12px;
  margin: 0;
  padding: 16px;
}

.context-block {
  background: var(--bg-primary, #fff);
  border: 1px solid var(--border-color, #ddd);
  border-radius: 8px;
  padding: 10px 12px;
}

.context-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.context-block label {
  display: block;
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 4px;
}

.context-text {
  margin: 0;
  white-space: pre-wrap;
  line-height: 1.55;
  color: var(--text-primary, #333);
  font-size: 13px;
}

.context-text.is-clamped {
  display: -webkit-box;
  overflow: hidden;
  -webkit-box-orient: vertical;
}

.context-text.lines-2 {
  -webkit-line-clamp: 2;
}

.context-text.lines-3 {
  -webkit-line-clamp: 3;
}

.context-text.is-expanded {
  display: block;
}

.context-toggle {
  border: none;
  background: none;
  padding: 0;
  color: var(--primary, #6366f1);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  flex-shrink: 0;
}

.prompt-section {
  padding: 0 16px 16px;
}

.prompt-input,
.prompt-text {
  width: 100%;
  white-space: pre-wrap;
  line-height: 1.6;
}

.prompt-input {
  padding: 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 8px;
  font-family: inherit;
}

.prompt-collapsed {
  background: var(--bg-primary, #fff);
  border: 1px solid var(--border-color, #ddd);
  border-radius: 8px;
  padding: 10px 12px;
}

.prompt-empty {
  margin: 0;
  color: var(--text-secondary, #666);
}

.prompt-collapsed-hint {
  margin: 0;
  color: var(--text-secondary, #666);
  font-size: 12px;
}

.image-actions {
  display: flex;
  gap: 8px;
  padding: 0 16px 16px;
}

.image-actions .btn {
  flex: 1;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
}

.btn.primary {
  background: var(--primary, #6366f1);
  color: white;
}

.btn.secondary {
  background: var(--bg-primary, #fff);
  color: var(--text-primary, #333);
  border: 1px solid var(--border-color, #ddd);
}

.btn.primary.large {
  width: 100%;
}

.progress-bar {
  height: 10px;
  background: #e5e7eb;
  border-radius: 999px;
  overflow: hidden;
  margin-top: 16px;
  position: relative;
}

.progress-fill {
  height: 100%;
  background: var(--primary, #6366f1);
}

.progress-text {
  position: absolute;
  right: 10px;
  top: -24px;
  font-size: 12px;
}

.image-status.pending {
  color: #92400e;
}

.image-status.generating {
  color: #1e40af;
}

.image-status.generated {
  color: #065f46;
}

.image-status.failed {
  color: #991b1b;
}

@media (max-width: 1024px) {
  .generated-images {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .image-generation-panel {
    padding: 16px;
  }

  .image-preview {
    min-height: 240px;
    padding: 12px;
  }

  .story-context,
  .prompt-section,
  .image-actions {
    padding-left: 12px;
    padding-right: 12px;
  }
}
</style>
