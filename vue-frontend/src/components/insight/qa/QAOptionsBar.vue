<script setup lang="ts">
import { computed } from 'vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import type { ProductSegmentedTab } from '@/components/product/ProductSegmentedTabs.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import type { QAMode } from '@/types/insight'
import EmbeddingRebuildControl from './EmbeddingRebuildControl.vue'

const props = defineProps<{
  globalModeExamples: string[]
  isRebuildingEmbeddings: boolean
  progressLabel: string
  qaMode: QAMode
  threshold: number
  topK: number
  useParentChild: boolean
  useReasoning: boolean
  useReranker: boolean
}>()

const emit = defineEmits<{
  (event: 'askExample', question: string): void
  (event: 'rebuild'): void
  (event: 'update:qaMode', value: QAMode): void
  (event: 'update:threshold', value: number): void
  (event: 'update:topK', value: number): void
  (event: 'update:useParentChild', value: boolean): void
  (event: 'update:useReasoning', value: boolean): void
  (event: 'update:useReranker', value: boolean): void
}>()

const qaModeTabs: ProductSegmentedTab[] = [
  { id: 'precise', label: '精确模式', iconName: 'target' },
  { id: 'global', label: '全局模式', iconName: 'globe' },
]
const qaModeGlyphs: Record<QAMode, string> = {
  precise: '🎯',
  global: '🌐',
}

function qaModeGlyph(tabId: string): string {
  return tabId === 'precise' || tabId === 'global' ? qaModeGlyphs[tabId] : ''
}
const showPreciseModeOptions = computed(() => props.qaMode === 'precise')
const parentChildModel = computed({
  get: () => props.useParentChild,
  set: value => emit('update:useParentChild', value),
})
const reasoningModel = computed({
  get: () => props.useReasoning,
  set: value => emit('update:useReasoning', value),
})
const rerankerModel = computed({
  get: () => props.useReranker,
  set: value => emit('update:useReranker', value),
})
const topKModel = computed({
  get: () => props.topK,
  set: value => emit('update:topK', value),
})
const thresholdModel = computed({
  get: () => props.threshold,
  set: value => emit('update:threshold', value),
})
const globalExampleChips = computed<ProductChipItem[]>(() => {
  return props.globalModeExamples.map(example => ({
    id: example,
    label: example,
    ariaLabel: `提问示例：${example}`,
    interactive: true,
    tone: 'neutral',
  }))
})

function askExample(id: string | number): void {
  if (typeof id === 'string') emit('askExample', id)
}

function updateQaMode(mode: string): void {
  if (mode !== 'precise' && mode !== 'global') return
  emit('update:qaMode', mode)
}
</script>

<template>
  <div class="qa-options-bar">
    <ProductSegmentedTabs
      class="qa-options-bar__mode-tabs"
      title="精确模式：使用RAG检索相关片段；全局模式：使用全文摘要"
      aria-label="问答模式"
      :tabs="qaModeTabs"
      :active-tab="qaMode"
      @update:active-tab="updateQaMode"
    >
      <template #tabIcon="{ tab }">{{ qaModeGlyph(tab.id) }}</template>
    </ProductSegmentedTabs>

    <span class="qa-options-bar__divider" aria-hidden="true">|</span>

    <div v-if="showPreciseModeOptions" class="qa-options-bar__precise-options">
      <UiCheckbox
        v-model="parentChildModel"
        class="qa-options-bar__option-checkbox"
        label="父子块模式"
        title="启用父子块模式"
      />
      <UiCheckbox
        v-model="reasoningModel"
        class="qa-options-bar__option-checkbox"
        label="推理检索"
        title="启用推理检索"
      />
      <UiCheckbox
        v-model="rerankerModel"
        class="qa-options-bar__option-checkbox"
        label="重排序"
        title="启用重排序"
      />

      <span class="qa-options-bar__divider" aria-hidden="true">|</span>

      <UiField
        class="qa-options-bar__number-field"
        variant="settings"
        layout="inline"
        label="Top K:"
        control-id="qaTopK"
        title="返回的最大结果数"
      >
        <UiNumberField
          v-model="topKModel"
          input-id="qaTopK"
          :min="1"
          size="xs"
        />
      </UiField>
      <UiField
        class="qa-options-bar__number-field"
        variant="settings"
        layout="inline"
        label="阈值:"
        control-id="qaThreshold"
        title="相关性阈值"
      >
        <UiNumberField
          v-model="thresholdModel"
          input-id="qaThreshold"
          :min="0"
          :max="1"
          :step="0.1"
          size="xs"
        />
      </UiField>

      <span class="qa-options-bar__divider" aria-hidden="true">|</span>

      <EmbeddingRebuildControl
        :is-rebuilding="isRebuildingEmbeddings"
        :progress-label="progressLabel"
        @rebuild="$emit('rebuild')"
      />
    </div>

    <div v-else class="qa-options-bar__global-hint">
      <span class="qa-options-bar__hint-text">全局模式使用全文摘要回答，适合总结性问题</span>
      <ProductChipList
        class="qa-options-bar__example-list"
        aria-label="全局模式示例问题"
        :items="globalExampleChips"
        @select="askExample"
      />
    </div>
  </div>
</template>

<style scoped>
.qa-options-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 16px;
  min-width: 0;
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--color-border-muted);
}

.qa-options-bar__mode-tabs {
  --product-segmented-tabs-active-background: var(--color-surface-brand);
  --product-segmented-tabs-active-text: var(--color-text-inverse);
  --product-segmented-tabs-active-shadow: none;
  --product-segmented-tabs-background: var(--insight-surface-secondary);
  --product-segmented-tabs-border: transparent;
  --product-segmented-tabs-gap: 2px;
  --product-segmented-tabs-padding: 2px;
  --product-segmented-tabs-tab-padding: 6px 12px;

  flex: 0 0 auto;
}

.qa-options-bar__divider {
  margin: 0 4px;
  color: var(--color-border-muted);
}

.qa-options-bar__precise-options {
  display: flex;
  flex-wrap: wrap;
  flex: 1 1 0;
  align-items: center;
  gap: 16px;
  min-width: 0;
}

.qa-options-bar__option-checkbox,
.qa-options-bar__number-field {
  --ui-field-inline-label-color: var(--insight-text-secondary);
  --ui-field-inline-label-font-size: 13px;

  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--insight-text-secondary);
  font-size: 13px;
  cursor: pointer;
}

.qa-options-bar__global-hint {
  display: flex;
  flex: 1 1 260px;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}

.qa-options-bar__hint-text {
  color: var(--insight-text-secondary);
  font-size: 13px;
  font-style: italic;
  overflow-wrap: anywhere;
}

.qa-options-bar__example-list {
  max-width: 100%;
  margin-top: 12px;
}
</style>
