<script setup lang="ts">
import { computed } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
import EmbeddingRebuildControl from './EmbeddingRebuildControl.vue'

type QAMode = 'precise' | 'global'

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

const showPreciseModeOptions = computed(() => props.qaMode === 'precise')
const parentChildModel = computed({
  get: () => props.useParentChild,
  set: value => emit('update:useParentChild', Boolean(value)),
})
const reasoningModel = computed({
  get: () => props.useReasoning,
  set: value => emit('update:useReasoning', Boolean(value)),
})
const rerankerModel = computed({
  get: () => props.useReranker,
  set: value => emit('update:useReranker', Boolean(value)),
})
const topKModel = computed({
  get: () => props.topK,
  set: value => emit('update:topK', Number(value)),
})
const thresholdModel = computed({
  get: () => props.threshold,
  set: value => emit('update:threshold', Number(value)),
})
</script>

<template>
  <div class="chat-options">
    <div class="qa-mode-toggle" title="精确模式：使用RAG检索相关片段；全局模式：使用全文摘要">
      <UiButton
        variant="toolbar"
        type="button"
        class="qa-mode-btn"
        :class="{ active: qaMode === 'precise' }"
        @click="$emit('update:qaMode', 'precise')"
      >
        🎯 精确模式
      </UiButton>
      <UiButton
        variant="toolbar"
        type="button"
        class="qa-mode-btn"
        :class="{ active: qaMode === 'global' }"
        @click="$emit('update:qaMode', 'global')"
      >
        🌐 全局模式
      </UiButton>
    </div>

    <span class="chat-option-divider">|</span>

    <div v-if="showPreciseModeOptions" class="precise-mode-options">
      <label class="ui-checkbox-label compact" title="启用父子块模式">
        <UiInput v-model="parentChildModel" type="checkbox" />
        <span>父子块模式</span>
      </label>
      <label class="ui-checkbox-label compact" title="启用推理检索">
        <UiInput v-model="reasoningModel" type="checkbox" />
        <span>推理检索</span>
      </label>
      <label class="ui-checkbox-label compact" title="启用重排序">
        <UiInput v-model="rerankerModel" type="checkbox" />
        <span>重排序</span>
      </label>

      <span class="chat-option-divider">|</span>

      <label class="input-label compact" title="返回的最大结果数">
        <span>Top K:</span>
        <UiInput v-model.number="topKModel" type="number" min="1" max="20" class="input-small" />
      </label>
      <label class="input-label compact" title="相关性阈值">
        <span>阈值:</span>
        <UiInput v-model.number="thresholdModel" type="number" min="0" max="1" step="0.1" class="input-small" />
      </label>

      <span class="chat-option-divider">|</span>

      <EmbeddingRebuildControl
        :is-rebuilding="isRebuildingEmbeddings"
        :progress-label="progressLabel"
        @rebuild="$emit('rebuild')"
      />
    </div>

    <div v-else class="global-mode-hint">
      <span class="hint-text">💡 全局模式使用全文摘要回答，适合总结性问题</span>
      <div class="welcome-examples">
        <span
          v-for="(example, index) in globalModeExamples"
          :key="index"
          class="example-tag"
          role="button"
          tabindex="0"
          @click="$emit('askExample', example)"
          @keydown.enter="$emit('askExample', example)"
          @keydown.space.prevent="$emit('askExample', example)"
        >
          {{ example }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-options {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--color-border-muted);
}

.qa-mode-toggle {
  display: flex;
  gap: 2px;
  padding: 2px;
  border-radius: 8px;
  background: var(--insight-bg-secondary);
}

.qa-mode-btn {
  padding: 6px 12px;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: var(--insight-text-secondary);
  font-size: 13px;
  white-space: nowrap;
  cursor: pointer;
  transition: all 0.2s;
}

.qa-mode-btn:hover {
  background: var(--insight-bg-tertiary);
  color: var(--insight-text-primary);
}

.qa-mode-btn.active {
  background: var(--insight-color-primary);
  color: white;
  font-weight: 500;
}

.chat-option-divider {
  margin: 0 4px;
  color: var(--color-border-muted);
}

.precise-mode-options {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 16px;
}

.ui-checkbox-label.compact,
.input-label.compact {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--insight-text-secondary);
  font-size: 13px;
  cursor: pointer;
}

.ui-checkbox-label.compact:hover {
  color: var(--insight-text-primary);
}

.input-small {
  width: 50px;
  padding: 2px 6px;
  border: 1px solid var(--color-border-muted);
  border-radius: 4px;
  background: var(--insight-bg-primary);
  color: var(--insight-text-primary);
  font-size: 12px;
}

.global-mode-hint {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.hint-text {
  color: var(--insight-text-secondary);
  font-size: 13px;
  font-style: italic;
}

.welcome-examples {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
  margin-top: 12px;
}

.example-tag {
  padding: 6px 12px;
  border-radius: 16px;
  background: var(--insight-bg-secondary);
  color: var(--insight-text-secondary);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
}

.example-tag:hover {
  background: var(--insight-color-primary);
  color: white;
}
</style>
