<script setup lang="ts">
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'

export interface DetectedTextItem {
  original: string
  translated: string
}

defineProps<{
  items: DetectedTextItem[]
}>()

const MAX_LINE_LENGTH = 60

function wrapText(text: string): string {
  if (!text || text.length <= MAX_LINE_LENGTH) return text

  let result = ''
  let currentLine = ''

  for (let i = 0; i < text.length; i++) {
    currentLine += text[i]
    if (currentLine.length >= MAX_LINE_LENGTH) {
      let breakPoint = -1
      for (let j = currentLine.length - 1; j >= 0; j--) {
        const char = currentLine[j]
        if (char && ['。', '！', '？', '.', '!', '?', '；', ';', '，', ','].includes(char)) {
          breakPoint = j + 1
          break
        }
      }

      if (breakPoint > MAX_LINE_LENGTH * 0.6) {
        result += `${currentLine.substring(0, breakPoint)}\n`
        currentLine = currentLine.substring(breakPoint)
      } else {
        result += `${currentLine}\n`
        currentLine = ''
      }
    }
  }

  if (currentLine) {
    result += currentLine
  }

  return result
}

function formatText(text: string): string {
  return wrapText((text || '').trim())
}

</script>

<template>
  <section class="detected-text-panel" aria-labelledby="detectedTextTitle">
    <h3 id="detectedTextTitle" class="detected-text-panel__title">检测到的文本（原文 → 译文）</h3>

    <ProductScrollStack
      class="detected-text-panel__list"
      role="list"
      aria-label="检测文本列表"
      gap="sm"
      padding="none"
      :empty="items.length === 0"
    >
      <template #empty>
        <ProductEmptyState
          icon-name="scan-line"
          role="note"
          size="compact"
          title="未检测到文本或尚未翻译"
        />
      </template>

      <article
        v-for="(item, index) in items"
        :key="index"
        class="detected-text-panel__item"
        role="listitem"
      >
        <p class="detected-text-panel__original">{{ formatText(item.original) }}</p>
        <p class="detected-text-panel__translated">
          {{ formatText(item.translated) }}
        </p>
      </article>
    </ProductScrollStack>
  </section>
</template>

<style scoped>
.detected-text-panel {
  /* owner tokens: detected-text-panel */
  --detected-text-panel-divider: var(--color-border-muted);
  --detected-text-panel-translated-text: var(--color-action-primary);

  display: flex;
  flex-direction: column;
  width: 100%;
  height: 300px;
  min-height: 0;
  margin-top: 20px;
  padding: 15px;
  overflow: hidden;
  border: 1px solid var(--color-border-muted);
  border-radius: 4px;
  background-color: var(--color-surface-quiet);
  font-family: var(--font-mono);
  font-size: 0.9em;
  text-align: left;
}

.detected-text-panel__title {
  flex: 0 0 auto;
  margin: 0 0 12px;
  color: var(--color-text-default);
  font-weight: 600;
  font-size: 14px;
}

.detected-text-panel__list {
  --product-scroll-stack-empty-justify-content: flex-start;

  min-height: 0;
  overflow-x: auto;
}

.detected-text-panel__item {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--detected-text-panel-divider);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.detected-text-panel__item:last-child {
  border-bottom: 0;
}

.detected-text-panel__original,
.detected-text-panel__translated {
  margin: 0;
}

.detected-text-panel__original {
  color: var(--color-text-default);
}

.detected-text-panel__translated {
  color: var(--detected-text-panel-translated-text);
}

</style>
