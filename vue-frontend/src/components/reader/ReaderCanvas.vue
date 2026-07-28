<script setup lang="ts">
import { computed } from 'vue'

import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'
import VirtualPageStream from '@/components/virtual/VirtualPageStream.vue'
import type { VirtualPageStreamItem } from '@/components/virtual/VirtualPageStream.vue'
import type { V2PageSummary } from '@/api/v2/content'

const props = defineProps<{
  images: V2PageSummary[]
  viewMode: 'original' | 'translated'
  isLoading: boolean
}>()

const emit = defineEmits<{
  (e: 'pageChange', page: number): void
  (e: 'goTranslate'): void
}>()

const showEmptyState = computed(() => !props.isLoading && props.images.length === 0)
const showImagesContainer = computed(() => !props.isLoading && props.images.length > 0)
const streamItems = computed<VirtualPageStreamItem[]>(() => props.images.map((page, index) => {
  const compatible = page as V2PageSummary & {
    original?: string
    translated?: string
  }
  const source = page.sourceUrl || compatible.original || ''
  return {
    alt: `第 ${index + 1} 页`,
    height: page.height ?? 1,
    id: page.id || String(index),
    label: `${index + 1} / ${props.images.length}`,
    url: props.viewMode === 'translated'
      ? page.translatedUrl || compatible.translated || source
      : source,
    width: page.width ?? 1,
  }
}))

function handleVisibleChange(ids: string[]): void {
  if (ids.length === 0) return
  const visibleIndexes = ids
    .map(id => props.images.findIndex(page => page.id === id))
    .filter(index => index >= 0)
  if (visibleIndexes.length === 0) return
  emit('pageChange', Math.min(...visibleIndexes) + 1)
}
</script>

<template>
  <main class="reader-canvas">
    <div v-if="isLoading" class="reader-canvas__loading-state">
      <UiSpinner size="48px" label="正在加载阅读内容" :decorative="false" />
      <p class="reader-canvas__loading-text">正在加载...</p>
    </div>

    <ProductEmptyState
      v-else-if="showEmptyState"
      class="reader-canvas__empty-state"
      icon-name="book-open"
      title="暂无图片"
      description="该章节还没有图片，点击下方按钮开始翻译"
      variant="inverse"
    >
      <template #actions>
        <UiButton variant="primary" @click="emit('goTranslate')">
          进入翻译
        </UiButton>
      </template>
    </ProductEmptyState>

    <VirtualPageStream
      v-else-if="showImagesContainer"
      class="reader-canvas__stream"
      :items="streamItems"
      :gap="8"
      :overscan-screens="2"
      @visible-change="handleVisibleChange"
    />
  </main>
</template>

<style scoped>
.reader-canvas {
  --reader-canvas-page-background: var(--color-surface-inverse);
  --reader-canvas-muted-text: color-mix(in srgb, var(--color-text-inverse) 70%, transparent);

  min-height: calc(100dvh - 56px);
  background: var(--reader-page-background, var(--reader-canvas-page-background));
}

.reader-canvas__stream {
  width: min(100%, 1200px);
  height: calc(100dvh - 56px);
  margin: 0 auto;
  padding: 16px 0 80px;
}

.reader-canvas__loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: calc(100dvh - 56px);
  color: var(--reader-canvas-muted-text);
}

.reader-canvas__loading-text {
  margin: 0;
}

.reader-canvas__empty-state {
  --product-empty-state-min-height: calc(100dvh - 56px);
}
</style>
