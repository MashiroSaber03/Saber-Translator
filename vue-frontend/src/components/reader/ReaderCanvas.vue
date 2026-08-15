<script setup lang="ts">
import { computed } from 'vue'

import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'
import VirtualPageStream from '@/components/virtual/VirtualPageStream.vue'
import type { VirtualPageStreamItem } from '@/components/virtual/VirtualPageStream.vue'
import type { V2PageSummary } from '@/api/v2/content'
import { DEFAULT_READER_SETTINGS } from './readerSettings'

const props = withDefaults(
  defineProps<{
    backgroundColor?: string
    imageGap?: number
    imageWidth?: number
    images: V2PageSummary[]
    viewMode: 'original' | 'translated'
    isLoading: boolean
  }>(),
  {
    backgroundColor: DEFAULT_READER_SETTINGS.bgColor,
    imageGap: DEFAULT_READER_SETTINGS.imageGap,
    imageWidth: DEFAULT_READER_SETTINGS.imageWidth,
  }
)

const emit = defineEmits<{
  (e: 'pageChange', page: number): void
  (e: 'goTranslate'): void
}>()

const showEmptyState = computed(() => !props.isLoading && props.images.length === 0)
const showImagesContainer = computed(() => !props.isLoading && props.images.length > 0)
const canvasStyle = computed(() => ({
  '--reader-page-background': props.backgroundColor,
}))
const streamStyle = computed(() => ({
  '--reader-image-width': `${props.imageWidth}%`,
}))
const pageIndexById = computed(() => new Map(props.images.map((page, index) => [page.id, index])))
const streamItems = computed<VirtualPageStreamItem[]>(() =>
  props.images.map((page, index) => {
    return {
      alt: `第 ${index + 1} 页`,
      badge: props.viewMode === 'translated' && page.translatedUrl === null ? '未翻译' : undefined,
      height: page.height ?? 1,
      id: page.id,
      label: `${index + 1} / ${props.images.length}`,
      url:
        props.viewMode === 'translated' ? (page.translatedUrl ?? page.sourceUrl) : page.sourceUrl,
      width: page.width ?? 1,
    }
  })
)

function handleVisibleChange(ids: string[]): void {
  if (ids.length === 0) return
  const visibleIndexes = ids
    .map(id => pageIndexById.value.get(id))
    .filter((index): index is number => index !== undefined)
  if (visibleIndexes.length === 0) return
  emit('pageChange', Math.min(...visibleIndexes) + 1)
}
</script>

<template>
  <main class="reader-canvas" :style="canvasStyle">
    <div v-if="isLoading" class="reader-canvas__loading-state">
      <UiSpinner size="48px" label="正在加载阅读内容" :decorative="false" />
      <p class="reader-canvas__loading-text">正在加载...</p>
    </div>

    <ProductEmptyState
      v-else-if="showEmptyState"
      class="reader-canvas__empty-state"
      title="暂无图片"
      description="该章节还没有图片，点击下方按钮开始翻译"
      variant="inverse"
    >
      <template #icon>📖</template>
      <template #actions>
        <UiButton variant="primary" @click="emit('goTranslate')"> 进入翻译 </UiButton>
      </template>
    </ProductEmptyState>

    <VirtualPageStream
      v-else-if="showImagesContainer"
      class="reader-canvas__stream"
      :style="streamStyle"
      :items="streamItems"
      :gap="imageGap"
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
  width: min(var(--reader-image-width, 100%), 1200px);
  height: calc(100dvh - 56px);
  margin: 0 auto;
  padding: 16px 0 80px;
}

.reader-canvas__loading-state {
  --ui-spinner-border-width: 3px;
  --ui-spinner-track-color: var(--color-overlay-inverse-soft);
  --ui-spinner-color: var(--color-action-brand);
  --ui-spinner-duration: 1s;

  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: calc(100dvh - 56px);
  color: var(--reader-canvas-muted-text);
}

.reader-canvas__loading-text {
  margin: 16px 0;
}

.reader-canvas__empty-state {
  --product-empty-state-min-height: calc(100dvh - 56px);
  --product-empty-state-max-width: none;
  --product-empty-state-padding: 20px;
  --product-empty-state-icon-width: auto;
  --product-empty-state-icon-height: auto;
  --product-empty-state-icon-margin-bottom: 16px;
  --product-empty-state-icon-border: 0;
  --product-empty-state-icon-radius: 0;
  --product-empty-state-icon-background: transparent;
  --product-empty-state-icon-font-size: 64px;
  --product-empty-state-title-margin: 0 0 8px;
  --product-empty-state-title-font-size: 1.5rem;
  --product-empty-state-title-font-weight: 500;
  --product-empty-state-description-margin: 0 0 24px;
  --product-empty-state-description-font-size: 14px;
  --product-empty-state-actions-margin-top: 0;
}
</style>
