<template>
  <div v-if="visible" class="edit-thumbnails-panel">
    <ProductHorizontalScrollStrip
      class="edit-thumbnails-panel__strip"
      aria-label="编辑模式缩略图滚动条"
    >
      <ProductThumbnailGrid
        class="edit-thumbnails-panel__grid"
        aria-label="编辑图片缩略图导航"
        :items="thumbnailItems"
        @select="handleThumbnailSelect"
      />
    </ProductHorizontalScrollStrip>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import ProductHorizontalScrollStrip from '@/components/product/ProductHorizontalScrollStrip.vue'
import ProductThumbnailGrid from '@/components/product/ProductThumbnailGrid.vue'
import type { ProductThumbnailGridItem } from '@/components/product/ProductThumbnailGrid.vue'
import type { ImageData } from '@/types/image'

const props = defineProps<{
  visible: boolean
  images: ImageData[]
  currentImageIndex: number
}>()

const emit = defineEmits<{
  (e: 'switch-to-image', index: number): void
}>()

const thumbnailItems = computed<ProductThumbnailGridItem[]>(() => {
  return props.images.map((image, index) => ({
    id: index,
    src: image.translatedDataURL || image.originalDataURL || '',
    alt: `图片 ${index + 1}`,
    label: String(index + 1),
    selected: index === props.currentImageIndex,
    fallbackLabel: String(index + 1),
    ariaLabel: `切换到图片 ${index + 1}`,
  }))
})

function handleThumbnailSelect(id: string | number): void {
  if (typeof id !== 'number') return
  emit('switch-to-image', id)
}
</script>

<style scoped>
.edit-thumbnails-panel {
  --edit-thumbnail-panel-background: color-mix(in srgb, var(--color-overlay-backdrop-solid) 30%, transparent);
  --edit-thumbnail-panel-divider-border: var(--color-overlay-inverse-subtle);

  position: relative;
  width: auto;
  background: var(--edit-thumbnail-panel-background);
  padding: 10px 15px;
  border-bottom: 1px solid var(--edit-thumbnail-panel-divider-border);
  flex-shrink: 0;
}

.edit-thumbnails-panel__strip {
  --product-horizontal-scroll-strip-padding: 5px 0;
}

.edit-thumbnails-panel__grid {
  --product-thumbnail-grid-aspect-ratio: 3 / 4;

  grid-auto-columns: 60px;
  grid-auto-flow: column;
  grid-template-columns: none;
  gap: 10px;
}
</style>
