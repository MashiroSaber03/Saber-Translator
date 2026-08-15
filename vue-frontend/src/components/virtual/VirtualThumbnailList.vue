<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

import ProductThumbnailGrid from '@/components/product/ProductThumbnailGrid.vue'
import type { ProductThumbnailGridItem } from '@/components/product/ProductThumbnailGrid.vue'
import { fixedVirtualWindow } from './virtualWindow'

const props = withDefaults(defineProps<{
  activeId?: string | number | null
  ariaLabel?: string
  items: ProductThumbnailGridItem[]
  overscanItems?: number
}>(), {
  activeId: null,
  ariaLabel: '漫画页面缩略图',
  overscanItems: 5,
})

defineEmits<{
  select: [id: string | number]
}>()

const containerRef = ref<HTMLElement | null>(null)
const scrollTop = ref(0)
const viewportHeight = ref(0)
const viewportWidth = ref(0)
let resizeObserver: ResizeObserver | null = null
const DEFAULT_ITEM_WIDTH = 120
const GRID_GAP = 6
const itemHeight = computed(() => (
  Math.max(1, viewportWidth.value || DEFAULT_ITEM_WIDTH) * 4 / 3 + GRID_GAP
))

const windowState = computed(() => fixedVirtualWindow(
  props.items.length,
  itemHeight.value,
  scrollTop.value,
  viewportHeight.value,
  props.overscanItems,
))
const visibleItems = computed(() => props.items.slice(
  windowState.value.start,
  windowState.value.end,
))
const contentSize = computed(() => Math.max(
  0,
  windowState.value.totalSize - (props.items.length > 0 ? GRID_GAP : 0),
))
const innerStyle = computed(() => ({
  height: `${contentSize.value}px`,
}))
const windowStyle = computed(() => ({
  transform: `translateY(${windowState.value.offset}px)`,
}))

function syncViewport(): void {
  const container = containerRef.value
  if (!container) return
  scrollTop.value = container.scrollTop
  viewportHeight.value = container.clientHeight
  viewportWidth.value = container.clientWidth
}

function scrollActiveIntoView(): void {
  const index = props.items.findIndex(item => item.id === props.activeId)
  const container = containerRef.value
  if (index < 0 || !container) return
  const itemTop = index * itemHeight.value
  const itemBottom = itemTop + itemHeight.value
  if (itemTop < container.scrollTop) {
    container.scrollTop = itemTop
  } else if (itemBottom > container.scrollTop + container.clientHeight) {
    container.scrollTop = itemBottom - container.clientHeight
  }
  syncViewport()
}

watch(() => props.activeId, () => void nextTick(scrollActiveIntoView))
watch(() => props.items.length, () => void nextTick(syncViewport))

onMounted(() => {
  syncViewport()
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(syncViewport)
    if (containerRef.value) resizeObserver.observe(containerRef.value)
  }
  scrollActiveIntoView()
})

onBeforeUnmount(() => {
  resizeObserver?.disconnect()
})
</script>

<template>
  <div
    ref="containerRef"
    class="virtual-thumbnail-list"
    role="navigation"
    :aria-label="ariaLabel"
    @scroll.passive="syncViewport"
  >
    <div class="virtual-thumbnail-list__inner" :style="innerStyle">
      <ProductThumbnailGrid
        class="virtual-thumbnail-list__window"
        :style="windowStyle"
        :aria-label="ariaLabel"
        :columns="1"
        :items="visibleItems"
        @select="$emit('select', $event)"
      />
    </div>
  </div>
</template>

<style scoped>
.virtual-thumbnail-list {
  block-size: 100%;
  min-block-size: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
}

.virtual-thumbnail-list__inner {
  position: relative;
  inline-size: 100%;
}

.virtual-thumbnail-list__window {
  position: absolute;
  inset-block-start: 0;
  inset-inline: 0;
  will-change: transform;
}
</style>
