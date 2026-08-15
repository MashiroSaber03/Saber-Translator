<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

import ProductThumbnailGrid from '@/components/product/ProductThumbnailGrid.vue'
import type { ProductThumbnailGridItem } from '@/components/product/ProductThumbnailGrid.vue'
import { fixedVirtualWindow } from './virtualWindow'

const props = withDefaults(defineProps<{
  activeId?: string | number | null
  ariaLabel?: string
  columns?: number
  items: ProductThumbnailGridItem[]
  maxHeight?: number
  minItemWidth?: number
  overscanRows?: number
}>(), {
  activeId: null,
  ariaLabel: '漫画页面缩略图',
  columns: 4,
  maxHeight: 420,
  minItemWidth: 0,
  overscanRows: 2,
})

defineEmits<{
  select: [id: string | number]
}>()

const containerRef = ref<HTMLElement | null>(null)
const scrollTop = ref(0)
const viewportHeight = ref(0)
const viewportWidth = ref(0)
let resizeObserver: ResizeObserver | null = null
const GRID_GAP = 6

const normalizedColumns = computed(() => {
  if (props.minItemWidth > 0) {
    const width = Math.max(1, viewportWidth.value || props.minItemWidth)
    return Math.max(
      1,
      Math.floor((width + GRID_GAP) / (props.minItemWidth + GRID_GAP)),
    )
  }
  return Math.max(1, Math.floor(props.columns))
})
const rowCount = computed(() => Math.ceil(props.items.length / normalizedColumns.value))
const rowHeight = computed(() => {
  const width = Math.max(1, viewportWidth.value || 240)
  const cardWidth = Math.max(
    1,
    (width - GRID_GAP * (normalizedColumns.value - 1)) / normalizedColumns.value,
  )
  return cardWidth * 4 / 3 + GRID_GAP
})
const windowState = computed(() => fixedVirtualWindow(
  rowCount.value,
  rowHeight.value,
  scrollTop.value,
  viewportHeight.value,
  props.overscanRows,
))
const visibleItems = computed(() => props.items.slice(
  windowState.value.start * normalizedColumns.value,
  windowState.value.end * normalizedColumns.value,
))
const contentSize = computed(() => Math.max(
  0,
  windowState.value.totalSize - (rowCount.value > 0 ? GRID_GAP : 0),
))
const containerStyle = computed(() => ({
  blockSize: `${Math.min(props.maxHeight, contentSize.value)}px`,
}))
const innerStyle = computed(() => ({
  blockSize: `${contentSize.value}px`,
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
  const row = Math.floor(index / normalizedColumns.value)
  const rowTop = row * rowHeight.value
  const rowBottom = rowTop + rowHeight.value
  if (rowTop < container.scrollTop) {
    container.scrollTop = rowTop
  } else if (rowBottom > container.scrollTop + container.clientHeight) {
    container.scrollTop = rowBottom - container.clientHeight
  }
  syncViewport()
}

function scrollToEnd(): void {
  const container = containerRef.value
  if (!container) return
  container.scrollTop = Math.max(0, contentSize.value - container.clientHeight)
  syncViewport()
}

defineExpose({
  scrollToEnd,
})

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
    class="virtual-thumbnail-grid"
    role="navigation"
    :aria-label="ariaLabel"
    :style="containerStyle"
    @scroll.passive="syncViewport"
  >
    <div class="virtual-thumbnail-grid__inner" :style="innerStyle">
      <ProductThumbnailGrid
        class="virtual-thumbnail-grid__window"
        :style="windowStyle"
        :aria-label="ariaLabel"
        :columns="normalizedColumns"
        :items="visibleItems"
        @select="$emit('select', $event)"
      />
    </div>
  </div>
</template>

<style scoped>
.virtual-thumbnail-grid {
  min-block-size: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
}

.virtual-thumbnail-grid__inner {
  position: relative;
  inline-size: 100%;
}

.virtual-thumbnail-grid__window {
  position: absolute;
  inset-block-start: 0;
  inset-inline: 0;
  will-change: transform;
}
</style>
