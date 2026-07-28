<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

import ProductThumbnailGrid from '@/components/product/ProductThumbnailGrid.vue'
import type { ProductThumbnailGridItem } from '@/components/product/ProductThumbnailGrid.vue'
import { fixedVirtualWindow } from './virtualWindow'

const props = withDefaults(defineProps<{
  activeId?: string | number | null
  ariaLabel?: string
  itemHeight?: number
  items: ProductThumbnailGridItem[]
  overscanItems?: number
}>(), {
  activeId: null,
  ariaLabel: '漫画页面缩略图',
  itemHeight: 164,
  overscanItems: 5,
})

const emit = defineEmits<{
  select: [id: string | number]
  visibleChange: [ids: Array<string | number>]
}>()

const containerRef = ref<HTMLElement | null>(null)
const scrollTop = ref(0)
const viewportHeight = ref(0)
let resizeObserver: ResizeObserver | null = null

const windowState = computed(() => fixedVirtualWindow(
  props.items.length,
  props.itemHeight,
  scrollTop.value,
  viewportHeight.value,
  props.overscanItems,
))
const visibleItems = computed(() => props.items.slice(
  windowState.value.start,
  windowState.value.end,
))
const innerStyle = computed(() => ({
  height: `${windowState.value.totalSize}px`,
}))
const windowStyle = computed(() => ({
  transform: `translateY(${windowState.value.offset}px)`,
}))

function syncViewport(): void {
  const container = containerRef.value
  if (!container) return
  scrollTop.value = container.scrollTop
  viewportHeight.value = container.clientHeight
}

function handleSelect(id: string | number): void {
  emit('select', id)
}

function scrollActiveIntoView(): void {
  const index = props.items.findIndex(item => item.id === props.activeId)
  const container = containerRef.value
  if (index < 0 || !container) return
  const itemTop = index * props.itemHeight
  const itemBottom = itemTop + props.itemHeight
  if (itemTop < container.scrollTop) {
    container.scrollTop = itemTop
  } else if (itemBottom > container.scrollTop + container.clientHeight) {
    container.scrollTop = itemBottom - container.clientHeight
  }
  syncViewport()
}

watch(visibleItems, items => {
  emit('visibleChange', items.map(item => item.id))
}, { immediate: true })
watch(() => props.activeId, () => nextTick(scrollActiveIntoView))

onMounted(() => {
  syncViewport()
  resizeObserver = new ResizeObserver(syncViewport)
  if (containerRef.value) resizeObserver.observe(containerRef.value)
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
        @select="handleSelect"
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
