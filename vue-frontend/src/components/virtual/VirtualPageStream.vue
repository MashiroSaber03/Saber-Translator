<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from 'vue'

import { variableVirtualWindow } from './virtualWindow'

export interface VirtualPageStreamItem {
  alt: string
  badge?: string
  height: number
  id: string
  label?: string
  url: string
  width: number
}

const props = withDefaults(defineProps<{
  gap?: number
  items: VirtualPageStreamItem[]
  overscanScreens?: number
}>(), {
  gap: 16,
  overscanScreens: 2,
})

const emit = defineEmits<{
  visibleChange: [ids: string[]]
}>()

const containerRef = ref<HTMLElement | null>(null)
const viewportHeight = ref(0)
const viewportWidth = ref(0)
const scrollTop = ref(0)
const visibleIds = ref<Set<string>>(new Set())
let resizeObserver: ResizeObserver | null = null
let intersectionObserver: IntersectionObserver | null = null

const itemSizes = computed(() => props.items.map(item => {
  const usableWidth = Math.max(1, viewportWidth.value)
  const renderedHeight = item.width > 0
    ? usableWidth * item.height / item.width
    : usableWidth
  return Math.max(1, renderedHeight) + props.gap
}))
const windowState = computed(() => variableVirtualWindow(
  itemSizes.value,
  scrollTop.value,
  viewportHeight.value,
  viewportHeight.value * props.overscanScreens,
))
const renderedItems = computed(() => props.items.slice(
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
  viewportHeight.value = container.clientHeight
  viewportWidth.value = container.clientWidth
  scrollTop.value = container.scrollTop
}

function rebuildIntersectionObserver(): void {
  intersectionObserver?.disconnect()
  visibleIds.value = new Set()
  const root = containerRef.value
  if (!root || typeof IntersectionObserver === 'undefined') return
  intersectionObserver = new IntersectionObserver(entries => {
    const next = new Set(visibleIds.value)
    for (const entry of entries) {
      const id = (entry.target as HTMLElement).dataset.pageId
      if (!id) continue
      if (entry.isIntersecting) next.add(id)
      else next.delete(id)
    }
    visibleIds.value = next
    emit('visibleChange', props.items
      .filter(item => next.has(item.id))
      .map(item => item.id))
  }, { root })
  root.querySelectorAll<HTMLElement>('[data-page-id]').forEach(element => {
    intersectionObserver?.observe(element)
  })
}

watch(renderedItems, () => nextTick(rebuildIntersectionObserver))
watch(
  () => props.items.map(item => item.id).join('\u0000'),
  () => {
    if (containerRef.value) containerRef.value.scrollTop = 0
    scrollTop.value = 0
  },
)
onMounted(() => {
  syncViewport()
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(syncViewport)
    if (containerRef.value) resizeObserver.observe(containerRef.value)
  }
  nextTick(rebuildIntersectionObserver)
})
onBeforeUnmount(() => {
  resizeObserver?.disconnect()
  intersectionObserver?.disconnect()
})
</script>

<template>
  <div
    ref="containerRef"
    class="virtual-page-stream"
    @scroll.passive="syncViewport"
  >
    <div class="virtual-page-stream__inner" :style="innerStyle">
      <div class="virtual-page-stream__window" :style="windowStyle">
        <figure
          v-for="item in renderedItems"
          :key="item.id"
          class="virtual-page-stream__page"
          :data-page-id="item.id"
          :style="{
            aspectRatio: `${Math.max(1, item.width)} / ${Math.max(1, item.height)}`,
            marginBlockEnd: `${gap}px`,
          }"
        >
          <img
            class="virtual-page-stream__image"
            :src="item.url"
            :alt="item.alt"
            loading="lazy"
            decoding="async"
          >
          <span v-if="item.label" class="virtual-page-stream__label">
            {{ item.label }}
          </span>
          <span v-if="item.badge" class="virtual-page-stream__badge">
            {{ item.badge }}
          </span>
        </figure>
      </div>
    </div>
  </div>
</template>

<style scoped>
.virtual-page-stream {
  block-size: 100%;
  min-block-size: 0;
  overflow: auto;
  overscroll-behavior: contain;
}

.virtual-page-stream__inner {
  position: relative;
  inline-size: 100%;
}

.virtual-page-stream__window {
  position: absolute;
  inset-block-start: 0;
  inset-inline: 0;
  will-change: transform;
}

.virtual-page-stream__page {
  position: relative;
  margin-block-start: 0;
  margin-inline: 0;
  inline-size: 100%;
}

.virtual-page-stream__label {
  position: absolute;
  inset-block-start: 8px;
  inset-inline-start: 8px;
  padding: 4px 8px;
  border-radius: 4px;
  color: var(--color-text-inverse);
  background: var(--color-overlay-scrim);
  font-size: 12px;
  opacity: 0;
  transition: opacity 0.2s;
}

.virtual-page-stream__page:hover .virtual-page-stream__label {
  opacity: 1;
}

.virtual-page-stream__badge {
  position: absolute;
  inset-block-start: 8px;
  inset-inline-end: 8px;
  padding: 5px 9px;
  border-radius: 999px;
  color: var(--color-text-inverse);
  background: var(--color-status-warning);
  box-shadow: 0 2px 8px var(--shadow-medium);
  font-size: 12px;
  font-weight: 700;
}

.virtual-page-stream__image {
  display: block;
  inline-size: 100%;
  block-size: auto;
}
</style>
