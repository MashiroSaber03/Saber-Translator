<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { variableItemOffsets, variableVirtualWindow } from './virtualWindow'
import { fitScale, pageIndexAtOffset, type ReaderPosition } from '../reader/readerLayout'
import type { ReaderFit } from '../reader/readerSettings'
import ReaderImage from '../reader/ReaderImage.vue'

export interface VirtualPageStreamItem {
  alt: string
  badge?: string
  height: number
  id: string
  url: string
  width: number
}
const props = withDefaults(
  defineProps<{
    items: VirtualPageStreamItem[]
    gap?: number
    overscanScreens?: number
    horizontal?: boolean
    direction?: 'ltr' | 'rtl'
    fit?: ReaderFit
    imageWidth?: number
    position?: ReaderPosition
    navigationId?: number
  }>(),
  { gap: 8, overscanScreens: 2, horizontal: false, direction: 'ltr', fit: 'width', imageWidth: 100 }
)
const emit = defineEmits<{
  positionChange: [position: ReaderPosition]
  size: [id: string, width: number, height: number]
}>()
const containerRef = ref<HTMLElement | null>(null)
const viewport = ref({ width: 1, height: 1 })
const scroll = ref(0)
let observer: ResizeObserver | undefined
let restoring = false
let restoreSequence = 0
let anchor: ReaderPosition | undefined
const rtl = computed(() => props.horizontal && props.direction === 'rtl')
const dimensions = computed(() =>
  props.items.map(item => {
    const scale = fitScale(
      item.width,
      item.height,
      (viewport.value.width * props.imageWidth) / 100,
      viewport.value.height,
      props.fit
    )
    return { width: Math.max(1, item.width * scale), height: Math.max(1, item.height * scale) }
  })
)
const sizes = computed(() =>
  dimensions.value.map(d => (props.horizontal ? d.width : d.height) + props.gap)
)
const offsets = computed(() => variableItemOffsets(sizes.value))
const extent = computed(() => (props.horizontal ? viewport.value.width : viewport.value.height))
const windowState = computed(() =>
  variableVirtualWindow(
    sizes.value,
    scroll.value,
    extent.value,
    extent.value * props.overscanScreens,
    offsets.value
  )
)
const rendered = computed(() =>
  props.items
    .slice(windowState.value.start, windowState.value.end)
    .map((item, i) => ({ item, index: i + windowState.value.start }))
)
const cross = computed(() =>
  dimensions.value.reduce(
    (max, d) => Math.max(max, props.horizontal ? d.height : d.width),
    props.horizontal ? viewport.value.height : viewport.value.width
  )
)
const innerStyle = computed(() => ({
  width: `${props.horizontal ? Math.max(viewport.value.width, windowState.value.totalSize) : cross.value}px`,
  height: `${props.horizontal ? cross.value : Math.max(viewport.value.height, windowState.value.totalSize)}px`,
}))
function itemStyle(index: number) {
  const d = dimensions.value[index]!
  const offset = offsets.value[index]!
  return {
    width: `${d.width}px`,
    height: `${d.height}px`,
    left: `${props.horizontal ? (rtl.value ? Math.max(viewport.value.width, windowState.value.totalSize) - offset - d.width - props.gap : offset) : (cross.value - d.width) / 2}px`,
    top: `${props.horizontal ? (cross.value - d.height) / 2 : offset}px`,
  }
}
function logicalScroll(): number {
  const el = containerRef.value
  if (!el) return 0
  return props.horizontal
    ? rtl.value
      ? el.scrollWidth - el.clientWidth - el.scrollLeft
      : el.scrollLeft
    : el.scrollTop
}
function onScroll() {
  scroll.value = Math.max(0, logicalScroll())
  if (restoring || !props.items.length) return
  const maximum = Math.max(0, windowState.value.totalSize - extent.value)
  const index =
    maximum > 0 && scroll.value >= maximum - 0.5
      ? props.items.length - 1
      : pageIndexAtOffset(offsets.value, scroll.value)
  anchor = {
    pageId: props.items[index]!.id,
    index,
    fraction: Math.min(
      1,
      Math.max(0, (scroll.value - offsets.value[index]!) / sizes.value[index]!)
    ),
  }
  emit('positionChange', anchor)
}
async function restore(position = props.position ?? anchor) {
  const sequence = ++restoreSequence
  restoring = true
  await nextTick()
  if (sequence !== restoreSequence) return
  const el = containerRef.value
  if (el) {
    const found = position ? props.items.findIndex(p => p.id === position.pageId) : 0
    const index = Math.max(
      0,
      Math.min(props.items.length - 1, found >= 0 ? found : (position?.index ?? 0))
    )
    const offset =
      (offsets.value[index] ?? 0) + (sizes.value[index] ?? 0) * (position?.fraction ?? 0)
    if (props.horizontal)
      el.scrollLeft = rtl.value ? el.scrollWidth - el.clientWidth - offset : offset
    else el.scrollTop = offset
    scroll.value = Math.max(0, logicalScroll())
    const item = props.items[index]
    if (item) {
      anchor = { pageId: item.id, index, fraction: position?.fraction ?? 0 }
      emit('positionChange', anchor)
    }
  }
  requestAnimationFrame(() => {
    if (sequence === restoreSequence) restoring = false
  })
}
function resize() {
  const el = containerRef.value
  if (!el) return
  if (viewport.value.width === el.clientWidth && viewport.value.height === el.clientHeight) return
  viewport.value = { width: Math.max(1, el.clientWidth), height: Math.max(1, el.clientHeight) }
  void restore()
}
function wheel(event: WheelEvent) {
  const el = containerRef.value
  if (
    !el ||
    !props.horizontal ||
    event.ctrlKey ||
    event.shiftKey ||
    event.deltaX ||
    el.scrollHeight > el.clientHeight + 1
  )
    return
  event.preventDefault()
  const delta =
    event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? el.clientWidth : 1)
  el.scrollLeft += rtl.value ? -delta : delta
}
watch(
  [
    () => props.navigationId,
    () => props.horizontal,
    () => props.direction,
    () => props.fit,
    () => props.imageWidth,
    () => props.gap,
    () => props.items.map(p => `${p.id}:${p.width}:${p.height}`).join('|'),
  ],
  () => void restore(),
  { flush: 'post' }
)
onMounted(() => {
  resize()
  if (typeof ResizeObserver !== 'undefined') {
    observer = new ResizeObserver(resize)
    if (containerRef.value) observer.observe(containerRef.value)
  }
  void restore()
})
onBeforeUnmount(() => {
  observer?.disconnect()
  restoreSequence++
})
</script>

<template>
  <div
    ref="containerRef"
    class="virtual-page-stream"
    :style="{
      overflowX: !horizontal && (fit === 'width' || fit === 'screen') ? 'hidden' : 'auto',
      overflowY: horizontal && (fit === 'height' || fit === 'screen') ? 'hidden' : 'auto',
    }"
    @scroll.passive="onScroll"
    @wheel="wheel"
  >
    <div class="virtual-page-stream__inner" :style="innerStyle">
      <figure
        v-for="{ item, index } in rendered"
        :key="item.id"
        class="virtual-page-stream__page"
        :data-page-id="item.id"
        :style="itemStyle(index)"
      >
        <ReaderImage
          :src="item.url"
          :alt="item.alt"
          :badge="item.badge"
          @size="(w, h) => emit('size', item.id, w, h)"
        />
      </figure>
    </div>
  </div>
</template>

<style scoped>
.virtual-page-stream {
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: auto;
  overscroll-behavior: contain;
  direction: ltr;
  overflow-anchor: none;
}

.virtual-page-stream__inner {
  position: relative;
}

.virtual-page-stream__page {
  position: absolute;
  margin: 0;
}
</style>
