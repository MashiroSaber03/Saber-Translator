<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'
import VirtualPageStream from '@/components/virtual/VirtualPageStream.vue'
import type { V2PageSummary } from '@/api/v2/content'
import { DEFAULT_READER_SETTINGS, type ReaderSettings } from './readerSettings'
import { fitScale, type ReaderPosition } from './readerLayout'
import ReaderImage from './ReaderImage.vue'
import { usePublicUserAccess } from '@/composables/usePublicUserAccess'

const props = withDefaults(
  defineProps<{
    images: V2PageSummary[]
    viewMode: 'original' | 'translated'
    isLoading: boolean
    settings?: ReaderSettings
    position?: ReaderPosition
    navigationId?: number
    group?: number[]
    neighbours?: number[]
    canPrev?: boolean
    canNext?: boolean
  }>(),
  { settings: () => ({ ...DEFAULT_READER_SETTINGS }), group: () => [0], neighbours: () => [] }
)
const emit = defineEmits<{
  positionChange: [position: ReaderPosition]
  goTranslate: []
  toggleControls: []
  navigate: [delta: number]
  size: [id: string, width: number, height: number]
}>()
const publicAccess = usePublicUserAccess()
const canTranslate = computed(() => publicAccess.featureAllowed('translation'))
const paged = computed(() => ['single', 'double'].includes(props.settings.layout))
const pageButtons = computed(() => (['left', 'right'] as const).map(side => {
  const delta = (side === 'left' ? -1 : 1) * (props.settings.direction === 'rtl' ? -1 : 1)
  return { side, delta, label: delta < 0 ? '上一页' : '下一页', enabled: delta < 0 ? props.canPrev : props.canNext }
}))
const fit = computed(() => props.settings.fits[props.settings.layout])
const stage = ref<HTMLElement | null>(null)
const viewport = ref({ width: 1, height: 1 })
let observer: ResizeObserver | undefined
let restoring = false
let restoreSequence = 0
const items = computed(() =>
  props.images.map((page, index) => ({
    id: page.id,
    alt: `第 ${index + 1} 页`,
    width: page.width || 800,
    height: page.height || 1200,
    badge: props.viewMode === 'translated' && !page.translatedUrl ? '未翻译' : undefined,
    url: props.viewMode === 'translated' ? page.translatedUrl || page.sourceUrl : page.sourceUrl,
  }))
)
const current = computed(() => {
  const selected = props.group
    .map(i => items.value[i])
    .filter((p): p is NonNullable<typeof p> => Boolean(p))
  return props.settings.direction === 'rtl' ? selected.reverse() : selected
})
const natural = computed(() => ({
  width: current.value.reduce((sum, p) => sum + p.width, 0),
  height: Math.max(1, ...current.value.map(p => p.height)),
}))
const scale = computed(() =>
  fitScale(
    natural.value.width,
    natural.value.height,
    viewport.value.width,
    viewport.value.height,
    fit.value
  )
)
const spreadStyle = computed(() => ({
  width: `${Math.max(viewport.value.width, natural.value.width * scale.value)}px`,
  height: `${Math.max(viewport.value.height, natural.value.height * scale.value)}px`,
}))
let preloaded: HTMLImageElement[] = []
watch(
  () => [paged.value, props.neighbours.map(i => items.value[i]?.url).join('|')],
  () => {
    preloaded = []
    if (paged.value)
      for (const i of props.neighbours) {
        const item = items.value[i]
        if (item) {
          const image = new Image()
          image.src = item.url
          preloaded.push(image)
        }
      }
  },
  { immediate: true }
)
function measure() {
  if (stage.value)
    viewport.value = { width: stage.value.clientWidth, height: stage.value.clientHeight }
}
watch(stage, el => {
  observer?.disconnect()
  if (el) {
    measure()
    observer?.observe(el)
  }
})
watch(
  [stage, () => props.navigationId, scale, () => props.settings.direction],
  async () => {
    const sequence = ++restoreSequence
    const fraction = props.position?.fraction ?? 0
    restoring = true
    await nextTick()
    if (sequence !== restoreSequence) return
    if (stage.value && paged.value) {
      stage.value.scrollTop = fraction * natural.value.height * scale.value
      stage.value.scrollLeft = props.settings.direction === 'rtl' ? stage.value.scrollWidth : 0
    }
    requestAnimationFrame(() => {
      if (sequence === restoreSequence) restoring = false
    })
  },
  { flush: 'post' }
)
function scrollPage() {
  if (restoring || !stage.value || !props.position) return
  emit('positionChange', {
    ...props.position,
    fraction: Math.min(1, stage.value.scrollTop / Math.max(1, natural.value.height * scale.value)),
  })
}
let pointer: { x: number; y: number; left: number; top: number; type: string } | null = null
let moved = false
let multiTouch = false
function down(event: PointerEvent) {
  if (pointer) {
    multiTouch = true
    return
  }
  if (event.button !== 0) return
  multiTouch = false
  moved = false
  pointer = {
    x: event.clientX,
    y: event.clientY,
    left: stage.value?.scrollLeft ?? 0,
    top: stage.value?.scrollTop ?? 0,
    type: event.pointerType,
  }
}
function move(event: PointerEvent) {
  if (!pointer) return
  const dx = event.clientX - pointer.x,
    dy = event.clientY - pointer.y
  if (Math.hypot(dx, dy) > 8) moved = true
  if (paged.value && pointer.type === 'mouse' && moved && stage.value) {
    stage.value.scrollLeft = pointer.left - dx
    stage.value.scrollTop = pointer.top - dy
  }
}
function up(event: PointerEvent) {
  if (!pointer) return
  const dx = event.clientX - pointer.x,
    dy = event.clientY - pointer.y
  if (
    paged.value &&
    pointer.type === 'touch' &&
    !multiTouch &&
    stage.value &&
    stage.value.scrollWidth <= stage.value.clientWidth + 1 &&
    Math.abs(dx) > 50 &&
    Math.abs(dx) > Math.abs(dy) * 1.5
  ) {
    emit('navigate', (dx < 0 ? 1 : -1) * (props.settings.direction === 'rtl' ? -1 : 1))
    moved = true
  }
  pointer = null
}
function cancelPointer() {
  pointer = null
  moved = true
}
function click(event: MouseEvent) {
  if (
    moved ||
    multiTouch ||
    window.getSelection()?.toString() ||
    (event.target as Element).closest('button,input,select,a')
  )
    return
  const bounds = (event.currentTarget as HTMLElement).getBoundingClientRect()
  const x = (event.clientX - bounds.left) / bounds.width
  if (!paged.value || (x >= 0.25 && x <= 0.75)) emit('toggleControls')
  else emit('navigate', (x > 0.75 ? 1 : -1) * (props.settings.direction === 'rtl' ? -1 : 1))
}
onMounted(() => {
  if (typeof ResizeObserver !== 'undefined') {
    observer = new ResizeObserver(measure)
    if (stage.value) observer.observe(stage.value)
  }
  measure()
})
onUnmounted(() => {
  restoreSequence++
  observer?.disconnect()
  preloaded = []
})
</script>

<template>
  <main class="reader-canvas" :style="{ background: settings.bgColor }">
    <div v-if="isLoading" class="reader-canvas__message">
      <UiSpinner label="正在加载阅读内容" :decorative="false" />
    </div>
    <ProductEmptyState
      v-else-if="!images.length"
      class="reader-canvas__message"
      title="暂无图片"
      description="该章节还没有图片"
      variant="inverse"
    >
      <template #actions>
        <UiButton v-if="canTranslate" @click="emit('goTranslate')">进入翻译</UiButton>
      </template>
    </ProductEmptyState>
    <div
      v-else
      class="reader-canvas__surface"
      @pointerdown="down"
      @pointermove="move"
      @pointerup="up"
      @pointercancel="cancelPointer"
      @pointerleave="cancelPointer"
      @click="click"
    >
      <div
        v-if="paged"
        ref="stage"
        class="reader-canvas__paged"
        :class="{ 'reader-canvas__paged--swipe': fit === 'screen' || fit === 'width' }"
        @scroll.passive="scrollPage"
      >
        <div class="reader-canvas__spread" :style="spreadStyle">
          <figure
            v-for="item in current"
            :key="item.id"
            :data-page-id="item.id"
            :style="{ width: `${item.width * scale}px`, height: `${item.height * scale}px` }"
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
      <VirtualPageStream
        v-else
        class="reader-canvas__stream"
        :items="items"
        :gap="settings.imageGap"
        :fit="fit"
        :image-width="settings.imageWidth"
        :direction="settings.direction"
        :horizontal="settings.layout === 'horizontal'"
        :position="position"
        :navigation-id="navigationId"
        @position-change="emit('positionChange', $event)"
        @size="(id, w, h) => emit('size', id, w, h)"
      />
    </div>
    <template v-if="paged && !isLoading && images.length">
      <UiButton
        v-for="button in pageButtons"
        :key="button.side"
        class="reader-canvas__page-button"
        :class="`reader-canvas__page-button--${button.side}`"
        variant="ghost"
        :aria-label="button.label"
        :title="button.label"
        :disabled="!button.enabled"
        @click.stop="emit('navigate', button.delta)"
      >
        <UiIcon :name="button.side === 'left' ? 'chevron-left' : 'chevron-right'" :size="28" />
      </UiButton>
    </template>
  </main>
</template>
<style scoped>
.reader-canvas {
  position: relative;
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
}

.reader-canvas__page-button {
  --ui-button-padding: 0;
  --ui-button-radius: 12px;
  --ui-button-ghost-background: var(--color-overlay-scrim);
  --ui-button-ghost-color: var(--color-text-inverse);
  --ui-button-ghost-border: 1px solid var(--color-overlay-inverse-muted);
  --ui-button-ghost-hover-background: var(--color-surface-inverse-raised);
  --ui-button-ghost-hover-color: var(--color-text-inverse);
  --ui-button-disabled-background: var(--color-overlay-scrim);
  --ui-button-disabled-color: var(--color-text-inverse);
  --ui-button-disabled-border: 1px solid var(--color-overlay-inverse-muted);
  --ui-button-disabled-opacity: 0.3;

  position: absolute;
  top: 50%;
  translate: 0 -50%;
  width: 44px;
  height: 56px;
  z-index: var(--z-local-overlay);
  opacity: 0.75;
}

.reader-canvas__page-button:hover:not(:disabled),
.reader-canvas__page-button:focus-visible {
  opacity: 1;
}

.reader-canvas__page-button--left {
  left: 12px;
}

.reader-canvas__page-button--right {
  right: 12px;
}

.reader-canvas__surface,
.reader-canvas__paged {
  width: 100%;
  height: 100%;
  min-height: 0;
}

.reader-canvas__paged {
  overflow: auto;
  overscroll-behavior: contain;
  overflow-anchor: none;
}

.reader-canvas__paged--swipe {
  touch-action: pan-y pinch-zoom;
}

.reader-canvas__spread {
  display: flex;
  justify-content: center;
  align-items: center;
}

.reader-canvas__spread figure {
  flex-shrink: 0;
  margin: 0;
}

.reader-canvas__message {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
