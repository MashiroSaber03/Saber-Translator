<template>
  <div v-if="visible" class="edit-thumbnails-panel">
    <div
      ref="viewportRef"
      class="edit-thumbnails-panel__viewport"
      role="list"
      aria-label="编辑图片缩略图导航"
      @scroll="updateWindow"
    >
      <div
        class="edit-thumbnails-panel__track"
        :style="{ width: `${images.length * ITEM_WIDTH}px` }"
      >
        <UiButton
          v-for="item in visibleItems"
          :key="item.image.id"
          class="edit-thumbnails-panel__item"
          :class="{ 'edit-thumbnails-panel__item--selected': item.index === currentImageIndex }"
          :style="{ transform: `translateX(${item.index * ITEM_WIDTH}px)` }"
          variant="card-action"
          role="listitem"
          :aria-label="`切换到图片 ${item.index + 1}`"
          :aria-current="item.index === currentImageIndex ? 'page' : undefined"
          @click="emit('switch-to-image', item.index)"
        >
          <img
            class="edit-thumbnails-panel__image"
            :src="item.image.thumbnailSourceUrl"
            :alt="`图片 ${item.index + 1}`"
            loading="lazy"
            decoding="async"
          >
          <span class="edit-thumbnails-panel__label">{{ item.index + 1 }}</span>
        </UiButton>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import type { ImageData } from '@/types/image'

const ITEM_WIDTH = 70
const BUFFER = 6

const props = defineProps<{
  visible: boolean
  images: ImageData[]
  currentImageIndex: number
}>()

const emit = defineEmits<{
  (e: 'switch-to-image', index: number): void
}>()

const viewportRef = ref<HTMLElement | null>(null)
const firstVisible = ref(0)
const visibleCount = ref(20)

const visibleItems = computed(() => {
  const start = Math.max(0, firstVisible.value - BUFFER)
  const end = Math.min(
    props.images.length,
    firstVisible.value + visibleCount.value + BUFFER,
  )
  return props.images
    .slice(start, end)
    .map((image, offset) => ({ image, index: start + offset }))
})

function updateWindow(): void {
  const viewport = viewportRef.value
  if (!viewport) return
  firstVisible.value = Math.floor(viewport.scrollLeft / ITEM_WIDTH)
  visibleCount.value = Math.ceil(viewport.clientWidth / ITEM_WIDTH) + 1
}

function revealCurrent(): void {
  const viewport = viewportRef.value
  if (!viewport || !props.visible) return
  const left = props.currentImageIndex * ITEM_WIDTH
  const right = left + ITEM_WIDTH
  if (left < viewport.scrollLeft) viewport.scrollLeft = left
  else if (right > viewport.scrollLeft + viewport.clientWidth) {
    viewport.scrollLeft = right - viewport.clientWidth
  }
  updateWindow()
}

onMounted(updateWindow)
watch(
  () => [props.visible, props.currentImageIndex, props.images.length],
  () => void nextTick(revealCurrent),
)
</script>

<style scoped>
.edit-thumbnails-panel {
  position: relative;
  flex-shrink: 0;
  width: auto;
  padding: 10px 15px;
  border-bottom: 1px solid var(--color-overlay-inverse-subtle);
  background: color-mix(in srgb, var(--color-overlay-backdrop-solid) 30%, transparent);
}

.edit-thumbnails-panel__viewport {
  width: 100%;
  height: 88px;
  overflow-x: auto;
  overflow-y: hidden;
  scrollbar-width: thin;
}

.edit-thumbnails-panel__track {
  position: relative;
  height: 80px;
}

.edit-thumbnails-panel__item {
  position: absolute;
  top: 0;
  left: 0;
  width: 60px;
  height: 80px;
  overflow: hidden;
  padding: 0;
  border: 2px solid transparent;
  border-radius: 6px;
  background: var(--color-surface-subtle);
  color: var(--color-text-inverse);
  cursor: pointer;
}

.edit-thumbnails-panel__item--selected {
  border-color: var(--color-action-primary);
  box-shadow: 0 0 0 2px var(--color-focus-brand-subtle);
}

.edit-thumbnails-panel__image {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: cover;
}

.edit-thumbnails-panel__label {
  position: absolute;
  right: 0;
  bottom: 0;
  left: 0;
  padding: 2px;
  background: linear-gradient(transparent, var(--color-overlay-backdrop-strong));
  font-size: 10px;
  text-align: center;
}
</style>
