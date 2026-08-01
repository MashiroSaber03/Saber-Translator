<script setup lang="ts">
import type { BubbleState } from '@/types/bubble'
import { computed, ref } from 'vue'

const props = defineProps<{
  debugBubbles?: readonly BubbleState[]
  imageAlt: string
  imageHeight?: number
  imageSize: number
  imageUrl: string
  imageWidth?: number
  showDetectionDebug?: boolean
}>()

const imageStyle = computed(() => ({
  width: `${props.imageSize}%`,
}))

const loadedImageSize = ref({ width: 0, height: 0 })

const overlayWidth = computed(() => (
  Number.isFinite(props.imageWidth) && Number(props.imageWidth) > 0
    ? Number(props.imageWidth)
    : loadedImageSize.value.width
))

const overlayHeight = computed(() => (
  Number.isFinite(props.imageHeight) && Number(props.imageHeight) > 0
    ? Number(props.imageHeight)
    : loadedImageSize.value.height
))

const debugRects = computed(() => (props.debugBubbles ?? []).flatMap((bubble, index) => {
  const [x1, y1, x2, y2] = bubble.coords
  if (![x1, y1, x2, y2].every(Number.isFinite) || x2 <= x1 || y2 <= y1) return []
  return [{
    key: bubble.backendBubbleId ?? `${index}-${x1}-${y1}-${x2}-${y2}`,
    x: x1,
    y: y1,
    width: x2 - x1,
    height: y2 - y1,
  }]
}))

const showDebugOverlay = computed(() => (
  props.showDetectionDebug === true
  && overlayWidth.value > 0
  && overlayHeight.value > 0
  && debugRects.value.length > 0
))

const overlayViewBox = computed(() => `0 0 ${overlayWidth.value} ${overlayHeight.value}`)

function handleImageLoad(event: Event): void {
  const image = event.currentTarget
  if (!(image instanceof HTMLImageElement)) return
  loadedImageSize.value = {
    width: image.naturalWidth,
    height: image.naturalHeight,
  }
}
</script>

<template>
  <figure class="result-image-canvas" aria-label="翻译结果图片">
    <div class="result-image-canvas__image-frame">
      <div class="result-image-canvas__image-layer" :style="imageStyle">
        <img
          class="result-image-canvas__image"
          :src="imageUrl"
          :alt="imageAlt"
          @load="handleImageLoad"
        />
        <svg
          v-if="showDebugOverlay"
          class="result-image-canvas__debug-overlay"
          :viewBox="overlayViewBox"
          preserveAspectRatio="none"
          aria-hidden="true"
          data-testid="detection-debug-overlay"
        >
          <rect
            v-for="rect in debugRects"
            :key="rect.key"
            class="result-image-canvas__debug-box"
            :x="rect.x"
            :y="rect.y"
            :width="rect.width"
            :height="rect.height"
          />
        </svg>
      </div>
    </div>
  </figure>
</template>

<style scoped>
.result-image-canvas {
  --result-image-canvas-frame-shadow: var(--shadow-soft);

  display: flex;
  justify-content: center;
  width: 100%;
  margin: 0 0 20px;
  overflow: hidden;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  background-color: var(--color-surface-app);
  box-shadow: 0 2px 8px var(--result-image-canvas-frame-shadow);
  text-align: center;
}

.result-image-canvas__image-frame {
  display: flex;
  justify-content: center;
  width: 100%;
  max-width: 100%;
  text-align: center;
}

.result-image-canvas__image-layer {
  position: relative;
  max-width: 100%;
  transition: width 0.3s ease;
}

.result-image-canvas__image {
  display: block;
  width: 100%;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
  border: 0;
  object-fit: contain;
}

.result-image-canvas__debug-overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.result-image-canvas__debug-box {
  fill: none;
  stroke: var(--color-status-error);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}

@media (--breakpoint-md-down) {
  .result-image-canvas__image-frame {
    max-width: 280px;
    margin-top: 25px;
  }
}
</style>
