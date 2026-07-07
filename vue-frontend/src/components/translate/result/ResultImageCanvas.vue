<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  imageAlt: string
  imageSize: number
  imageUrl: string
}>()

const imageStyle = computed(() => ({
  width: `${props.imageSize}%`,
}))
</script>

<template>
  <figure class="result-image-canvas" aria-label="翻译结果图片">
    <div class="result-image-canvas__image-frame">
      <img
        class="result-image-canvas__image"
        :src="imageUrl"
        :alt="imageAlt"
        :style="imageStyle"
      />
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
  max-width: 100%;
  text-align: center;
}

.result-image-canvas__image {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
  border: 0;
  object-fit: contain;
  transition: width 0.3s ease;
}

@media (--breakpoint-md-down) {
  .result-image-canvas__image-frame {
    max-width: 280px;
    margin-top: 25px;
  }
}
</style>
