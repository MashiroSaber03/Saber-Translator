<script setup lang="ts">
import { ref, watch } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
const props = defineProps<{ src: string; alt: string; badge?: string }>()
const emit = defineEmits<{ size: [width: number, height: number] }>()
const failed = ref(false)
watch(
  () => props.src,
  () => {
    failed.value = false
  }
)
function loaded(event: Event) {
  const image = event.target as HTMLImageElement
  emit('size', image.naturalWidth, image.naturalHeight)
}
</script>
<template>
  <div class="reader-image">
    <img
      v-if="!failed"
      class="virtual-page-stream__image"
      :src="src"
      :alt="alt"
      draggable="false"
      decoding="async"
      @load="loaded"
      @error="failed = true"
    />
    <div v-else class="reader-image__error" role="status">
      <span>{{ alt }}加载失败</span>
      <UiButton variant="inverse" size="sm" @click.stop="failed = false">重试</UiButton>
    </div>
    <span v-if="badge" class="reader-image__badge">{{ badge }}</span>
  </div>
</template>
<style scoped>
.reader-image {
  position: relative;
  width: 100%;
  height: 100%;
}

.reader-image img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.reader-image__error {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: var(--color-surface-inverse-raised);
  color: var(--color-text-inverse);
}

.reader-image__badge {
  position: absolute;
  top: 8px;
  right: 8px;
  padding: 4px 8px;
  border-radius: 4px;
  background: var(--color-overlay-scrim);
  color: var(--color-text-inverse);
  font-size: 12px;
}
</style>
