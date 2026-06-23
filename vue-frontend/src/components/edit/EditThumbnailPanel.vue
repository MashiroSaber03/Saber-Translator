<!--
  编辑模式缩略图面板组件
  显示所有图片的缩略图，支持点击切换
-->
<template>
  <div v-if="visible" class="edit-thumbnails-panel">
    <div class="thumbnails-scroll">
      <UiButton
        v-for="(image, index) in images"
        :key="image.id"
        variant="toolbar"
        class="edit-thumbnail-item"
        :class="{ active: index === currentImageIndex }"
        :aria-label="`切换到图片 ${index + 1}`"
        :aria-pressed="index === currentImageIndex"
        @click="$emit('switch-to-image', index)"
      >
        <img :src="image.translatedDataURL || image.originalDataURL" :alt="`图片 ${index + 1}`" />
        <span class="thumb-index">{{ index + 1 }}</span>
      </UiButton>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 编辑模式缩略图面板组件
 * 显示所有图片的缩略图并发出切换请求。
 */
import UiButton from '@/components/ui/UiButton.vue'
import type { ImageData } from '@/types/image'

// ============================================================
// Props
// ============================================================

defineProps<{
  /** 是否显示 */
  visible: boolean
  /** 图片列表 */
  images: ImageData[]
  /** 当前图片索引 */
  currentImageIndex: number
}>()

// ============================================================
// Emits
// ============================================================

defineEmits<{
  /** 切换到指定图片 */
  (e: 'switch-to-image', index: number): void
}>()
</script>

<style scoped>
.edit-thumbnails-panel {
  --edit-thumbnail-panel-surface-subtle: rgba(0, 0, 0, .7);
}

/* 编辑模式缩略图面板 */
.edit-thumbnails-panel {
  /* owner tokens: edit-thumbnail-panel */
  --edit-thumbnail-panel-border-default: rgba(255, 255, 255, .1);
  --edit-thumbnail-panel-border-strong: rgba(255, 255, 255, .5);
  --edit-thumbnail-panel-shadow-default: rgba(102, 126, 234, .5);
  --edit-thumbnail-panel-surface-base: rgba(0, 0, 0, .3);
  --edit-thumbnail-panel-surface-raised: rgba(255, 255, 255, .1);
  --edit-thumbnail-panel-surface-muted: rgba(255, 255, 255, .3);

  position: relative;
  width: auto;
  background: var(--edit-thumbnail-panel-surface-base);
  padding: 10px 15px;
  border-bottom: 1px solid var(--edit-thumbnail-panel-border-default);
  flex-shrink: 0;
}

.thumbnails-scroll {
  display: flex;
  flex-direction: row;
  gap: 10px;
  overflow: auto hidden;
  padding: 5px 0;
}

.thumbnails-scroll::-webkit-scrollbar {
  height: 6px;
}

.thumbnails-scroll::-webkit-scrollbar-track {
  background: var(--edit-thumbnail-panel-surface-raised);
  border-radius: 3px;
}

.thumbnails-scroll::-webkit-scrollbar-thumb {
  background: var(--edit-thumbnail-panel-surface-muted);
  border-radius: 3px;
}

.edit-thumbnail-item {
  flex-shrink: 0;
  width: 60px;
  height: 80px;
  border-radius: 6px;
  overflow: hidden;
  cursor: pointer;
  border: 2px solid transparent;
  transition: all 0.2s;
  position: relative;
}

.edit-thumbnail-item:hover {
  border-color: var(--edit-thumbnail-panel-border-strong);
  transform: scale(1.05);
}

.edit-thumbnail-item.active {
  border-color: var(--color-border-brand-gradient);
  box-shadow: 0 0 10px var(--edit-thumbnail-panel-shadow-default);
}

.edit-thumbnail-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.edit-thumbnail-item .thumb-index {
  position: absolute;
  bottom: 2px;
  right: 2px;
  background: var(--edit-thumbnail-panel-surface-subtle);
  color: var(--color-text-inverse);
  font-size: 10px;
  padding: 1px 4px;
  border-radius: 3px;
}
</style>
