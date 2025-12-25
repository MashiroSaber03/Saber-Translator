<!--
  编辑模式缩略图面板组件
  显示所有图片的缩略图，支持点击切换
  从 EditWorkspace.vue 拆分出来
-->
<template>
  <div v-if="visible" class="edit-thumbnails-panel">
    <div class="thumbnails-scroll">
      <div
        v-for="(image, index) in images"
        :key="image.id"
        class="edit-thumbnail-item"
        :class="{ active: index === currentImageIndex }"
        @click="$emit('switch-to-image', index)"
      >
        <img :src="image.translatedDataURL || image.originalDataURL" :alt="`图片 ${index + 1}`" />
        <span class="thumb-index">{{ index + 1 }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 编辑模式缩略图面板组件
 * 显示所有图片的缩略图，支持点击切换
 */
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
/* 编辑模式缩略图面板 */
.edit-thumbnails-panel {
  position: relative;
  width: auto;
  background: rgba(0,0,0,0.3);
  padding: 10px 15px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
  flex-shrink: 0;
}

.thumbnails-scroll {
  display: flex;
  flex-direction: row;
  gap: 10px;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 5px 0;
}

.thumbnails-scroll::-webkit-scrollbar {
  height: 6px;
}

.thumbnails-scroll::-webkit-scrollbar-track {
  background: rgba(255,255,255,0.1);
  border-radius: 3px;
}

.thumbnails-scroll::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.3);
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
  border-color: rgba(255,255,255,0.5);
  transform: scale(1.05);
}

.edit-thumbnail-item.active {
  border-color: #667eea;
  box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
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
  background: rgba(0,0,0,0.7);
  color: #fff;
  font-size: 10px;
  padding: 1px 4px;
  border-radius: 3px;
}
</style>
