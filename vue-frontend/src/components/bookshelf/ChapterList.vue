<script setup lang="ts">
/**
 * 章节列表组件
 * 支持拖拽排序、编辑、删除等操作
 */

import { ref } from 'vue'
import type { ChapterData } from '@/types'

interface Props {
  chapters: ChapterData[]
}

const props = defineProps<Props>()

const emit = defineEmits<{
  edit: [chapterId: string]
  delete: [chapterId: string]
  translate: [chapterId: string]
  read: [chapterId: string]
  reorder: [chapterIds: string[]]
}>()

// 拖拽状态
const draggedIndex = ref<number | null>(null)
const dragOverIndex = ref<number | null>(null)

// 开始拖拽
function handleDragStart(event: DragEvent, index: number) {
  draggedIndex.value = index
  if (event.dataTransfer) {
    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData('text/plain', index.toString())
  }
}

// 拖拽经过
function handleDragOver(event: DragEvent, index: number) {
  event.preventDefault()
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = 'move'
  }
  dragOverIndex.value = index
}

// 拖拽离开
function handleDragLeave() {
  dragOverIndex.value = null
}

// 放置
function handleDrop(event: DragEvent, targetIndex: number) {
  event.preventDefault()
  
  if (draggedIndex.value === null || draggedIndex.value === targetIndex) {
    draggedIndex.value = null
    dragOverIndex.value = null
    return
  }

  // 重新排序
  const newOrder = [...props.chapters]
  const [removed] = newOrder.splice(draggedIndex.value, 1)
  if (!removed) return
  newOrder.splice(targetIndex, 0, removed)

  // 发送新顺序
  emit('reorder', newOrder.map(c => c.id))

  draggedIndex.value = null
  dragOverIndex.value = null
}

// 拖拽结束
function handleDragEnd() {
  draggedIndex.value = null
  dragOverIndex.value = null
}
</script>

<template>
  <div class="chapters-list">
    <div
      v-for="(chapter, index) in chapters"
      :key="chapter.id"
      class="chapter-item"
      :class="{
        dragging: draggedIndex === index,
        'drag-over': dragOverIndex === index && draggedIndex !== index,
      }"
      draggable="true"
      @dragstart="handleDragStart($event, index)"
      @dragover="handleDragOver($event, index)"
      @dragleave="handleDragLeave"
      @drop="handleDrop($event, index)"
      @dragend="handleDragEnd"
    >
      <!-- 拖拽手柄 -->
      <div class="drag-handle" title="拖拽排序">
        <span>⋮⋮</span>
      </div>

      <!-- 章节信息 -->
      <div class="chapter-info">
        <span class="chapter-order">{{ index + 1 }}</span>
        <span class="chapter-title">{{ chapter.title }}</span>
        <span v-if="chapter.imageCount > 0" class="chapter-count">
          {{ chapter.imageCount }} 张图片
        </span>
        <span v-if="chapter.hasSession" class="chapter-status" title="有保存的翻译进度">
          💾
        </span>
      </div>

      <!-- 操作按钮 -->
      <div class="chapter-actions">
        <button
          class="btn btn-xs btn-primary"
          title="翻译"
          @click.stop="emit('translate', chapter.id)"
        >
          翻译
        </button>
        <button
          v-if="chapter.hasSession"
          class="btn btn-xs btn-secondary"
          title="阅读"
          @click.stop="emit('read', chapter.id)"
        >
          阅读
        </button>
        <button
          class="btn btn-xs btn-icon"
          title="编辑"
          @click.stop="emit('edit', chapter.id)"
        >
          ✏️
        </button>
        <button
          class="btn btn-xs btn-icon btn-danger-icon"
          title="删除"
          @click.stop="emit('delete', chapter.id)"
        >
          🗑️
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chapters-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chapter-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: var(--bg-secondary, #f8f9fa);
  border-radius: 8px;
  transition: all 0.2s;
  cursor: default;
}

.chapter-item:hover {
  background: var(--bg-tertiary, #f0f0f0);
}

.chapter-item.dragging {
  opacity: 0.5;
  background: var(--primary-light, #e8ecff);
}

.chapter-item.drag-over {
  border-top: 2px solid var(--primary-color, #667eea);
  margin-top: -2px;
}

.drag-handle {
  cursor: grab;
  color: var(--text-tertiary, #bbb);
  font-size: 16px;
  padding: 4px;
  user-select: none;
}

.drag-handle:active {
  cursor: grabbing;
}

.chapter-info {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.chapter-order {
  flex-shrink: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--primary-color, #667eea);
  color: #fff;
  font-size: 12px;
  font-weight: 600;
  border-radius: 50%;
}

.chapter-title {
  flex: 1;
  font-size: 14px;
  color: var(--text-primary, #333);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chapter-count {
  flex-shrink: 0;
  font-size: 12px;
  color: var(--text-secondary, #999);
}

.chapter-status {
  flex-shrink: 0;
  font-size: 14px;
}

.chapter-actions {
  display: flex;
  gap: 8px;
  opacity: 0;
  transition: opacity 0.2s;
}

.chapter-item:hover .chapter-actions {
  opacity: 1;
}

.btn-xs {
  padding: 4px 10px;
  font-size: 12px;
  border-radius: 4px;
}

.btn-icon {
  padding: 4px 8px;
  background: transparent;
  border: none;
}

.btn-icon:hover {
  background: var(--bg-tertiary, #e0e0e0);
}

.btn-danger-icon:hover {
  background: #fee;
}
</style>
