<script setup lang="ts">
/**
 * 网页导入入口按钮
 */
import { useWebImportStore } from '@/stores/webImportStore'

const webImportStore = useWebImportStore()

function handleClick() {
  webImportStore.openModal()
}
</script>

<template>
  <button
    class="web-import-btn"
    @click="handleClick"
    title="从网页导入漫画"
  >
    <span class="icon">🌐</span>
    <span class="text">从网页导入</span>
  </button>
</template>

<style scoped>
.web-import-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  background: var(--btn-secondary-bg, #f0f0f0);
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  color: var(--text-primary, #333);
  transition: all 0.2s ease;
}

.web-import-btn:hover {
  background: var(--btn-secondary-hover-bg, #e5e5e5);
  border-color: var(--border-color-hover, #ccc);
}

.web-import-btn:active {
  transform: scale(0.98);
}

.icon {
  font-size: 16px;
}

.text {
  font-weight: 500;
}
</style>
