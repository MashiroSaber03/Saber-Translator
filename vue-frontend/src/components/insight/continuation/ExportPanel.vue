<template>
  <div class="export-panel">
    <h3>📦 导出成品</h3>
    
    <div class="export-options">
      <div class="export-summary">
        <p>共生成 <strong>{{ generatedCount }}</strong> 页图片，可导出为以下格式：</p>
      </div>
      
      <div class="export-formats">
        <div 
          class="format-card" 
          @click="selectedFormat = 'images'" 
          :class="{ selected: selectedFormat === 'images' }"
        >
          <span class="format-icon">🖼️</span>
          <span class="format-name">图片 ZIP</span>
          <span class="format-desc">所有页面打包下载</span>
        </div>
        <div 
          class="format-card" 
          @click="selectedFormat = 'pdf'" 
          :class="{ selected: selectedFormat === 'pdf' }"
        >
          <span class="format-icon">📄</span>
          <span class="format-name">PDF 文档</span>
          <span class="format-desc">方便阅读和分享</span>
        </div>
      </div>
      
      <button 
        class="btn primary large" 
        :disabled="isExporting"
        @click="handleExport"
      >
        {{ isExporting ? '导出中...' : '📥 下载' }}
      </button>
      
      <div class="export-actions">
        <button class="btn secondary" @click="clearAndRestart">🗑️ 清空并重新开始</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useContinuationStateInject } from '@/composables/continuation/useContinuationState'
import * as continuationApi from '@/api/continuation'

const props = defineProps<{
  bookId: string
  generatedCount: number
}>()

const emit = defineEmits<{
  'clear-and-restart': []
}>()

const state = useContinuationStateInject()
const selectedFormat = ref<'images' | 'pdf'>('images')
const isExporting = ref(false)

async function handleExport() {
  if (!props.bookId || state.pages.value.length === 0) {
    state.showMessage('没有可导出的页面', 'error')
    return
  }
  
  isExporting.value = true
  
  try {
    let blob: Blob
    let filename: string
    
    if (selectedFormat.value === 'images') {
      blob = await continuationApi.exportAsImages(props.bookId)
      filename = `continuation_${Date.now()}.zip`
    } else {
      blob = await continuationApi.exportAsPdf(props.bookId)
      filename = `continuation_${Date.now()}.pdf`
    }
    
    // 创建下载链接
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
    
    state.showMessage('导出成功', 'success')
  } catch (error) {
    state.showMessage('导出失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isExporting.value = false
  }
}

async function clearAndRestart() {
  if (!confirm('确定要清空所有续写数据并重新开始吗？此操作不可恢复。')) {
    return
  }
  
  emit('clear-and-restart')
}
</script>

<style scoped>
.export-panel {
  padding: 24px;
}

.export-panel h3 {
  margin: 0 0 20px 0;
  font-size: 18px;
  font-weight: 600;
}

.export-options {
  max-width: 600px;
  margin: 0 auto;
}

.export-summary {
  margin-bottom: 24px;
  text-align: center;
}

.export-summary p {
  margin: 0;
  font-size: 16px;
  color: var(--text-secondary, #666);
}

.export-summary strong {
  color: var(--primary, #6366f1);
  font-size: 20px;
}

.export-formats {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.format-card {
  padding: 24px;
  border: 2px solid var(--border-color, #e0e0e0);
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s;
  text-align: center;
}

.format-card:hover {
  border-color: var(--primary, #6366f1);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
}

.format-card.selected {
  border-color: var(--primary, #6366f1);
  background: rgba(99, 102, 241, 0.05);
}

.format-icon {
  font-size: 48px;
  display: block;
  margin-bottom: 12px;
}

.format-name {
  display: block;
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 8px;
}

.format-desc {
  display: block;
  font-size: 14px;
  color: var(--text-secondary, #666);
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn.primary {
  background: var(--primary, #6366f1);
  color: white;
  width: 100%;
  margin-bottom: 16px;
}

.btn.primary:hover:not(:disabled) {
  background: var(--primary-dark, #4f46e5);
}

.btn.primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn.primary.large {
  padding: 14px 28px;
  font-size: 16px;
}

.btn.secondary {
  background: var(--bg-secondary, #f3f4f6);
  color: var(--text-primary, #333);
  border: 1px solid var(--border-color, #e0e0e0);
}

.btn.secondary:hover {
  background: var(--bg-hover, #e5e7eb);
}

.export-actions {
  text-align: center;
}
</style>
