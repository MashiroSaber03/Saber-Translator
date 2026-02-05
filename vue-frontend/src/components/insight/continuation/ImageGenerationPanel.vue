<template>
  <div class="image-generation-panel">
    <h3>🎨 图片生成</h3>
    
    <div class="generation-controls">
      <button 
        class="btn primary large"
        :disabled="isGenerating || pages.length === 0"
        @click="$emit('batch-generate')"
      >
        {{ isGenerating ? '生成中...' : '🚀 批量生成图片' }}
      </button>
      
      <div v-if="isGenerating" class="progress-bar">
        <div class="progress-fill" :style="{ width: progress + '%' }"></div>
        <span class="progress-text">{{ progress }}%</span>
      </div>
    </div>
    
    <div class="generated-images">
      <div v-for="page in pages" :key="page.page_number" class="image-card">
        <div class="image-header">
          <h4>页面 {{ page.page_number }}</h4>
          <span class="image-status" :class="page.status">{{ getStatusText(page.status) }}</span>
        </div>
        
        <div class="image-preview">
          <img 
            v-if="page.image_url" 
            :src="getImageUrl(page.image_url)"
            :alt="`页面 ${page.page_number}`"
          >
          <div v-else class="no-image">
            <span>{{ page.status === 'generating' ? '⏳' : '📷' }}</span>
            <p>{{ page.status === 'generating' ? '生成中...' : '未生成' }}</p>
          </div>
        </div>
        
        <!-- 提示词显示和编辑区域 -->
        <div class="prompt-section">
          <div class="prompt-header">
            <label>📝 生图提示词</label>
            <button 
              class="btn-mini" 
              @click="togglePromptEdit(page.page_number)"
            >
              {{ editingPromptPage === page.page_number ? '收起' : '编辑' }}
            </button>
          </div>
          <div v-if="editingPromptPage === page.page_number" class="prompt-edit">
            <textarea 
              v-model="page.image_prompt"
              rows="4"
              class="prompt-input"
              placeholder="输入生图提示词..."
              @input="$emit('prompt-change', page.page_number, page.image_prompt)"
            ></textarea>
          </div>
          <div v-else class="prompt-preview">
            <p v-if="page.image_prompt" class="prompt-text">{{ page.image_prompt }}</p>
            <p v-else class="prompt-empty">暂无提示词</p>
          </div>
        </div>
        
        <div class="image-actions">
          <button 
            class="btn secondary small"
            :disabled="page.status === 'generating'"
            @click="$emit('regenerate', page.page_number)"
          >
            ↺ 重新生成
          </button>
          <button 
            v-if="page.previous_url"
            class="btn secondary small"
            @click="$emit('use-previous', page.page_number)"
          >
            ◀ 上一版本
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { PageContent } from '@/api/continuation'
import { useContinuationStateInject } from '@/composables/continuation/useContinuationState'

const props = defineProps<{
  pages: PageContent[]
  isGenerating: boolean
  progress: number
}>()

const emit = defineEmits<{
  'batch-generate': []
  'regenerate': [pageNumber: number]
  'use-previous': [pageNumber: number]
  'prompt-change': [pageNumber: number, prompt: string]
}>()

const state = useContinuationStateInject()

// 当前正在编辑提示词的页面
const editingPromptPage = ref<number | null>(null)

function togglePromptEdit(pageNumber: number) {
  if (editingPromptPage.value === pageNumber) {
    editingPromptPage.value = null
  } else {
    editingPromptPage.value = pageNumber
  }
}

function getImageUrl(imagePath: string): string {
  return state.getGeneratedImageUrl(imagePath)
}

function getStatusText(status: string): string {
  const map: Record<string, string> = {
    'pending': '待生成',
    'generating': '生成中',
    'generated': '已生成',
    'failed': '失败'
  }
  return map[status] || status
}
</script>

<style scoped>
.image-generation-panel {
  padding: 24px;
}

.image-generation-panel h3 {
  margin: 0 0 20px 0;
  font-size: 18px;
  font-weight: 600;
}

.generation-controls {
  margin-bottom: 24px;
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

.btn.secondary:hover:not(:disabled) {
  background: var(--bg-hover, #e5e7eb);
}

.btn.secondary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn.small {
  padding: 6px 12px;
  font-size: 13px;
}

.progress-bar {
  position: relative;
  height: 32px;
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 16px;
  overflow: hidden;
  margin-top: 16px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--primary, #6366f1), #8b5cf6);
  transition: width 0.3s ease;
}

.progress-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-weight: 600;
  font-size: 14px;
  color: var(--text-primary, #333);
}

.generated-images {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
  gap: 20px;
}

.image-card {
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--border-color, #e0e0e0);
}

.image-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: var(--bg-primary, #fff);
  border-bottom: 1px solid var(--border-color, #e0e0e0);
}

.image-header h4 {
  margin: 0;
  font-size: 15px;
}

.image-status {
  padding: 4px 10px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 500;
}

.image-status.pending {
  background: #fef3c7;
  color: #92400e;
}

.image-status.generating {
  background: #dbeafe;
  color: #1e40af;
}

.image-status.generated {
  background: #d1fae5;
  color: #065f46;
}

.image-status.failed {
  background: #fee2e2;
  color: #991b1b;
}

.image-preview {
  aspect-ratio: 2 / 3;
  background: #e0e0e0;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.image-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.no-image {
  display: flex;
  flex-direction: column;
  align-items: center;
  color: #999;
}

.no-image span {
  font-size: 48px;
  margin-bottom: 8px;
}

.no-image p {
  margin: 0;
  font-size: 14px;
}

.image-actions {
  display: flex;
  gap: 8px;
  padding: 12px;
  background: var(--bg-primary, #fff);
}

.image-actions .btn {
  flex: 1;
}

/* 提示词区域样式 */
.prompt-section {
  padding: 12px;
  background: linear-gradient(to bottom, #fafbff, #f5f7ff);
  border-top: 1px solid #e8eaf6;
}

.prompt-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.prompt-header label {
  font-size: 12px;
  font-weight: 600;
  color: #4b5563;
}

.btn-mini {
  padding: 4px 10px;
  font-size: 11px;
  font-weight: 500;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  cursor: pointer;
  color: #6366f1;
  transition: all 0.2s;
}

.btn-mini:hover {
  background: #eef2ff;
  border-color: #c7d2fe;
}

.prompt-preview {
  max-height: 80px;
  overflow-y: auto;
}

.prompt-text {
  margin: 0;
  font-size: 11px;
  line-height: 1.5;
  color: #4b5563;
  white-space: pre-wrap;
  word-break: break-word;
}

.prompt-empty {
  margin: 0;
  font-size: 11px;
  color: #9ca3af;
  font-style: italic;
}

.prompt-edit {
  margin-top: 4px;
}

.prompt-input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 12px;
  font-family: inherit;
  line-height: 1.5;
  resize: vertical;
  min-height: 80px;
}

.prompt-input:focus {
  outline: none;
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}
</style>
