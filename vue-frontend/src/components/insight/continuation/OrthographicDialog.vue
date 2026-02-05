<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-dialog ortho-dialog">
      <div class="modal-header">
        <h3>🎨 生成三视图 - {{ characterName }} <span v-if="formName && formName !== '默认'">({{ formName }})</span></h3>
        <button class="close-btn" @click="$emit('close')">×</button>
      </div>
      
      <div class="modal-body">
        <div class="ortho-upload-section">
          <label 
            class="upload-area"
            :class="{ 'drag-over': isDragging }"
            @dragenter="handleDragEnter"
            @dragover="handleDragOver"
            @dragleave="handleDragLeave"
            @drop="handleDrop"
          >
            <input 
              type="file" 
              accept="image/*" 
              multiple 
              hidden 
              @change="selectImages"
            >
            <div class="upload-placeholder">
              <span class="upload-icon">{{ isDragging ? '📥' : '📁' }}</span>
              <p v-if="isDragging">释放以上传图片</p>
              <p v-else>点击选择或拖拽角色图片（1-5张）</p>
              <p class="hint">可上传多张图片帮助AI理解角色特征</p>
            </div>
          </label>
          
          <div v-if="sourceImages.length > 0" class="source-images">
            <div v-for="(file, index) in sourceImages" :key="index" class="source-image">
              <img :src="createObjectURL(file)" :alt="`源图${index + 1}`">
              <span class="image-index">{{ index + 1 }}</span>
            </div>
          </div>
        </div>
        
        <div v-if="isGenerating" class="generating-state">
          <div class="spinner"></div>
          <p class="progress-message">{{ progressMessage }}</p>
          <p class="progress-tip">⏱️ AI 生成通常需要 30-60 秒</p>
        </div>
        
        <div v-else-if="resultImagePath" class="ortho-result">
          <h4>生成结果：</h4>
          <div class="result-preview">
            <img :src="getResultUrl()" alt="三视图">
          </div>
        </div>
      </div>
      
      <div class="modal-footer">
        <button class="btn secondary" @click="$emit('close')">取消</button>
        <button 
          v-if="!resultImagePath"
          class="btn primary"
          :disabled="sourceImages.length === 0 || isGenerating"
          @click="generate"
        >
          {{ isGenerating ? '生成中...' : '🎨 生成三视图' }}
        </button>
        <div v-else class="result-actions">
          <button class="btn secondary" @click="generate">重新生成</button>
          <button class="btn primary" @click="useResult">✓ 使用三视图</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  characterName: string
  formId: string
  formName: string
  bookId: string
}>()

const emit = defineEmits<{
  'close': []
  'generate': [sourceImages: File[]]
  'use-result': [imagePath: string]
}>()

const sourceImages = ref<File[]>([])
const isDragging = ref(false)
const isGenerating = ref(false)
const progressMessage = ref('')
const resultImagePath = ref<string | null>(null)

function selectImages(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files) return
  
  const files = Array.from(input.files).slice(0, 5)
  sourceImages.value = files
}

function handleDragEnter(event: DragEvent) {
  event.preventDefault()
  event.stopPropagation()
  isDragging.value = true
}

function handleDragOver(event: DragEvent) {
  event.preventDefault()
  event.stopPropagation()
  isDragging.value = true
}

function handleDragLeave(event: DragEvent) {
  event.preventDefault()
  event.stopPropagation()
  isDragging.value = false
}

function handleDrop(event: DragEvent) {
  event.preventDefault()
  event.stopPropagation()
  isDragging.value = false
  
  const files = event.dataTransfer?.files
  if (!files || files.length === 0) return
  
  const imageFiles = Array.from(files)
    .filter(file => file.type.startsWith('image/'))
    .slice(0, 5)
  
  if (imageFiles.length > 0) {
    sourceImages.value = imageFiles
  }
}

async function generate() {
  if (sourceImages.value.length === 0) return
  
  isGenerating.value = true
  progressMessage.value = `正在上传 ${sourceImages.value.length} 张图片...`
  
  // 模拟进度提示
  setTimeout(() => {
    if (isGenerating.value) {
      progressMessage.value = 'AI 正在分析角色特征...'
    }
  }, 500)
  
  setTimeout(() => {
    if (isGenerating.value) {
      progressMessage.value = '正在生成三视图，请耐心等待...'
    }
  }, 2000)
  
  emit('generate', sourceImages.value)
}

function useResult() {
  if (resultImagePath.value) {
    emit('use-result', resultImagePath.value)
  }
}

function createObjectURL(file: File): string {
  return window.URL.createObjectURL(file)
}

function getResultUrl(): string {
  if (!props.bookId || !resultImagePath.value) return ''
  return `/api/manga-insight/${props.bookId}/continuation/generated-image?path=${encodeURIComponent(resultImagePath.value)}`
}

// 暴露方法给父组件
function setResult(imagePath: string) {
  resultImagePath.value = imagePath
  isGenerating.value = false
}

function setGenerating(generating: boolean) {
  isGenerating.value = generating
}

defineExpose({
  setResult,
  setGenerating
})
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: fadeIn 0.2s;
}

.modal-dialog {
  background: var(--bg-primary, #fff);
  border-radius: 12px;
  width: 90%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  animation: slideUp 0.3s;
}

.ortho-dialog {
  max-width: 600px;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-bottom: 1px solid var(--border-color, #e0e0e0);
}

.modal-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
}

.close-btn {
  background: none;
  border: none;
  font-size: 28px;
  line-height: 1;
  cursor: pointer;
  color: var(--text-secondary, #666);
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  transition: background 0.2s;
}

.close-btn:hover {
  background: var(--bg-secondary, #f5f5f5);
}

.modal-body {
  padding: 24px;
}

.ortho-upload-section {
  margin-bottom: 20px;
}

.upload-area {
  display: block;
  border: 2px dashed var(--border-color, #ddd);
  border-radius: 12px;
  padding: 40px 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s;
}

.upload-area:hover,
.upload-area.drag-over {
  border-color: var(--primary, #6366f1);
  background: rgba(99, 102, 241, 0.05);
}

.upload-placeholder {
  pointer-events: none;
}

.upload-icon {
  font-size: 48px;
  display: block;
  margin-bottom: 12px;
}

.upload-placeholder p {
  margin: 8px 0;
  font-size: 14px;
}

.hint {
  color: var(--text-secondary, #666);
  font-size: 12px;
}

.source-images {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.source-image {
  position: relative;
  aspect-ratio: 1;
  border-radius: 8px;
  overflow: hidden;
  border: 2px solid var(--border-color, #e0e0e0);
}

.source-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-index {
  position: absolute;
  top: 4px;
  right: 4px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
}

.generating-state {
  text-align: center;
  padding: 40px 20px;
}

.spinner {
  width: 48px;
  height: 48px;
  border: 4px solid var(--border-color, #e0e0e0);
  border-top-color: var(--primary, #6366f1);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 16px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.progress-message {
  font-size: 16px;
  font-weight: 500;
  margin: 12px 0;
}

.progress-tip {
  font-size: 14px;
  color: var(--text-secondary, #666);
}

.ortho-result h4 {
  margin: 0 0 16px 0;
  font-size: 16px;
}

.result-preview {
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border-color, #e0e0e0);
}

.result-preview img {
  width: 100%;
  display: block;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 16px 24px;
  border-top: 1px solid var(--border-color, #e0e0e0);
}

.result-actions {
  display: flex;
  gap: 12px;
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
}

.btn.primary:hover:not(:disabled) {
  background: var(--primary-dark, #4f46e5);
}

.btn.primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn.secondary {
  background: var(--bg-secondary, #f3f4f6);
  color: var(--text-primary, #333);
  border: 1px solid var(--border-color, #e0e0e0);
}

.btn.secondary:hover {
  background: var(--bg-hover, #e5e7eb);
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideUp {
  from {
    transform: translateY(20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}
</style>
