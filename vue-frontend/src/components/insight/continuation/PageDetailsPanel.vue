<template>
  <div class="page-details-panel">
    <div class="panel-header">
      <h3>📄 页面详情管理</h3>
      <button 
        v-if="pages.length > 0"
        class="btn secondary small"
        :disabled="isGenerating"
        @click="$emit('regenerate-all-prompts')"
      >
        {{ isGenerating ? '生成中...' : '🎨 重新生成提示词' }}
      </button>
    </div>
    
    <div v-if="pages.length === 0" class="empty-state">
      <p>尚未生成页面详情</p>
      <button 
        class="btn primary"
        :disabled="isGenerating"
        @click="$emit('generate-details')"
      >
        {{ isGenerating ? '生成中...' : '🎯 生成页面详情' }}
      </button>
    </div>
    
    <div v-else class="pages-list">
      <div v-for="page in pages" :key="page.page_number" class="page-card">
        <div class="page-header">
          <h4>页面 {{ page.page_number }}</h4>
          <span class="page-status" :class="page.status">{{ getStatusText(page.status) }}</span>
        </div>
        
        <div class="page-fields">
          <div class="page-field">
            <label>角色（逗号分隔）：</label>
            <input 
              :value="page.characters.join(', ')"
              @input="updateCharacters(page, $event)"
              type="text"
              class="field-input"
            >
          </div>
          
          <div class="page-field">
            <label>场景描述：</label>
            <textarea 
              v-model="page.description"
              rows="2"
              class="field-input"
              @input="onDataChange"
            ></textarea>
          </div>
          
          <div class="page-field">
            <label>对话：</label>
            <div class="dialogues">
              <div v-for="(dialogue, idx) in page.dialogues" :key="idx" class="dialogue-item">
                <strong>{{ dialogue.character }}:</strong> {{ dialogue.text }}
              </div>
            </div>
          </div>
          
          <div class="page-field prompt-field">
            <label>图片提示词：</label>
            <textarea 
              v-model="page.image_prompt"
              rows="3"
              class="field-input"
              @input="onDataChange"
            ></textarea>
            <button 
              class="btn secondary small"
              :disabled="regeneratingPage === page.page_number"
              @click="$emit('regenerate-prompt', page.page_number)"
            >
              {{ regeneratingPage === page.page_number ? '生成中...' : '🎨 重新生成提示词' }}
            </button>
          </div>
        </div>
      </div>
      
      <div class="page-actions">
        <button class="btn secondary" @click="$emit('save-changes')">💾 保存修改</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { PageContent } from '@/api/continuation'

const props = defineProps<{
  pages: PageContent[]
  isGenerating: boolean
  regeneratingPage: number | null
}>()

const emit = defineEmits<{
  'generate-details': []
  'regenerate-prompt': [pageNumber: number]
  'regenerate-all-prompts': []
  'save-changes': []
  'data-change': []
}>()

function updateCharacters(page: PageContent, event: Event) {
  const input = event.target as HTMLInputElement
  const value = input.value
  page.characters = value.split(',').map(s => s.trim()).filter(s => s)
  onDataChange()
}

function onDataChange() {
  emit('data-change')
}

function getStatusText(status: string): string {
  const map: Record<string, string> = {
    'pending': '待处理',
    'generating': '生成中',
    'generated': '已生成',
    'failed': '失败'
  }
  return map[status] || status
}
</script>

<style scoped>
.page-details-panel {
  padding: 24px;
}

.page-details-panel h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}


.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: var(--text-secondary, #666);
}

.empty-state p {
  margin: 0 0 20px 0;
  font-size: 16px;
}

.pages-list {
  display: grid;
  gap: 16px;
}

.page-card {
  padding: 16px;
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 12px;
  border: 1px solid var(--border-color, #e0e0e0);
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.page-header h4 {
  margin: 0;
  font-size: 16px;
}

.page-status {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.page-status.pending {
  background: #fef3c7;
  color: #92400e;
}

.page-status.generating {
  background: #dbeafe;
  color: #1e40af;
}

.page-status.generated {
  background: #d1fae5;
  color: #065f46;
}

.page-status.failed {
  background: #fee2e2;
  color: #991b1b;
}

.page-fields {
  display: grid;
  gap: 12px;
}

.page-field label {
  display: block;
  font-size: 13px;
  font-weight: 500;
  margin-bottom: 6px;
}

.field-input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  font-size: 13px;
  font-family: inherit;
}

.field-input:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.dialogues {
  background: var(--bg-primary, #fff);
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 13px;
}

.dialogue-item {
  margin-bottom: 4px;
}

.dialogue-item:last-child {
  margin-bottom: 0;
}

.dialogue-item strong {
  color: var(--primary, #6366f1);
}

.prompt-field {
  position: relative;
}

.prompt-field .btn {
  margin-top: 8px;
}

.page-actions {
  margin-top: 16px;
  text-align: center;
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
</style>
