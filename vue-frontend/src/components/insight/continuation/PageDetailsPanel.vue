<template>
  <div class="page-details-panel">
    <div class="panel-header">
      <h3>📄 页面剧情管理</h3>
    </div>

    <div v-if="pages.length === 0" class="empty-state">
      <p>尚未生成页面剧情</p>
      <button
        class="btn primary"
        :disabled="isGenerating"
        @click="$emit('generate-details')"
      >
        {{ isGenerating ? '生成中...' : '🎯 生成页面剧情' }}
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
            <label>上一页剧情承接：</label>
            <textarea
              v-model="page.continuity_text"
              rows="3"
              class="field-input"
              @input="$emit('story-change', page.page_number)"
            ></textarea>
          </div>

          <div class="page-field">
            <label>本页剧情：</label>
            <textarea
              v-model="page.story_text"
              rows="4"
              class="field-input"
              @input="$emit('story-change', page.page_number)"
            ></textarea>
          </div>

          <div class="page-field">
            <label>关键对白：</label>
            <textarea
              v-model="page.dialogue_text"
              rows="3"
              class="field-input"
              @input="$emit('story-change', page.page_number)"
            ></textarea>
          </div>

          <div class="page-field">
            <label>角色（逗号分隔）：</label>
            <input
              :value="page.characters.join(', ')"
              @input="updateCharacters(page, $event)"
              type="text"
              class="field-input"
            >
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

defineProps<{
  pages: PageContent[]
  isGenerating: boolean
}>()

const emit = defineEmits<{
  'generate-details': []
  'save-changes': []
  'story-change': [pageNumber: number]
}>()

function updateCharacters(page: PageContent, event: Event) {
  const input = event.target as HTMLInputElement
  const value = input.value
  page.characters = value.split(',').map(s => s.trim()).filter(s => s)
  emit('story-change', page.page_number)
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
  margin: 0 0 20px;
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

.btn.secondary {
  background: var(--bg-primary, #fff);
  color: var(--text-primary, #333);
  border: 1px solid var(--border-color, #ddd);
}

.btn.secondary:hover:not(:disabled) {
  border-color: var(--primary, #6366f1);
  color: var(--primary, #6366f1);
}
</style>
