<template>
  <div class="page-details-panel">
    <div class="panel-header">
      <h3>📄 页面剧情管理</h3>
    </div>

    <div v-if="pages.length === 0" class="empty-state">
      <p>尚未生成页面剧情</p>
      <UiButton
        variant="primary"
       
        :disabled="isGenerating"
        @click="$emit('generate-details')"
      >
        {{ isGenerating ? '生成中...' : '🎯 生成页面剧情' }}
      </UiButton>
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
            <UiTextarea
              v-model="page.continuity_text"
              rows="3"
              class="field-input"
              @input="$emit('story-change', page.page_number)"
            />
          </div>

          <div class="page-field">
            <label>本页剧情：</label>
            <UiTextarea
              v-model="page.story_text"
              rows="4"
              class="field-input"
              @input="$emit('story-change', page.page_number)"
            />
          </div>

          <div class="page-field">
            <label>关键对白：</label>
            <UiTextarea
              v-model="page.dialogue_text"
              rows="3"
              class="field-input"
              @input="$emit('story-change', page.page_number)"
            />
          </div>

          <div class="page-field">
            <label>角色（逗号分隔）：</label>
            <UiInput
              :value="page.characters.join(', ')"
              @input="updateCharacters(page, $event)"
              type="text"
              class="field-input"
            />
          </div>
        </div>
      </div>

      <div class="page-actions">
        <UiButton variant="secondary" @click="$emit('save-changes')">💾 保存修改</UiButton>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
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
  /* owner tokens: page-details-panel */
  --page-details-panel-surface-base: #fef3c7;
  --page-details-panel-surface-raised: #dbeafe;
  --page-details-panel-surface-muted: #d1fae5;
  --page-details-panel-surface-subtle: #fee2e2;
  --page-details-panel-text-primary: #92400e;
  --page-details-panel-text-secondary: #1e40af;
  --page-details-panel-text-muted: #065f46;
  --page-details-panel-text-subtle: #991b1b;
  --ui-button-padding: 10px 20px;
  --ui-button-radius: 8px;
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--color-surface-brand);
  --ui-button-primary-hover-background: var(--color-surface-brand-strong);
  --ui-button-secondary-background: var(--color-surface-base);
  --ui-button-secondary-color: var(--color-text-default, var(--color-text-default));
  --ui-button-secondary-border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  --ui-button-secondary-hover-border-color: var(--color-border-brand);
  --ui-button-secondary-hover-color: var(--color-text-brand);

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
  color: var(--color-text-supporting, var(--color-text-secondary));
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
  background: var(--color-surface-subtle);
  border-radius: 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
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
  background: var(--page-details-panel-surface-base);
  color: var(--page-details-panel-text-primary);
}

.page-status.generating {
  background: var(--page-details-panel-surface-raised);
  color: var(--page-details-panel-text-secondary);
}

.page-status.generated {
  background: var(--page-details-panel-surface-muted);
  color: var(--page-details-panel-text-muted);
}

.page-status.failed {
  background: var(--page-details-panel-surface-subtle);
  color: var(--page-details-panel-text-subtle);
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
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 6px;
  font-size: 13px;
  font-family: inherit;
}

.field-input:focus {
  outline: none;
  border-color: var(--color-border-brand);
}

.page-actions {
  margin-top: 16px;
  text-align: center;
}

</style>
