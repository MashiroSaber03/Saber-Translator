<script setup lang="ts">
import { computed } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

interface PendingQAData {
  messageId: string
  question: string
  answer: string
  citations: Array<{ page: number }>
}

const props = defineProps<{
  noteComment: string
  noteTitle: string
  pendingQAData: PendingQAData | null
  renderMarkdown: (content: string) => string
  visible: boolean
}>()

const emit = defineEmits<{
  (event: 'close'): void
  (event: 'save'): void
  (event: 'update:noteComment', value: string): void
  (event: 'update:noteTitle', value: string): void
}>()

const noteTitleModel = computed({
  get: () => props.noteTitle,
  set: value => emit('update:noteTitle', value),
})

const noteCommentModel = computed({
  get: () => props.noteComment,
  set: value => emit('update:noteComment', value),
})

</script>

<template>
  <BaseModal
    :model-value="visible"
    title="📝 添加笔记"
    size="medium"
    custom-class="qa-note-modal"
    body-padding="spacious"
    width="90%"
    max-width="560px"
    border-radius="16px"
    footer-background="var(--insight-surface-secondary)"
    @close="$emit('close')"
  >
    <div class="qa-note-modal-body">
      <div v-if="pendingQAData" class="qa-preview">
        <div class="qa-preview-section">
          <label>问题</label>
          <div class="qa-preview-content">{{ pendingQAData.question }}</div>
        </div>
        <div class="qa-preview-section">
          <label>回答</label>
          <div class="qa-preview-content" v-html="renderMarkdown(pendingQAData.answer)"></div>
        </div>
        <div v-if="pendingQAData.citations.length > 0" class="qa-preview-section">
          <label>引用页码</label>
          <div class="qa-preview-citations">
            <span
              v-for="citation in pendingQAData.citations"
              :key="citation.page"
              class="qa-citation-badge"
            >
              第{{ citation.page }}页
            </span>
          </div>
        </div>
      </div>

      <div class="note-form">
        <div class="qa-note-modal__field">
          <label for="qaNoteTitle">笔记标题 <span class="optional">(可选)</span></label>
          <UiInput
            id="qaNoteTitle"
            v-model="noteTitleModel"
            type="text"
            class="qa-note-modal__form-input"
            placeholder="默认使用问题作为标题..."
          />
        </div>
        <div class="qa-note-modal__field">
          <label for="qaNoteComment">补充说明 <span class="optional">(可选)</span></label>
          <UiTextarea
            id="qaNoteComment"
            v-model="noteCommentModel"
            class="qa-note-modal__form-textarea"
            rows="3"
            placeholder="添加你的评论或补充..."
          />
        </div>
      </div>
    </div>

    <template #footer>
      <UiButton variant="secondary" @click="$emit('close')">取消</UiButton>
      <UiButton variant="primary" @click="$emit('save')">保存笔记</UiButton>
    </template>
  </BaseModal>
</template>

<style scoped>
.qa-preview {
  margin-bottom: 16px;
  padding: 16px;
  border-radius: 12px;
  background: var(--insight-surface-tertiary);
}

.qa-preview-section {
  margin-bottom: 16px;
}

.qa-preview-section:last-child {
  margin-bottom: 0;
}

.qa-preview-section label {
  display: block;
  margin-bottom: 8px;
  color: var(--insight-text-secondary);
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 0;
  text-transform: uppercase;
}

.qa-preview-content {
  max-height: 150px;
  padding: 12px;
  overflow-y: auto;
  border-radius: 8px;
  background: var(--insight-surface-secondary);
  color: var(--insight-text-primary);
  font-size: 14px;
  line-height: 1.6;
}

.qa-preview-citations {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.qa-citation-badge {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 12px;
  background: var(--insight-action-primary);
  color: var(--color-text-inverse);
  font-weight: 500;
  font-size: 12px;
}

.note-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.qa-note-modal__field {
  margin-bottom: 0;
}

.note-form label {
  display: block;
  margin-bottom: 6px;
  color: var(--insight-text-primary);
  font-weight: 500;
  font-size: 13px;
}

.optional {
  color: var(--insight-text-secondary);
  font-weight: 400;
  font-size: 12px;
}

.qa-note-modal__form-input,
.qa-note-modal__form-textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  background: var(--insight-surface-secondary);
  color: var(--insight-text-primary);
  font-size: 14px;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.qa-note-modal__form-input:focus,
.qa-note-modal__form-textarea:focus {
  border-color: var(--insight-action-primary);
  outline: none;
  box-shadow: 0 0 0 3px var(--color-focus-brand-soft);
}

.qa-note-modal__form-textarea {
  min-height: 80px;
  resize: vertical;
  font-family: inherit;
  line-height: 1.5;
}
</style>
