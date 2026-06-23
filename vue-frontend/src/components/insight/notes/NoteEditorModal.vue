<script setup lang="ts">
import { computed } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import type { NoteData, NoteType } from '@/stores/insightStore'

const props = defineProps<{
  editingNote: NoteData | null
  noteContent: string
  notePageNum: number | null
  noteTags: string
  noteTitle: string
  noteType: NoteType
  noteTypeOptions: Array<{ label: string; value: NoteType }>
  visible: boolean
}>()

const emit = defineEmits<{
  (event: 'close'): void
  (event: 'save'): void
  (event: 'showPage', pageNum: number): void
  (event: 'update:noteContent', value: string): void
  (event: 'update:notePageNum', value: number | null): void
  (event: 'update:noteTags', value: string): void
  (event: 'update:noteTitle', value: string): void
  (event: 'update:noteType', value: NoteType): void
}>()

const noteTitleModel = computed({
  get: () => props.noteTitle,
  set: value => emit('update:noteTitle', value),
})

const noteContentModel = computed({
  get: () => props.noteContent,
  set: value => emit('update:noteContent', value),
})

const noteTypeModel = computed({
  get: () => props.noteType,
  set: value => emit('update:noteType', value),
})

const notePageNumModel = computed({
  get: () => props.notePageNum,
  set: value => emit('update:notePageNum', value),
})

const noteTagsModel = computed({
  get: () => props.noteTags,
  set: value => emit('update:noteTags', value),
})

const saveDisabled = computed(() => props.editingNote?.type !== 'qa' && !props.noteContent.trim())
</script>

<template>
  <BaseModal
    :model-value="visible"
    :title="editingNote ? '编辑笔记' : '添加笔记'"
    size="small"
    custom-class="notes-panel-modal"
    width="90%"
    max-width="450px"
    border-radius="16px"
    @close="$emit('close')"
  >
    <template #title>
      <span>{{ editingNote ? '编辑笔记' : '添加笔记' }}</span>
    </template>

    <div class="notes-modal-body">
      <template v-if="editingNote && editingNote.type === 'qa'">
        <div class="qa-note-view">
          <div class="qa-section">
            <label class="qa-label">问题</label>
            <div class="qa-content">{{ editingNote.question }}</div>
          </div>
          <div class="qa-section">
            <label class="qa-label">回答</label>
            <div class="qa-content qa-answer">{{ editingNote.answer }}</div>
          </div>
          <div v-if="editingNote.citations && editingNote.citations.length > 0" class="qa-section">
            <label class="qa-label">引用页码</label>
            <div class="qa-citations">
              <UiButton
                v-for="citation in editingNote.citations"
                :key="citation.page"
                variant="toolbar"
                class="qa-citation-badge"
                :aria-label="`查看第 ${citation.page} 页`"
                @click="$emit('showPage', citation.page)"
              >
                第{{ citation.page }}页
              </UiButton>
            </div>
          </div>
          <div v-if="editingNote.comment" class="qa-section">
            <label class="qa-label">补充说明</label>
            <div class="qa-content">{{ editingNote.comment }}</div>
          </div>
        </div>
        <div class="notes-panel__field">
          <label>笔记标题 <span class="label-optional">(可选)</span></label>
          <UiInput
            v-model="noteTitleModel"
            type="text"
            class="notes-panel__form-input"
            placeholder="修改标题..."
          />
        </div>
      </template>

      <template v-else>
        <div class="notes-panel__field">
          <label>笔记类型</label>
          <CustomSelect
            v-model="noteTypeModel"
            :options="noteTypeOptions"
          />
        </div>
        <div class="notes-panel__field">
          <label>标题 <span class="label-optional">(可选)</span></label>
          <UiInput
            v-model="noteTitleModel"
            type="text"
            class="notes-panel__form-input"
            placeholder="给笔记起个标题..."
          />
        </div>
        <div class="notes-panel__field">
          <label>内容 <span class="label-required">*</span></label>
          <UiTextarea
            v-model="noteContentModel"
            class="notes-panel__form-textarea"
            rows="5"
            placeholder="写下你的想法..."
          />
        </div>
        <div class="notes-panel__field">
          <label>关联页码 <span class="label-optional">(可选)</span></label>
          <UiInput
            v-model.number="notePageNumModel"
            type="number"
            class="notes-panel__form-input"
            placeholder="输入页码"
            min="1"
          />
        </div>
        <div class="notes-panel__field">
          <label>标签 <span class="label-optional">(可选)</span></label>
          <UiInput
            v-model="noteTagsModel"
            type="text"
            class="notes-panel__form-input"
            placeholder="多个标签用逗号分隔，如: 角色,剧情"
          />
        </div>
      </template>
    </div>

    <template #footer>
      <UiButton variant="secondary" @click="$emit('close')">取消</UiButton>
      <UiButton
        variant="primary"
        :disabled="saveDisabled"
        @click="$emit('save')"
      >
        保存
      </UiButton>
    </template>
  </BaseModal>
</template>

<style scoped>
.notes-modal-body {
  --notes-panel-surface-base: rgba(239, 68, 68, .1);

  color: var(--insight-text-primary);
}

.notes-panel__field {
  margin-bottom: 16px;
}

.notes-panel__field label {
  display: block;
  margin-bottom: 6px;
  color: var(--insight-text-primary);
  font-weight: 500;
  font-size: 14px;
}

.notes-panel__form-input,
.notes-panel__form-textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--color-border-muted);
  border-radius: 6px;
  background: var(--insight-surface-page);
  color: var(--insight-text-primary);
  font-size: 14px;
  line-height: normal;
  transition: border-color 0.2s;
}

.notes-panel__form-input:focus,
.notes-panel__form-textarea:focus {
  border-color: var(--insight-action-primary);
  outline: none;
}

.label-optional {
  color: var(--insight-text-secondary);
  font-weight: normal;
  font-size: 12px;
}

.label-required {
  color: var(--color-status-error);
  font-weight: normal;
}

.qa-note-view {
  margin-bottom: 16px;
  padding: 16px;
  border-radius: 12px;
  background: var(--insight-surface-tertiary);
}

.qa-section {
  margin-bottom: 16px;
}

.qa-section:last-child {
  margin-bottom: 0;
}

.qa-label {
  display: block;
  margin-bottom: 8px;
  color: var(--insight-text-secondary);
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.qa-content {
  padding: 12px;
  border-radius: 8px;
  background: var(--insight-surface-secondary);
  color: var(--insight-text-primary);
  font-size: 14px;
  line-height: 1.6;
}

.qa-answer {
  max-height: 200px;
  overflow-y: auto;
}

.qa-citations {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.qa-citation-badge {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border: 0;
  border-radius: 12px;
  background: var(--insight-action-primary);
  color: white;
  font: inherit;
  font-weight: 500;
  font-size: 12px;
  cursor: pointer;
  transition: opacity 0.2s;
}

.qa-citation-badge:hover {
  opacity: 0.8;
}
</style>
