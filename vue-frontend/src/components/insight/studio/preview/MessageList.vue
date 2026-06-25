<script setup lang="ts">
import { ref, watch } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import {
  attachmentTypeLabel,
  canEditStudioChatMessage as canEditMessage,
  canRegenerateStudioChatMessage as canRegenerateMessage,
} from '../characterStudioPreviewHelpers'
import type { CharacterStudioChatAttachment, CharacterStudioChatSession } from '@/types/characterStudio'

const props = defineProps<{
  assistantName: string
  attachmentUrlFor: (attachment: CharacterStudioChatAttachment) => string
  chatMutating: boolean
  chatStreaming: boolean
  session: CharacterStudioChatSession
}>()

const emit = defineEmits<{
  (event: 'delete-message', messageId: string): void
  (event: 'edit-message', value: { messageId: string; content: string }): void
  (event: 'open-image-preview', attachment: CharacterStudioChatAttachment): void
  (event: 'regenerate-message', messageId: string): void
}>()

const editingMessageId = ref('')
const editingContent = ref('')

watch(() => props.session.session_id, () => {
  cancelEdit()
})

function startEdit(message: CharacterStudioChatSession['messages'][number]) {
  editingMessageId.value = message.message_id
  editingContent.value = message.content
}

function cancelEdit() {
  editingMessageId.value = ''
  editingContent.value = ''
}

function commitEdit(message: CharacterStudioChatSession['messages'][number]) {
  if (!editingContent.value.trim()) return
  emit('edit-message', { messageId: message.message_id, content: editingContent.value.trim() })
  cancelEdit()
}
</script>

<template>
  <div class="messages-panel">
    <div v-if="session.messages.length === 0" class="empty-copy">当前会话还没有消息。</div>
    <article v-for="item in session.messages" :key="item.message_id" class="message-card" :class="item.role">
      <div class="message-head">
        <span class="message-role">{{ item.role === 'assistant' ? assistantName : '你' }}</span>
        <div class="message-actions">
          <UiButton
            v-if="canEditMessage(item)"
            variant="toolbar"
            class="action-ghost tiny"
            :disabled="chatStreaming || chatMutating"
            @click="startEdit(item)"
          >
            编辑
          </UiButton>
          <UiButton
            variant="toolbar"
            class="action-ghost tiny"
            :disabled="chatStreaming || chatMutating"
            @click="$emit('delete-message', item.message_id)"
          >
            从这里回退
          </UiButton>
          <UiButton
            v-if="canRegenerateMessage(item)"
            variant="toolbar"
            class="action-ghost tiny"
            :disabled="chatStreaming"
            @click="$emit('regenerate-message', item.message_id)"
          >
            重新生成
          </UiButton>
        </div>
      </div>

      <div v-if="editingMessageId === item.message_id" class="editor-row">
        <UiTextarea v-model="editingContent" rows="4" />
        <div class="editor-actions">
          <UiButton variant="toolbar" class="action-primary tiny" :disabled="!editingContent.trim() || chatMutating" @click="commitEdit(item)">保存并重新生成</UiButton>
          <UiButton variant="toolbar" class="action-ghost tiny" @click="cancelEdit">取消</UiButton>
        </div>
      </div>
      <div v-else class="message-body">{{ item.content }}</div>

      <div v-if="item.attachments.length > 0" class="attachment-grid">
        <UiButton
          v-for="attachment in item.attachments"
          :key="attachment.attachment_id"
          variant="toolbar"
          type="button"
          class="attachment-card"
          @click="$emit('open-image-preview', attachment)"
        >
          <div class="attachment-frame">
            <img
              v-if="attachment.mime_type.startsWith('image/')"
              :src="attachmentUrlFor(attachment)"
              :alt="attachment.filename"
            >
          </div>
          <div class="attachment-info">
            <strong>{{ attachment.filename }}</strong>
            <span>{{ attachmentTypeLabel(attachment.mime_type) }}</span>
          </div>
        </UiButton>
      </div>
    </article>
  </div>
</template>

<style scoped>
.message-head,
.editor-actions {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  justify-content: space-between;
}

.editor-row {
  --ui-textarea-border: 1px solid var(--studio-border-strong);
  --ui-textarea-background: var(--studio-surface-soft);
  --ui-textarea-radius: 14px;
  --ui-textarea-padding: 10px 12px;
  --ui-textarea-color: var(--studio-text-strong);
  --ui-textarea-font-size: 13px;
}

.messages-panel {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
  padding: 12px;
  overflow: auto;
  border: 1px solid var(--studio-border-default);
  border-radius: 20px;
  background: linear-gradient(180deg, var(--character-studio-preview-message-list-background-start), var(--character-studio-preview-message-list-background-end));
}

.message-card {
  width: min(100%, 88%);
  padding: 14px;
  border: 1px solid var(--studio-border-default);
  border-radius: 18px;
  background: var(--character-studio-preview-assistant-message-background);
}

.message-card.user {
  margin-left: auto;
  background: var(--character-studio-preview-user-message-background);
}

.message-card.assistant {
  margin-right: auto;
  background: var(--studio-surface-tint);
}

.message-role {
  color: var(--character-studio-preview-message-role-text);
  font-size: 11px;
}

.message-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
}

.message-body {
  margin-top: 8px;
  color: var(--studio-text-strong);
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
}

.attachment-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
  gap: 10px;
  margin-top: 12px;
}

.attachment-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
  border: 1px solid var(--studio-border-default);
  border-radius: 14px;
  background: var(--character-studio-preview-attachment-card-background);
  text-align: left;
}

.attachment-frame {
  aspect-ratio: 4 / 3;
  overflow: hidden;
  border-radius: 10px;
  background: var(--studio-surface-soft);
}

.attachment-card img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.attachment-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.attachment-info strong {
  overflow: hidden;
  color: var(--studio-text-strong);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.attachment-info span {
  color: var(--studio-text-muted);
  font-size: 11px;
}

.action-ghost,
.action-primary {
  border: none;
  border-radius: 14px;
  font-size: 13px;
  cursor: pointer;
}

.action-ghost {
  padding: 10px 14px;
  background: var(--studio-surface-muted);
  color: var(--studio-text-default);
}

.action-primary {
  padding: 11px 16px;
  background: linear-gradient(135deg, var(--character-studio-preview-primary-action-background-start), var(--character-studio-preview-primary-action-background-end));
  box-shadow: 0 12px 24px var(--character-studio-preview-primary-action-shadow);
  color: var(--color-text-inverse);
}

.action-ghost:disabled,
.action-primary:disabled {
  cursor: not-allowed;
  box-shadow: none;
  opacity: 0.68;
}

.tiny {
  padding: 6px 10px;
  font-size: 12px;
}

.empty-copy {
  color: var(--studio-text-subtle);
  font-size: 13px;
  line-height: 1.7;
}

@media (--breakpoint-studio-down) {
  .message-card {
    width: 100%;
  }
}
</style>
