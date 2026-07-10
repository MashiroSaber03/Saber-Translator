<script setup lang="ts">
import { ref, watch } from 'vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductMessageBubble from '@/components/product/ProductMessageBubble.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
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
  <div class="studio-message-list">
    <ProductEmptyState
      v-if="session.messages.length === 0"
      icon-name="message"
      role="note"
      size="compact"
      title="当前会话还没有消息"
    />
    <ProductMessageBubble
      v-for="item in session.messages"
      :key="item.message_id"
      class="studio-message-list__bubble"
      appearance="reading"
      :role="item.role"
      :avatar-icon-name="item.role === 'assistant' ? 'sparkles' : 'users'"
      :avatar-label="item.role === 'assistant' ? assistantName : '你'"
      :aria-label="`${item.role === 'assistant' ? assistantName : '你'}的聊天消息`"
      data-testid="studio-chat-message"
      :data-message-role="item.role"
    >
      <template #meta>
        <span class="studio-message-list__role">{{ item.role === 'assistant' ? assistantName : '你' }}</span>
      </template>

      <div v-if="editingMessageId === item.message_id" class="studio-message-list__editor">
        <UiTextarea v-model="editingContent" rows="4" variant="studio" aria-label="编辑聊天消息内容" />
        <ProductActionRow class="studio-message-list__editor-actions" aria-label="编辑聊天消息操作" justify="start" variant="toolbar">
          <UiButton variant="primary" size="xs" :disabled="!editingContent.trim() || chatMutating" @click="commitEdit(item)">
            保存并重新生成
          </UiButton>
          <UiButton variant="secondary" size="xs" @click="cancelEdit">取消</UiButton>
        </ProductActionRow>
      </div>
      <div v-else class="studio-message-list__body">{{ item.content }}</div>

      <template v-if="item.attachments.length > 0" #footer>
        <div class="studio-message-list__attachment-grid">
          <ProductRecordCard
            v-for="attachment in item.attachments"
            :key="attachment.attachment_id"
            as="button"
            class="studio-message-list__attachment-card"
            :aria-label="`预览附件：${attachment.filename}`"
            @click="$emit('open-image-preview', attachment)"
          >
            <div class="studio-message-list__attachment-frame">
              <img
                v-if="attachment.mime_type.startsWith('image/')"
                class="studio-message-list__attachment-image"
                :src="attachmentUrlFor(attachment)"
                :alt="attachment.filename"
              >
            </div>
            <div class="studio-message-list__attachment-info">
              <strong class="studio-message-list__attachment-name">{{ attachment.filename }}</strong>
              <span class="studio-message-list__attachment-type">{{ attachmentTypeLabel(attachment.mime_type) }}</span>
            </div>
          </ProductRecordCard>
        </div>
      </template>

      <template #actions>
        <ProductActionRow class="studio-message-list__actions" aria-label="聊天消息操作" justify="start" variant="toolbar">
          <UiButton
            v-if="canEditMessage(item)"
            variant="secondary"
            size="xs"
            :disabled="chatStreaming || chatMutating"
            @click="startEdit(item)"
          >
            编辑
          </UiButton>
          <UiButton
            variant="secondary"
            size="xs"
            :disabled="chatStreaming || chatMutating"
            @click="$emit('delete-message', item.message_id)"
          >
            从这里回退
          </UiButton>
          <UiButton
            v-if="canRegenerateMessage(item)"
            variant="secondary"
            size="xs"
            :disabled="chatStreaming"
            @click="$emit('regenerate-message', item.message_id)"
          >
            重新生成
          </UiButton>
        </ProductActionRow>
      </template>
    </ProductMessageBubble>
  </div>
</template>

<style scoped>
.studio-message-list__editor {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.studio-message-list {
  --product-message-bubble-reading-assistant-background: color-mix(in srgb, var(--color-action-brand) 9%, var(--color-surface-card));
  --product-message-bubble-reading-user-background: color-mix(in srgb, var(--color-text-heading) 7%, var(--color-surface-card));
  --product-message-bubble-reading-border: color-mix(in srgb, var(--color-action-brand) 16%, var(--studio-border-default));
  --product-message-bubble-reading-user-border: var(--studio-border-default);
  --product-message-bubble-reading-text: var(--studio-text-strong);
  --product-message-bubble-reading-shadow: inset 0 1px 0 color-mix(in srgb, var(--color-surface-card) 58%, transparent);

  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 14px;
  min-height: 0;
  padding: 16px;
  overflow: auto;
  scrollbar-gutter: stable;
  border: 1px solid var(--studio-border-default);
  border-radius: 20px;
  background: linear-gradient(180deg, color-mix(in srgb, var(--studio-surface-tint-muted) 74%, var(--color-surface-card)), color-mix(in srgb, var(--studio-surface-soft) 92%, var(--color-surface-card)));
}

.studio-message-list__role {
  color: inherit;
  font-size: 11px;
  opacity: 0.72;
}

.studio-message-list__body {
  color: inherit;
  font-size: 14px;
  line-height: 1.7;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

.studio-message-list__attachment-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 110px), 1fr));
  gap: 10px;
  margin-top: 0;
}

.studio-message-list__attachment-card {
  --product-record-card-accent: var(--studio-border-strong);
  --product-record-card-background: color-mix(in srgb, var(--color-surface-card) 86%, transparent);
  --product-record-card-border: var(--studio-border-default);
  --product-record-card-padding: 8px;
  --product-record-card-radius: 14px;
  --product-record-card-shadow-hover: none;
}

.studio-message-list__attachment-frame {
  display: block;
  aspect-ratio: 4 / 3;
  overflow: hidden;
  border-radius: 10px;
  background: var(--studio-surface-soft);
}

.studio-message-list__attachment-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.studio-message-list__attachment-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
  margin-top: 8px;
}

.studio-message-list__attachment-name {
  overflow: hidden;
  color: var(--studio-text-strong);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.studio-message-list__attachment-type {
  color: var(--studio-text-muted);
  font-size: 11px;
}

</style>
