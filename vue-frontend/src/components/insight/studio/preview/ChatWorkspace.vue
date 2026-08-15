<script setup lang="ts">
import { computed } from 'vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import { buildCharacterStudioGreetingOptions } from '@/utils/characterStudioGreetings'
import ChatComposer from './ChatComposer.vue'
import MessageList from './MessageList.vue'
import SessionToolbar from './SessionToolbar.vue'
import StudioPreviewWorkspacePanel from './StudioPreviewWorkspacePanel.vue'
import type {
  CharacterStudioChatAttachment,
  CharacterStudioChatSession,
  CharacterStudioChatSessionSummary,
  CharacterStudioDocument,
  CharacterStudioGreetingOption,
} from '@/types/characterStudio'

const props = defineProps<{
  archivedSessions: CharacterStudioChatSessionSummary[]
  acceptedChatSubmissionCount?: number
  availableGreetings?: CharacterStudioGreetingOption[]
  bookId: string
  chatExporting: boolean
  chatAbortable?: boolean
  chatImporting: boolean
  chatLoading: boolean
  chatMutating: boolean
  chatPromptLoading: boolean
  chatStreaming: boolean
  chatSummarizing: boolean
  document: CharacterStudioDocument | null
  session: CharacterStudioChatSession | null
}>()

const emit = defineEmits<{
  (event: 'abort-chat'): void
  (event: 'delete-session', session: CharacterStudioChatSessionSummary): void
  (event: 'delete-message', messageId: string): void
  (event: 'edit-message', value: { messageId: string; content: string }): void
  (event: 'export-session'): void
  (event: 'import-session', file: File): void
  (event: 'new-session', greetingId?: string): void
  (event: 'open-greeting-picker'): void
  (event: 'open-image-preview', attachment: CharacterStudioChatAttachment): void
  (event: 'open-prompt-preview'): void
  (event: 'regenerate-message', messageId: string): void
  (event: 'send-chat', value: { content: string; attachments: File[] }): void
  (event: 'summarize-session'): void
  (event: 'switch-session', sessionId: string): void
}>()

const currentSessionId = computed(() => props.session?.session_id || '')
const currentSessionLabel = computed(() => props.session?.title || '当前会话')
const currentSessionExcerpt = computed(() => {
  const messages = props.session?.messages || []
  const last = messages[messages.length - 1]
  return last?.content || ''
})
const currentSessionMeta = computed(() => `${props.session?.messages.length || 0} 条消息`)
const displayGreetings = computed(() =>
  props.availableGreetings?.length
    ? props.availableGreetings
    : buildCharacterStudioGreetingOptions(props.document)
)
const currentGreetingId = computed(() => {
  const source = props.session?.greeting_source || {}
  if (source.type === 'first_message') {
    const hasFirstMessage = displayGreetings.value.some(item => item.greeting_id === 'first')
    if (hasFirstMessage) return 'first'
  }
  if (source.type === 'alternate_greeting' && typeof source.index === 'number') {
    const greetingId = `alternate-${source.index}`
    const hasAlternate = displayGreetings.value.some(item => item.greeting_id === greetingId)
    if (hasAlternate) return greetingId
  }
  return displayGreetings.value[0]?.greeting_id || ''
})
const currentGreetingLabel = computed(() => {
  const selected = displayGreetings.value.find(item => item.greeting_id === currentGreetingId.value)
  return selected?.label || '选择开场白'
})
const assistantName = computed(() => props.document?.identity.name || '角色')

function attachmentUrl(attachment: CharacterStudioChatAttachment) {
  return attachment.asset_path
}

function switchSession(sessionId: string) {
  if (!sessionId || sessionId === props.session?.session_id) return
  emit('switch-session', sessionId)
}
</script>

<template>
  <StudioPreviewWorkspacePanel class="chat-workspace">
    <SessionToolbar
      :archived-sessions="archivedSessions"
      :can-use-greeting="displayGreetings.length > 0"
      :chat-exporting="chatExporting"
      :chat-importing="chatImporting"
      :chat-mutating="chatMutating"
      :chat-prompt-loading="chatPromptLoading"
      :chat-streaming="chatStreaming"
      :chat-summarizing="chatSummarizing"
      :current-greeting-label="currentGreetingLabel"
      :current-session-excerpt="currentSessionExcerpt"
      :current-session-id="currentSessionId"
      :current-session-label="currentSessionLabel"
      :current-session-meta="currentSessionMeta"
      :has-document="Boolean(document)"
      :has-session="Boolean(session)"
      @choose-session="switchSession"
      @delete-session="$emit('delete-session', $event)"
      @export-session="$emit('export-session')"
      @import-session="$emit('import-session', $event)"
      @new-session="$emit('new-session')"
      @open-greeting-picker="$emit('open-greeting-picker')"
      @open-prompt-preview="$emit('open-prompt-preview')"
      @summarize-session="$emit('summarize-session')"
    />

    <ProductEmptyState
      v-if="!document"
      icon-name="users"
      role="note"
      size="compact"
      title="选择角色文档后可开始聊天"
    />
    <template v-else-if="!session">
      <ProductEmptyState
        icon-name="message"
        role="note"
        size="compact"
        :title="chatLoading ? '聊天会话加载中...' : '当前还没有聊天会话'"
      />
    </template>
    <template v-else>
      <MessageList
        :assistant-name="assistantName"
        :attachment-url-for="attachmentUrl"
        :chat-mutating="chatMutating"
        :chat-streaming="chatStreaming"
        :session="session"
        @delete-message="$emit('delete-message', $event)"
        @edit-message="$emit('edit-message', $event)"
        @open-image-preview="$emit('open-image-preview', $event)"
        @regenerate-message="$emit('regenerate-message', $event)"
      />
      <ChatComposer
        :accepted-chat-submission-count="acceptedChatSubmissionCount"
        :chat-abortable="chatAbortable"
        :chat-streaming="chatStreaming"
        @abort-chat="$emit('abort-chat')"
        @send-chat="$emit('send-chat', $event)"
      />
    </template>
  </StudioPreviewWorkspacePanel>
</template>

<style scoped>
.chat-workspace {
  --studio-preview-workspace-panel-background: color-mix(
    in srgb,
    var(--color-surface-card) 96%,
    transparent
  );
  --studio-preview-workspace-panel-shadow: var(--shadow-soft);

  gap: 12px;
  min-height: 0;
}
</style>
