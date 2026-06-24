<template>
  <div class="chat-shell">
    <PreviewTabs
      :active-tab="activeTab"
      :tabs="tabs"
      @update:active-tab="$emit('update:activeTab', $event)"
    />

    <ChatWorkspace
      v-if="activeTab === 'chat'"
      :archived-sessions="archivedSessions"
      :book-id="bookId"
      :chat-exporting="chatExporting"
      :chat-importing="chatImporting"
      :chat-loading="chatLoading"
      :chat-mutating="chatMutating"
      :chat-prompt-loading="chatPromptLoading"
      :chat-streaming="chatStreaming"
      :chat-summarizing="chatSummarizing"
      :document="document"
      :session="session"
      @delete-message="$emit('delete-message', $event)"
      @edit-message="$emit('edit-message', $event)"
      @export-session="$emit('export-session')"
      @import-session="$emit('import-session', $event)"
      @new-session="$emit('new-session', $event)"
      @open-greeting-picker="openGreetingPicker"
      @open-image-preview="openImagePreview"
      @open-prompt-preview="openPromptPreviewModal"
      @regenerate-message="$emit('regenerate-message', $event)"
      @send-chat="$emit('send-chat', $event)"
      @summarize-session="$emit('summarize-session', $event)"
      @switch-session="$emit('switch-session', $event)"
    />

    <AgentWorkspace
      v-else-if="activeTab === 'assistant'"
      v-model:agent-input="agentInput"
      :agent-busy="agentBusy"
      :agent-html-preview="agentHtmlPreview"
      :agent-messages="agentMessages"
      :can-undo-patch="canUndoPatch"
      :document="document"
      :patch-summary-sections="patchSummarySections"
      :pending-patch="pendingPatch"
      @apply-patch="$emit('apply-patch')"
      @send-agent="sendAgent"
      @undo-patch="$emit('undo-patch')"
    />

    <RuntimeWorkspace
      v-else
      :latest-runtime-message="latestRuntimeMessage"
      :summarize-log="summarizeLog"
    />

    <CharacterStudioPreviewModals
      v-model:greeting-open="greetingPickerOpen"
      v-model:image-open="imagePreviewOpen"
      v-model:prompt-open="promptPreviewModalOpen"
      v-model:selected-greeting-id="selectedGreetingId"
      :chat-mutating="chatMutating"
      :chat-prompt-loading="chatPromptLoading"
      :chat-streaming="chatStreaming"
      :display-greetings="displayGreetings"
      :image-src="imagePreviewSrc"
      :image-title="imagePreviewTitle"
      :prompt-preview="promptPreview"
      :prompt-preview-error="promptPreviewError"
      @confirm-greeting-selection="confirmGreetingSelection"
      @copy-prompt-preview="copyPromptPreview"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { getCharacterStudioChatAttachmentUrl } from '@/api/characterStudio'
import { buildCharacterStudioPatchSummary } from '@/stores/characterStudioPatchSummary'
import { buildCharacterStudioGreetingOptions } from '@/utils/characterStudioGreetings'
import CharacterStudioPreviewModals from './CharacterStudioPreviewModals.vue'
import { studioPreviewTabs } from './characterStudioEditorConfig'
import AgentWorkspace from './preview/AgentWorkspace.vue'
import ChatWorkspace from './preview/ChatWorkspace.vue'
import PreviewTabs from './preview/PreviewTabs.vue'
import RuntimeWorkspace from './preview/RuntimeWorkspace.vue'
import { summarizeStudioRuntimeLog as summarizeLog } from './characterStudioPreviewHelpers'
import type {
  CharacterStudioAgentPatchV2,
  CharacterStudioChatAttachment,
  CharacterStudioChatSession,
  CharacterStudioChatSessionSummary,
  CharacterStudioDocument,
} from '@/types/characterStudio'

const props = defineProps<{
  activeTab: 'chat' | 'assistant' | 'runtime'
  agentBusy: boolean
  agentHtmlPreview: string
  agentMessages: Array<{ role: 'user' | 'assistant'; content: string }>
  archivedSessions: CharacterStudioChatSessionSummary[]
  bookId: string
  canUndoPatch: boolean
  chatExporting: boolean
  chatImporting: boolean
  chatLoading: boolean
  chatMutating: boolean
  chatPromptLoading: boolean
  chatStreaming: boolean
  chatSummarizing: boolean
  document: CharacterStudioDocument | null
  pendingPatch: CharacterStudioAgentPatchV2 | null
  promptPreview: string
  promptPreviewError: string
  session: CharacterStudioChatSession | null
}>()

const emit = defineEmits<{
  (event: 'apply-patch'): void
  (event: 'delete-message', messageId: string): void
  (event: 'edit-message', value: { messageId: string; content: string }): void
  (event: 'export-session'): void
  (event: 'import-session', file: File): void
  (event: 'load-prompt-preview'): void
  (event: 'new-session', greetingId?: string): void
  (event: 'regenerate-message', messageId: string): void
  (event: 'send-agent', value: string): void
  (event: 'send-chat', value: { content: string; attachments: File[] }): void
  (event: 'summarize-session', cutoffMessageId?: string): void
  (event: 'switch-session', sessionId: string): void
  (event: 'undo-patch'): void
  (event: 'update:activeTab', value: 'chat' | 'assistant' | 'runtime'): void
}>()

const tabs = studioPreviewTabs
const agentInput = ref('')
const selectedGreetingId = ref('')
const greetingPickerOpen = ref(false)
const promptPreviewModalOpen = ref(false)
const imagePreviewOpen = ref(false)
const imagePreviewAttachment = ref<CharacterStudioChatAttachment | null>(null)

const latestRuntimeMessage = computed(() => {
  const messages = props.session?.messages || []
  return [...messages].reverse().find(item => item.role === 'assistant' && item.runtime_log.length > 0) || null
})

const displayGreetings = computed(() => buildCharacterStudioGreetingOptions(props.document))
const currentGreetingId = computed(() => {
  const source = props.session?.greeting_source || {}
  if (source.type === 'first_message') {
    const hasFirstMessage = displayGreetings.value.some(item => item.greeting_id === 'first_message')
    if (hasFirstMessage) return 'first_message'
  }
  if (source.type === 'alternate_greetings' && typeof source.index === 'number') {
    const greetingId = `alternate_${source.index + 1}`
    const hasAlternate = displayGreetings.value.some(item => item.greeting_id === greetingId)
    if (hasAlternate) return greetingId
  }
  return displayGreetings.value[0]?.greeting_id || ''
})

const patchSummarySections = computed(() => buildCharacterStudioPatchSummary(props.pendingPatch, props.document))
const imagePreviewTitle = computed(() => imagePreviewAttachment.value?.filename || '图片预览')
const imagePreviewSrc = computed(() => (
  imagePreviewAttachment.value ? attachmentUrl(imagePreviewAttachment.value) : ''
))

watch(() => props.session?.session_id, () => {
  selectedGreetingId.value = ''
})

function openGreetingPicker() {
  if (!displayGreetings.value.length) return
  selectedGreetingId.value = currentGreetingId.value || displayGreetings.value[0]?.greeting_id || ''
  greetingPickerOpen.value = true
}

function confirmGreetingSelection() {
  if (!selectedGreetingId.value) return
  emit('new-session', selectedGreetingId.value)
  greetingPickerOpen.value = false
}

function openPromptPreviewModal() {
  promptPreviewModalOpen.value = true
  emit('load-prompt-preview')
}

async function copyPromptPreview() {
  if (!props.promptPreview.trim()) return
  await navigator.clipboard.writeText(props.promptPreview)
}

function sendAgent() {
  const value = agentInput.value.trim()
  if (!value) return
  emit('send-agent', value)
  agentInput.value = ''
}

function attachmentUrl(attachment: CharacterStudioChatAttachment) {
  if (!props.bookId || !props.document) return attachment.asset_path
  return getCharacterStudioChatAttachmentUrl(props.bookId, props.document.id, attachment.asset_path)
}

function openImagePreview(attachment: CharacterStudioChatAttachment) {
  imagePreviewAttachment.value = attachment
  imagePreviewOpen.value = true
}
</script>

<style scoped>
.chat-shell {
  --character-studio-preview-details-border-default: rgba(37, 99, 199, .28);
  --character-studio-preview-details-shadow-default: rgba(37, 99, 199, .16);
  --character-studio-preview-details-shadow-raised: rgba(37, 99, 199, .18);
  --character-studio-preview-details-surface-base: rgba(20, 56, 106, .06);
  --character-studio-preview-details-surface-active: #2563c7;
  --character-studio-preview-details-surface-selected: #4d86ee;
  --character-studio-preview-details-text-primary: #16365b;
  --character-studio-preview-shell-border-default: rgba(28, 55, 94, .1);
  --character-studio-preview-shell-shadow-default: rgba(20, 46, 82, .06);
  --character-studio-preview-shell-shadow-floating: rgba(255, 255, 255, .5);
  --character-studio-preview-shell-shadow-raised: rgba(37, 99, 199, .16);
  --character-studio-preview-shell-shadow-strong: rgba(20, 46, 82, .18);
  --character-studio-preview-shell-surface-base: rgba(77, 134, 238, .1);
  --character-studio-preview-shell-surface-muted: rgba(255, 255, 255, .96);
  --character-studio-preview-shell-surface-raised: rgba(252, 253, 255, .92);
  --character-studio-preview-shell-text-disabled: #6f84a2;
  --character-studio-preview-shell-text-muted: #16365b;
  --character-studio-preview-shell-text-primary: #102741;
  --character-studio-preview-shell-text-secondary: #55708f;
  --character-studio-preview-shell-text-subtle: #14304c;
  --character-studio-preview-shell-text-supporting: #5f7591;
  --character-studio-preview-workspace-border-default: rgba(25, 55, 94, .08);
  --character-studio-preview-workspace-surface-base: rgba(244, 248, 255, .95);
  --character-studio-preview-workspace-surface-hover: rgba(255, 255, 255, .88);
  --character-studio-preview-workspace-surface-muted: rgba(247, 250, 254, .96);
  --character-studio-preview-workspace-surface-overlay: rgba(244, 248, 255, .94);
  --character-studio-preview-workspace-surface-raised: rgba(238, 244, 252, .9);
  --character-studio-preview-workspace-surface-soft: rgba(244, 248, 255, .88);
  --character-studio-preview-workspace-surface-subtle: rgba(20, 56, 106, .08);
  --character-studio-preview-workspace-surface-tint: rgba(255, 255, 255, .86);
  --character-studio-preview-workspace-text-primary: #5f7591;

  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
  max-width: none;
  height: 100%;
  min-height: 0;
  padding: 0;
}
</style>
