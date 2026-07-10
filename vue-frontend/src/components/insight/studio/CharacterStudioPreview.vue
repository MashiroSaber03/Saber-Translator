<template>
  <div class="character-studio-preview">
    <ProductSegmentedTabs
      :active-tab="activeTab"
      aria-label="角色工坊预览工作区"
      :tabs="tabs"
      @update:active-tab="selectTab"
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
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import { buildCharacterStudioPatchSummary } from '@/stores/characterStudioPatchSummary'
import { buildCharacterStudioGreetingOptions } from '@/utils/characterStudioGreetings'
import { copyTextToClipboard } from '@/utils/clipboard'
import CharacterStudioPreviewModals from './CharacterStudioPreviewModals.vue'
import { studioPreviewTabs } from './characterStudioEditorConfig'
import AgentWorkspace from './preview/AgentWorkspace.vue'
import ChatWorkspace from './preview/ChatWorkspace.vue'
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

const tabs = studioPreviewTabs.map(tab => ({
  id: tab.value,
  iconName: tab.iconName,
  label: tab.label,
}))
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
  await copyTextToClipboard(props.promptPreview)
}

function sendAgent() {
  const value = agentInput.value.trim()
  if (!value) return
  emit('send-agent', value)
  agentInput.value = ''
}

function selectTab(tabId: string) {
  const tab = studioPreviewTabs.find(item => item.value === tabId)
  if (!tab) return
  emit('update:activeTab', tab.value)
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
.character-studio-preview {
  --product-segmented-tabs-background: color-mix(in srgb, var(--color-surface-card) 94%, transparent);
  --product-segmented-tabs-border: var(--studio-border-default);
  --product-segmented-tabs-active-background: color-mix(in srgb, var(--color-action-brand) 12%, var(--color-surface-card));
  --product-segmented-tabs-active-text: var(--studio-text-strong);
  --product-segmented-tabs-text: var(--studio-text-muted);
  --product-segmented-tabs-gap: 8px;
  --product-segmented-tabs-padding: 6px;
  --product-segmented-tabs-radius: 20px;
  --product-segmented-tabs-shadow: 0 18px 32px var(--studio-shadow-floating);
  --product-segmented-tabs-tab-padding: 10px 14px;
  --product-segmented-tabs-tab-radius: 14px;
  --product-segmented-tabs-active-shadow: inset 0 0 0 1px color-mix(in srgb, var(--color-action-brand) 24%, transparent);

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
