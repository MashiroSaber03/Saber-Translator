<template>
  <div class="chat-shell">
    <div class="workspace-tabs" role="tablist">
      <UiButton
        variant="toolbar"
        v-for="item in tabs"
        :key="item.value"
        class="tab-btn"
        :class="{ active: activeTab === item.value }"
        @click="$emit('update:activeTab', item.value)"
      >
        <span>{{ item.icon }}</span>
        <strong>{{ item.label }}</strong>
      </UiButton>
    </div>

    <section v-if="activeTab === 'chat'" class="workspace-card chat-workspace">
      <div class="session-toolbar">
        <div class="session-triggers">
          <div class="trigger-stack trigger-stack-wide">
            <UiButton
              variant="toolbar"
              data-testid="session-list-trigger"
              class="session-trigger session-trigger-inline"
              :disabled="chatMutating || chatStreaming"
              @click="toggleSessionList"
            >
              <div class="trigger-copy trigger-copy-inline">
                <span class="trigger-tag">会话</span>
                <strong>{{ currentSessionLabel }}</strong>
                <span class="trigger-meta">{{ currentSessionMeta }}</span>
              </div>
              <span class="trigger-arrow">▾</span>
            </UiButton>
            <div v-if="sessionListOpen" ref="sessionListRef" class="session-list-panel">
              <UiButton
                variant="toolbar"
                class="session-list-item current"
                :class="{ active: session?.session_id === currentSessionId }"
                @click="closeSessionList"
              >
                <div class="item-main">
                  <strong>{{ currentSessionLabel }}</strong>
                  <p>{{ currentSessionExcerpt || '当前活跃会话' }}</p>
                </div>
                <div class="item-meta">
                  <span>{{ currentSessionMeta }}</span>
                  <span class="item-badge">当前</span>
                </div>
              </UiButton>
              <div v-if="archivedSessions.length === 0" class="session-list-empty">还没有归档会话。</div>
              <UiButton
                variant="toolbar"
                v-for="item in archivedSessions"
                :key="item.session_id"
                class="session-list-item"
                @click="chooseSession(item.session_id)"
              >
                <div class="item-main">
                  <strong>{{ item.title }}</strong>
                  <p>{{ item.last_message_excerpt || '暂无摘要' }}</p>
                </div>
                <div class="item-meta">
                  <span>{{ item.message_count }} 条</span>
                  <span>{{ formatSessionTime(item.updated_at) }}</span>
                </div>
              </UiButton>
            </div>
          </div>

          <div class="trigger-stack">
            <UiButton
              variant="toolbar"
              data-testid="greeting-picker-trigger"
              class="session-trigger session-trigger-inline"
              :disabled="displayGreetings.length === 0 || chatMutating || chatStreaming"
              @click="openGreetingPicker"
            >
              <div class="trigger-copy trigger-copy-inline">
                <span class="trigger-tag">开场白</span>
                <strong>{{ currentGreetingLabel }}</strong>
              </div>
              <span class="trigger-arrow">▾</span>
            </UiButton>
          </div>
        </div>
        <div class="toolbar-buttons">
          <UiButton variant="toolbar" class="action-ghost" :disabled="!document || chatMutating || chatStreaming" @click="$emit('new-session')" size="sm">
            新对话
          </UiButton>
          <UiButton
            variant="toolbar"
            data-testid="prompt-preview-trigger"
            class="action-ghost"
            :disabled="!document || chatPromptLoading || chatStreaming"
            @click="openPromptPreviewModal" size="sm"
          >
            {{ chatPromptLoading ? '加载中...' : '查看提示词' }}
          </UiButton>
          <UiButton variant="toolbar" class="action-ghost" :disabled="!document || chatMutating || chatStreaming" @click="openGreetingPicker" size="sm">
            重选开场白
          </UiButton>
          <UiButton variant="toolbar" class="action-ghost" :disabled="!session || chatSummarizing || chatStreaming" @click="$emit('summarize-session')" size="sm">
            {{ chatSummarizing ? '总结中...' : '手动总结' }}
          </UiButton>
          <UiButton variant="toolbar" class="action-ghost" :disabled="!session || chatExporting || chatStreaming" @click="$emit('export-session')" size="sm">
            {{ chatExporting ? '导出中...' : '导出聊天' }}
          </UiButton>
          <UiButton variant="toolbar" class="action-ghost" :disabled="chatImporting || chatStreaming" @click="pickImport" size="sm">
            {{ chatImporting ? '导入中...' : '导入聊天' }}
          </UiButton>
        </div>
        <UiFileInput ref="importInput" hidden accept=".json" @change="handleImportChange" />
      </div>

      <div v-if="!document" class="empty-copy">选择角色文档后可开始聊天。</div>
      <template v-else-if="!session">
        <div class="empty-copy">{{ chatLoading ? '聊天会话加载中...' : '当前还没有聊天会话。' }}</div>
      </template>
      <template v-else>
        <div class="messages-panel">
          <div v-if="session.messages.length === 0" class="empty-copy">当前会话还没有消息。</div>
          <article v-for="item in session.messages" :key="item.message_id" class="message-card" :class="item.role">
            <div class="message-head">
              <span class="message-role">{{ item.role === 'assistant' ? (document.identity.name || '角色') : '你' }}</span>
              <div class="message-actions">
                <UiButton
                  variant="toolbar"
                  v-if="canEditMessage(item)"
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
                  variant="toolbar"
                  v-if="canRegenerateMessage(item)"
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
                variant="toolbar"
                v-for="attachment in item.attachments"
                :key="attachment.attachment_id"
                type="button"
                class="attachment-card"
                @click="openImagePreview(attachment)"
              >
                <div class="attachment-frame">
                  <img
                    v-if="attachment.mime_type.startsWith('image/')"
                    :src="attachmentUrl(attachment)"
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

        <div class="composer-card">
          <div v-if="pendingFiles.length > 0" class="pending-files">
            <UiButton
              variant="toolbar"
              v-for="(file, index) in pendingFiles"
              :key="file.id"
              type="button"
              class="pending-image-card"
            >
              <div class="pending-image-thumb">
                <img :src="file.previewUrl" :alt="file.file.name">
              </div>
              <div class="pending-image-copy">
                <strong>{{ file.file.name }}</strong>
                <span>{{ attachmentTypeLabel(file.file.type || 'application/octet-stream') }}</span>
              </div>
              <span class="pending-remove" @click.stop="removePendingFile(index)">×</span>
            </UiButton>
          </div>
          <div class="composer-main">
            <UiTextarea
              v-model="chatInput"
              class="chat-composer-input"
              rows="1"
              placeholder="输入消息，或添加图片后让角色结合画面继续聊天。"
            />
            <div class="composer-actions compact-actions">
              <UiButton
                variant="toolbar"
                data-testid="chat-upload-trigger"
                class="action-ghost icon-btn"
                type="button"
                title="添加图片"
                aria-label="添加图片"
                :disabled="chatStreaming"
                @click="pickAttachments"
              >
                +
              </UiButton>
              <UiButton
                variant="toolbar"
                data-testid="chat-send-trigger"
                class="action-primary icon-btn"
                type="button"
                :title="chatStreaming ? '回复生成中...' : '发送消息'"
                :aria-label="chatStreaming ? '回复生成中...' : '发送消息'"
                :disabled="chatStreaming || (!chatInput.trim() && pendingFiles.length === 0)"
                @click="sendChat"
              >
                {{ chatStreaming ? '…' : '↗' }}
              </UiButton>
            </div>
          </div>
          <UiFileInput ref="attachmentInput" hidden accept="image/*" multiple @change="handleAttachmentChange" />
        </div>
      </template>
    </section>

    <section v-else-if="activeTab === 'assistant'" class="workspace-card assistant-workspace">
      <div class="assistant-head">
        <div>
          <h4>卡片助手</h4>
          <p>围绕角色卡本体给出结构化建议，可应用 patch 或撤销。</p>
        </div>
        <div class="assistant-actions">
          <UiButton variant="toolbar" class="action-ghost" :disabled="!pendingPatch" @click="$emit('apply-patch')" size="sm">应用 patch</UiButton>
          <UiButton variant="toolbar" class="action-ghost" :disabled="!canUndoPatch" @click="$emit('undo-patch')" size="sm">撤销 patch</UiButton>
        </div>
      </div>

      <div class="assistant-main">
        <div class="messages-panel assistant-messages">
          <div v-if="agentMessages.length === 0" class="empty-copy">还没有与卡片助手对话。</div>
          <article v-for="(item, index) in agentMessages" :key="`agent-${index}`" class="message-card" :class="item.role">
            <div class="message-head">
              <span class="message-role">{{ item.role === 'assistant' ? '卡片助手' : '你' }}</span>
            </div>
            <pre class="agent-text">{{ item.content }}</pre>
          </article>
        </div>

        <div class="composer-card assistant-composer">
          <div class="composer-main">
            <UiTextarea
              v-model="agentInput"
              class="chat-composer-input"
              rows="1"
              placeholder="例如：请审查当前角色卡，并建议补充世界书与状态任务。"
            />
            <div class="composer-actions compact-actions">
              <UiButton
                variant="toolbar"
                data-testid="assistant-send-trigger"
                class="action-primary icon-btn"
                type="button"
                :title="agentBusy ? '助手处理中...' : '发送给助手'"
                :aria-label="agentBusy ? '助手处理中...' : '发送给助手'"
                :disabled="agentBusy || !agentInput.trim() || !document"
                @click="sendAgent"
              >
                {{ agentBusy ? '…' : '↗' }}
              </UiButton>
            </div>
          </div>
        </div>
      </div>

      <div v-if="pendingPatch" class="prompt-preview-card">
        <h4>待应用 Patch</h4>
        <div v-if="patchSummarySections.length > 0" class="patch-summary">
          <section
            v-for="section in patchSummarySections"
            :key="section.key"
            class="patch-summary-section"
          >
            <div class="patch-summary-head">
              <strong>{{ section.title }}</strong>
              <span>{{ section.items.length }} 项</span>
            </div>
            <ul class="patch-summary-list">
              <li v-for="(item, index) in section.items" :key="`${section.key}-${index}`">{{ item }}</li>
            </ul>
          </section>
        </div>
        <details class="patch-raw-details">
          <summary>查看原始 JSON</summary>
          <pre>{{ JSON.stringify(pendingPatch, null, 2) }}</pre>
        </details>
      </div>

      <div v-if="agentHtmlPreview" class="html-preview-card">
        <h4>HTML 预览块</h4>
        <iframe class="preview-frame" :srcdoc="agentHtmlPreview" sandbox="allow-scripts"></iframe>
      </div>
    </section>

    <section v-else class="workspace-card runtime-workspace">
      <div class="assistant-head">
        <div>
          <h4>运行日志</h4>
          <p>查看最新一轮的变量快照、世界书命中、正则命中与任务执行记录。</p>
        </div>
      </div>
      <div class="runtime-main">
        <template v-if="latestRuntimeMessage">
          <div class="runtime-grid">
            <section class="runtime-card">
              <h5>变量快照</h5>
              <pre>{{ JSON.stringify(latestRuntimeMessage.variables_snapshot || {}, null, 2) }}</pre>
            </section>
            <section class="runtime-card">
              <h5>运行日志</h5>
              <div v-if="latestRuntimeMessage.runtime_log.length > 0" class="log-list">
                <div v-for="(item, index) in latestRuntimeMessage.runtime_log" :key="`runtime-${index}`" class="log-item">
                  {{ summarizeLog(item) }}
                </div>
              </div>
              <div v-else class="empty-copy">当前还没有运行日志。</div>
            </section>
          </div>
        </template>
        <div v-else class="messages-panel runtime-empty-panel">
          <div class="empty-copy">发送消息后，这里会显示最新一轮的运行结果。</div>
        </div>
      </div>
    </section>

    <CharacterStudioPreviewModals
      v-model:greeting-open="greetingPickerOpen"
      v-model:prompt-open="promptPreviewModalOpen"
      v-model:image-open="imagePreviewOpen"
      v-model:selected-greeting-id="selectedGreetingId"
      :display-greetings="displayGreetings"
      :chat-mutating="chatMutating"
      :chat-streaming="chatStreaming"
      :chat-prompt-loading="chatPromptLoading"
      :prompt-preview="promptPreview"
      :prompt-preview-error="promptPreviewError"
      :image-title="imagePreviewTitle"
      :image-src="imagePreviewSrc"
      @confirm-greeting-selection="confirmGreetingSelection"
      @copy-prompt-preview="copyPromptPreview"
    />
  </div>
</template>

<script setup lang="ts">

import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { getCharacterStudioChatAttachmentUrl } from '@/api/characterStudio'
import { buildCharacterStudioPatchSummary } from '@/stores/characterStudioPatchSummary'
import { buildCharacterStudioGreetingOptions } from '@/utils/characterStudioGreetings'
import CharacterStudioPreviewModals from './CharacterStudioPreviewModals.vue'
import { studioPreviewTabs } from './characterStudioEditorConfig'
import {
  attachmentTypeLabel,
  canEditStudioChatMessage as canEditMessage,
  canRegenerateStudioChatMessage as canRegenerateMessage,
  formatSessionTime,
  type PendingAttachmentCard,
  summarizeStudioRuntimeLog as summarizeLog,
} from './characterStudioPreviewHelpers'
import type {
  CharacterStudioAgentPatchV2,
  CharacterStudioChatAttachment,
  CharacterStudioChatSession,
  CharacterStudioChatSessionSummary,
  CharacterStudioDocument,
} from '@/types/characterStudio'

const props = defineProps<{
  bookId: string
  document: CharacterStudioDocument | null
  session: CharacterStudioChatSession | null
  archivedSessions: CharacterStudioChatSessionSummary[]
  promptPreview: string
  promptPreviewError: string
  activeTab: 'chat' | 'assistant' | 'runtime'
  chatLoading: boolean
  chatStreaming: boolean
  chatMutating: boolean
  chatSummarizing: boolean
  chatExporting: boolean
  chatImporting: boolean
  chatPromptLoading: boolean
  agentBusy: boolean
  agentMessages: Array<{ role: 'user' | 'assistant'; content: string }>
  pendingPatch: CharacterStudioAgentPatchV2 | null
  canUndoPatch: boolean
  agentHtmlPreview: string
}>()

const emit = defineEmits<{
  (e: 'update:activeTab', value: 'chat' | 'assistant' | 'runtime'): void
  (e: 'send-chat', value: { content: string; attachments: File[] }): void
  (e: 'edit-message', value: { messageId: string; content: string }): void
  (e: 'delete-message', messageId: string): void
  (e: 'regenerate-message', messageId: string): void
  (e: 'new-session', greetingId?: string): void
  (e: 'switch-session', sessionId: string): void
  (e: 'summarize-session', cutoffMessageId?: string): void
  (e: 'export-session'): void
  (e: 'import-session', file: File): void
  (e: 'load-prompt-preview'): void
  (e: 'send-agent', value: string): void
  (e: 'apply-patch'): void
  (e: 'undo-patch'): void
}>()

const tabs = studioPreviewTabs
const chatInput = ref('')
const agentInput = ref('')
const pendingFiles = ref<PendingAttachmentCard[]>([])
const selectedGreetingId = ref('')
const sessionListOpen = ref(false)
const greetingPickerOpen = ref(false)
const promptPreviewModalOpen = ref(false)
const imagePreviewOpen = ref(false)
const imagePreviewAttachment = ref<CharacterStudioChatAttachment | null>(null)
const editingMessageId = ref('')
const editingContent = ref('')
const attachmentInput = ref<HTMLInputElement | null>(null)
const importInput = ref<HTMLInputElement | null>(null)
const sessionListRef = ref<HTMLElement | null>(null)

const latestRuntimeMessage = computed(() => {
  const messages = props.session?.messages || []
  return [...messages].reverse().find(item => item.role === 'assistant' && item.runtime_log.length > 0) || null
})

const currentSessionId = computed(() => props.session?.session_id || '')
const currentSessionLabel = computed(() => props.session?.title || '当前会话')
const currentSessionExcerpt = computed(() => {
  const messages = props.session?.messages || []
  const last = messages[messages.length - 1]
  return last?.content || ''
})
const currentSessionMeta = computed(() => {
  const count = props.session?.messages.length || 0
  return `${count} 条消息`
})
const displayGreetings = computed(() => {
  return buildCharacterStudioGreetingOptions(props.document)
})
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
const currentGreetingLabel = computed(() => {
  const selected = displayGreetings.value.find(item => item.greeting_id === currentGreetingId.value)
  return selected?.label || '选择开场白'
})

const patchSummarySections = computed(() => {
  return buildCharacterStudioPatchSummary(props.pendingPatch, props.document)
})

const imagePreviewTitle = computed(() => imagePreviewAttachment.value?.filename || '图片预览')
const imagePreviewSrc = computed(() => (
  imagePreviewAttachment.value ? attachmentUrl(imagePreviewAttachment.value) : ''
))

watch(() => props.session?.session_id, () => {
  selectedGreetingId.value = ''
  sessionListOpen.value = false
})

function pickAttachments() {
  attachmentInput.value?.click()
}

function handleAttachmentChange(event: Event) {
  const target = event.target as HTMLInputElement
  const files = Array.from(target.files || [])
  pendingFiles.value = [
    ...pendingFiles.value,
    ...files.map(file => ({
      id: `pending-${Date.now()}-${Math.random().toString(16).slice(2, 6)}`,
      file,
      previewUrl: URL.createObjectURL(file),
    })),
  ]
  target.value = ''
}

function removePendingFile(index: number) {
  const removed = pendingFiles.value[index]
  if (removed) {
    URL.revokeObjectURL(removed.previewUrl)
  }
  pendingFiles.value.splice(index, 1)
}

function sendChat() {
  const content = chatInput.value.trim()
  if (!content && pendingFiles.value.length === 0) return
  emit('send-chat', { content, attachments: pendingFiles.value.map(item => item.file) })
  chatInput.value = ''
  pendingFiles.value.forEach(item => URL.revokeObjectURL(item.previewUrl))
  pendingFiles.value = []
}

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

function pickImport() {
  importInput.value?.click()
}

function handleImportChange(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return
  emit('import-session', file)
  target.value = ''
}

function toggleSessionList() {
  sessionListOpen.value = !sessionListOpen.value
}

function closeSessionList() {
  sessionListOpen.value = false
}

function chooseSession(sessionId: string) {
  if (!sessionId || sessionId === props.session?.session_id) {
    closeSessionList()
    return
  }
  emit('switch-session', sessionId)
  closeSessionList()
}

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

function handleDocumentClick(event: MouseEvent) {
  if (!sessionListOpen.value) return
  if (sessionListRef.value?.contains(event.target as Node)) return
  const trigger = document.querySelector('[data-testid="session-list-trigger"]')
  if (trigger instanceof HTMLElement && trigger.contains(event.target as Node)) return
  sessionListOpen.value = false
}

onMounted(() => {
  document.addEventListener('click', handleDocumentClick)
})

onUnmounted(() => {
  document.removeEventListener('click', handleDocumentClick)
  pendingFiles.value.forEach(item => URL.revokeObjectURL(item.previewUrl))
})
</script>

<style scoped>
.chat-shell {
  --character-studio-preview-details-border-default: rgba(37, 99, 199, .28);
  --character-studio-preview-details-shadow-default: rgba(37, 99, 199, .16);
  --character-studio-preview-details-shadow-raised: rgba(37, 99, 199, .18);
  --character-studio-preview-details-surface-base: rgba(20, 56, 106, .06);
  --character-studio-preview-details-surface-raised: rgba(244, 248, 255, .84);
  --character-studio-preview-details-surface-muted: rgba(237, 244, 255, .96);
  --character-studio-preview-details-surface-subtle: rgba(244, 248, 255, .92);
  --character-studio-preview-details-surface-hover: rgba(244, 248, 255, .9);
  --character-studio-preview-details-surface-active: #2563c7;
  --character-studio-preview-details-surface-selected: #4d86ee;
  --character-studio-preview-details-text-primary: #16365b;
  --character-studio-preview-shell-border-default: rgba(28, 55, 94, .1);
  --character-studio-preview-shell-shadow-default: rgba(20, 46, 82, .06);
  --character-studio-preview-shell-shadow-raised: rgba(37, 99, 199, .16);
  --character-studio-preview-shell-shadow-floating: rgba(255, 255, 255, .5);
  --character-studio-preview-shell-shadow-strong: rgba(20, 46, 82, .18);
  --character-studio-preview-shell-surface-base: rgba(77, 134, 238, .1);
  --character-studio-preview-shell-surface-raised: rgba(252, 253, 255, .92);
  --character-studio-preview-shell-surface-muted: rgba(255, 255, 255, .96);
  --character-studio-preview-shell-surface-subtle: rgba(20, 56, 106, .06);
  --character-studio-preview-shell-surface-hover: rgba(255, 255, 255, .98);
  --character-studio-preview-shell-surface-active: rgba(37, 99, 199, .08);
  --character-studio-preview-shell-text-primary: #102741;
  --character-studio-preview-shell-text-secondary: #55708f;
  --character-studio-preview-shell-text-muted: #16365b;
  --character-studio-preview-shell-text-subtle: #14304c;
  --character-studio-preview-shell-text-supporting: #5f7591;
  --character-studio-preview-shell-text-disabled: #6f84a2;
  --character-studio-preview-workspace-border-default: rgba(25, 55, 94, .08);
  --character-studio-preview-workspace-surface-base: rgba(244, 248, 255, .95);
  --character-studio-preview-workspace-surface-raised: rgba(238, 244, 252, .9);
  --character-studio-preview-workspace-surface-muted: rgba(247, 250, 254, .96);
  --character-studio-preview-workspace-surface-subtle: rgba(20, 56, 106, .08);
  --character-studio-preview-workspace-surface-hover: rgba(255, 255, 255, .88);
  --character-studio-preview-workspace-surface-active: rgba(225, 235, 250, .72);
  --character-studio-preview-workspace-surface-selected: rgba(241, 246, 255, .96);
  --character-studio-preview-workspace-surface-overlay: rgba(244, 248, 255, .94);
  --character-studio-preview-workspace-surface-inverse: rgba(255, 255, 255, .94);
  --character-studio-preview-workspace-surface-contrast: rgba(37, 99, 199, .08);
  --character-studio-preview-workspace-surface-tint: rgba(255, 255, 255, .86);
  --character-studio-preview-workspace-surface-soft: rgba(244, 248, 255, .88);
  --character-studio-preview-workspace-text-primary: #5f7591;
  --character-studio-preview-workspace-text-secondary: #14304c;

  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
  height: 100%;
  width: 100%;
  max-width: none;
  padding: 0;
}

.assistant-head,
.message-head,
.toolbar-buttons,
.prompt-head,
.composer-actions,
.editor-actions {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  justify-content: space-between;
}

.assistant-head h4,
.prompt-preview-card h4,
.html-preview-card h4,
.runtime-card h5 {
  margin: 8px 0 0;
  color: var(--character-studio-preview-shell-text-primary);
}

.assistant-head p {
  margin: 8px 0 0;
  color: var(--color-text-studio-muted);
  font-size: 13px;
  line-height: 1.7;
}

.workspace-tabs {
  display: flex;
  gap: 8px;
  padding: 6px;
  border-radius: 20px;
  background: var(--color-surface-raised);
  border: 1px solid var(--color-border-studio);
  width: 100%;
  box-shadow: 0 18px 32px var(--character-studio-preview-shell-shadow-default);
}

.tab-btn {
  display: inline-flex;
  flex: 1 1 0;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border: none;
  border-radius: 14px;
  padding: 10px 14px;
  background: transparent;
  color: var(--character-studio-preview-shell-text-secondary);
  cursor: pointer;
}

.tab-btn.active {
  background: linear-gradient(135deg, var(--color-surface-studio-tint-strong), var(--character-studio-preview-shell-surface-base));
  color: var(--character-studio-preview-shell-text-muted);
  box-shadow: inset 0 0 0 1px var(--character-studio-preview-shell-shadow-raised);
}

.workspace-card,
.prompt-preview-card,
.html-preview-card {
  border-radius: 24px;
  padding: 14px;
  background: var(--character-studio-preview-shell-surface-raised);
  border: 1px solid var(--color-border-studio);
  box-shadow: 0 24px 40px var(--shadow-studio-floating);
  width: 100%;
}

.workspace-card {
  display: flex;
  flex-direction: column;
  min-height: 0;
  flex: 1 1 auto;
}

.chat-workspace {
  min-height: 0;
  gap: 12px;
}

.assistant-workspace,
.runtime-workspace {
  min-height: 0;
}

.assistant-workspace {
  gap: 12px;
}

.runtime-workspace {
  gap: 12px;
}

.session-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px 12px;
  margin-bottom: 4px;
  width: 100%;
}

.session-triggers {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  flex: 1 1 440px;
  min-width: 0;
}

.trigger-stack {
  position: relative;
  flex: 1 1 220px;
  min-width: 0;
}

.trigger-stack-wide {
  flex: 1 1 260px;
}

.session-trigger {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  border: 1px solid var(--character-studio-preview-shell-border-default);
  background: linear-gradient(180deg, var(--character-studio-preview-shell-surface-muted), var(--color-surface-studio-soft));
  border-radius: 16px;
  padding: 12px 14px;
  color: var(--color-text-studio-strong);
  cursor: pointer;
  box-shadow: inset 0 1px 0 var(--character-studio-preview-shell-shadow-floating);
}

.session-trigger-inline {
  min-height: 46px;
  padding: 10px 14px;
}

.session-trigger:disabled {
  opacity: 0.62;
  cursor: not-allowed;
}

.trigger-copy {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  text-align: left;
}

.trigger-copy-inline {
  flex-direction: row;
  align-items: center;
  gap: 8px;
}

.trigger-copy strong {
  font-size: 14px;
  color: var(--character-studio-preview-shell-text-subtle);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.trigger-tag,
.trigger-meta {
  font-size: 11px;
  color: var(--character-studio-preview-shell-text-supporting);
  white-space: nowrap;
}

.trigger-tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 4px 8px;
  border-radius: 999px;
  background: var(--character-studio-preview-shell-surface-subtle);
}

.trigger-meta {
  overflow: hidden;
  text-overflow: ellipsis;
}

.trigger-arrow {
  color: var(--character-studio-preview-shell-text-supporting);
  flex-shrink: 0;
}

.session-list-panel {
  position: absolute;
  z-index: var(--z-local-overlay);
  top: calc(100% + 6px);
  left: 0;
  width: min(460px, calc(100vw - 80px));
  max-height: 420px;
  overflow: auto;
  border-radius: 20px;
  padding: 10px;
  background: var(--character-studio-preview-shell-surface-hover);
  border: 1px solid var(--color-border-studio);
  box-shadow: 0 18px 38px var(--character-studio-preview-shell-shadow-strong);
}

.session-list-item {
  width: 100%;
  border: none;
  background: transparent;
  border-radius: 16px;
  padding: 12px 14px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  text-align: left;
  cursor: pointer;
}

.session-list-item:hover,
.session-list-item.active {
  background: var(--character-studio-preview-shell-surface-active);
}

.session-list-item.current {
  border-bottom: 1px solid var(--color-border-studio);
  margin-bottom: 6px;
  padding-bottom: 14px;
}

.session-list-empty {
  padding: 12px 14px;
  color: var(--color-text-studio-subtle);
  font-size: 13px;
}

.item-main {
  min-width: 0;
}

.item-main strong {
  display: block;
  color: var(--character-studio-preview-shell-text-subtle);
  font-size: 14px;
}

.item-main p {
  margin: 6px 0 0;
  color: var(--color-text-studio-muted);
  font-size: 12px;
  line-height: 1.5;
}

.item-meta {
  display: flex;
  flex-direction: column;
  gap: 6px;
  align-items: flex-end;
  color: var(--character-studio-preview-shell-text-disabled);
  font-size: 11px;
}

.item-badge {
  display: inline-flex;
  border-radius: 999px;
  padding: 4px 8px;
  background: var(--color-surface-studio-tint-muted);
  color: var(--color-text-primary-strong);
}

.composer-card textarea,
.editor-row textarea {
  width: 100%;
  border: 1px solid var(--color-border-studio-strong);
  background: var(--color-surface-studio-soft);
  border-radius: 14px;
  padding: 10px 12px;
  color: var(--color-text-studio-strong);
  font-size: 13px;
}

.composer-main {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 10px;
  align-items: stretch;
}

.chat-composer-input {
  min-height: 64px;
  resize: vertical;
}

.toolbar-buttons {
  justify-content: flex-end;
  flex-wrap: wrap;
  align-items: center;
  flex: 0 0 auto;
}

.messages-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1 1 auto;
  min-height: 0;
  overflow: auto;
  padding: 12px;
  border-radius: 20px;
  background: linear-gradient(180deg, var(--character-studio-preview-workspace-surface-base), var(--character-studio-preview-workspace-surface-raised));
  border: 1px solid var(--color-border-studio);
}

.assistant-main {
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1 1 auto;
  min-height: 0;
}

.assistant-messages {
  flex: 1 1 auto;
  min-height: 0;
}

.assistant-composer {
  margin-top: 0;
  flex: 0 0 auto;
}

.runtime-main {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  min-height: 0;
}

.message-card {
  border-radius: 18px;
  padding: 14px;
  border: 1px solid var(--color-border-studio);
  background: var(--character-studio-preview-workspace-surface-muted);
  width: min(100%, 88%);
}

.message-card.user {
  margin-left: auto;
  background: var(--character-studio-preview-workspace-surface-subtle);
}

.message-card.assistant {
  margin-right: auto;
  background: var(--color-surface-studio-tint);
}

.message-role {
  font-size: 11px;
  color: var(--character-studio-preview-workspace-text-primary);
}

.message-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.message-body,
.agent-text {
  margin-top: 8px;
  color: var(--color-text-studio-strong);
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
}

.agent-text {
  font-family: inherit;
}

.attachment-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 10px;
  margin-top: 12px;
}

.attachment-card {
  margin: 0;
  padding: 0;
  text-align: left;
  cursor: pointer;
  border-radius: 14px;
  overflow: hidden;
  background: var(--character-studio-preview-workspace-surface-hover);
  border: 1px solid var(--color-border-studio);
}

.attachment-frame {
  aspect-ratio: 1 / 1;
  overflow: hidden;
  background: linear-gradient(180deg, var(--character-studio-preview-workspace-surface-active), var(--character-studio-preview-workspace-surface-selected));
}

.attachment-card img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.attachment-info {
  padding: 10px 12px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.attachment-info strong {
  color: var(--character-studio-preview-workspace-text-secondary);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.attachment-info span {
  font-size: 11px;
  color: var(--color-text-studio-muted);
}

.composer-card {
  margin-top: 2px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 12px;
  border-radius: 20px;
  background: var(--character-studio-preview-workspace-surface-overlay);
  border: 1px solid var(--color-border-studio);
}

.compact-actions {
  flex-direction: column;
  justify-content: flex-end;
  align-items: stretch;
  gap: 6px;
}

.icon-btn {
  width: 44px;
  min-width: 44px;
  height: 44px;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  line-height: 1;
}

.pending-files {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 10px;
  flex-wrap: wrap;
}

.pending-image-card {
  position: relative;
  display: grid;
  grid-template-columns: 56px minmax(0, 1fr);
  gap: 10px;
  align-items: center;
  border: 1px solid var(--color-border-studio);
  background: var(--character-studio-preview-workspace-surface-inverse);
  border-radius: 16px;
  padding: 10px 12px;
  text-align: left;
  cursor: default;
}

.pending-image-thumb {
  width: 56px;
  height: 56px;
  border-radius: 12px;
  overflow: hidden;
  background: var(--character-studio-preview-workspace-surface-contrast);
}

.pending-image-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.pending-image-copy {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.pending-image-copy strong {
  font-size: 12px;
  color: var(--character-studio-preview-workspace-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pending-image-copy span {
  font-size: 11px;
  color: var(--color-text-studio-muted);
}

.pending-remove {
  position: absolute;
  top: 8px;
  right: 10px;
  width: 20px;
  height: 20px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: var(--color-surface-danger-soft);
  color: var(--color-text-studio-danger);
  cursor: pointer;
}

.runtime-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  width: 100%;
  flex: 1 1 auto;
  min-height: 0;
}

.runtime-card {
  border-radius: 18px;
  padding: 16px;
  background: var(--character-studio-preview-workspace-surface-tint);
  border: 1px solid var(--character-studio-preview-workspace-border-default);
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.runtime-empty-panel {
  flex: 1 1 auto;
  align-items: center;
  justify-content: center;
}

.runtime-card pre,
.prompt-preview-card pre {
  margin: 10px 0 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  color: var(--color-text-studio-strong);
  max-height: 280px;
  overflow: auto;
  flex: 1 1 auto;
  min-height: 0;
}

.patch-summary {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 12px;
}

.patch-summary-section {
  border-radius: 16px;
  padding: 14px;
  background: var(--character-studio-preview-workspace-surface-soft);
  border: 1px solid var(--color-border-studio);
}

.patch-summary-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.patch-summary-head strong {
  color: var(--character-studio-preview-details-text-primary);
}

.patch-summary-head span {
  color: var(--color-text-studio-muted);
  font-size: 12px;
}

.patch-summary-list {
  margin: 10px 0 0;
  padding-left: 18px;
  color: var(--color-text-studio);
  font-size: 13px;
  line-height: 1.7;
}

.patch-raw-details {
  margin-top: 12px;
}

.patch-raw-details summary {
  cursor: pointer;
  color: var(--color-text-studio-muted);
  font-size: 12px;
}

.log-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 10px;
  overflow: auto;
  flex: 1 1 auto;
  min-height: 0;
}

.log-item {
  border-radius: 12px;
  padding: 10px 12px;
  background: var(--character-studio-preview-details-surface-base);
  color: var(--color-text-studio);
  font-size: 12px;
  line-height: 1.6;
}

.preview-frame {
  width: 100%;
  height: 260px;
  border: 1px solid var(--color-border-studio);
  border-radius: 16px;
  margin-top: 12px;
  background: var(--color-surface-base);
}

.empty-copy {
  color: var(--color-text-studio-subtle);
  font-size: 13px;
  line-height: 1.7;
}

.modal-copy p,
.modal-empty,
.modal-loading {
  color: var(--color-text-studio-muted);
  font-size: 13px;
  line-height: 1.7;
}

.greeting-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.greeting-card {
  border: 1px solid var(--color-border-studio);
  border-radius: 18px;
  padding: 16px;
  background: var(--character-studio-preview-details-surface-raised);
  text-align: left;
  cursor: pointer;
}

.greeting-card.active {
  border-color: var(--character-studio-preview-details-border-default);
  box-shadow: inset 0 0 0 1px var(--character-studio-preview-details-shadow-default);
  background: var(--character-studio-preview-details-surface-muted);
}

.greeting-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.greeting-badge {
  display: inline-flex;
  border-radius: 999px;
  padding: 4px 9px;
  background: var(--color-surface-studio-tint);
  color: var(--color-text-primary-strong);
  font-size: 11px;
}

.greeting-check {
  color: var(--color-text-primary-strong);
  font-weight: 700;
}

.greeting-card p {
  margin: 12px 0 0;
  color: var(--color-text-studio-strong);
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
}

.prompt-preview-body pre {
  margin: 0;
  padding: 16px;
  border-radius: 16px;
  background: var(--character-studio-preview-details-surface-subtle);
  border: 1px solid var(--color-border-studio);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 60vh;
  overflow: auto;
  color: var(--color-text-studio-strong);
  font-size: 12px;
}

.prompt-tools {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 12px;
}

.image-preview-body {
  display: flex;
  justify-content: center;
}

.image-preview-body img {
  max-width: 100%;
  max-height: 72vh;
  border-radius: 18px;
  object-fit: contain;
  background: var(--character-studio-preview-details-surface-hover);
}

.action-ghost,
.action-primary {
  border: none;
  border-radius: 14px;
  cursor: pointer;
}

.action-ghost {
  padding: 10px 14px;
  background: var(--color-surface-studio-muted);
  color: var(--color-text-studio);
}

.action-primary {
  padding: 11px 16px;
  background: linear-gradient(135deg, var(--character-studio-preview-details-surface-active), var(--character-studio-preview-details-surface-selected));
  color: var(--color-text-inverse);
  box-shadow: 0 12px 24px var(--character-studio-preview-details-shadow-raised);
}

.action-ghost:disabled,
.action-primary:disabled {
  opacity: 0.68;
  cursor: not-allowed;
  box-shadow: none;
}

.small {
  padding: 8px 12px;
  font-size: 12px;
}

.tiny {
  padding: 6px 10px;
  font-size: 12px;
}

@media (--breakpoint-studio-down) {
  .runtime-grid {
    grid-template-columns: 1fr;
  }

  .session-toolbar {
    align-items: stretch;
  }

  .session-triggers {
    flex-direction: column;
  }

  .greeting-grid {
    grid-template-columns: 1fr;
  }

  .toolbar-buttons {
    justify-content: flex-start;
  }
}

@media (--breakpoint-preview-down) {
  .tab-btn {
    flex: initial;
    justify-content: flex-start;
  }

  .workspace-tabs {
    overflow-x: auto;
  }

  .composer-main {
    grid-template-columns: 1fr;
  }

  .compact-actions {
    flex-direction: row;
    justify-content: flex-end;
  }

  .message-card {
    width: 100%;
  }
}
</style>
