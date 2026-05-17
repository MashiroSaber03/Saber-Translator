<template>
  <div class="chat-shell">
    <div class="workspace-head">
      <div class="workspace-copy">
        <div class="kicker">聊天工作区</div>
        <h3>聊天 / 助手 / 运行日志</h3>
        <p>在同一个区域里完成继续聊天、卡片助手修卡和命中调试。</p>
      </div>
      <div class="head-actions">
        <button class="ghost-btn" :disabled="!document || chatMutating || chatStreaming" @click="$emit('new-session')">
          新对话
        </button>
        <button
          data-testid="prompt-preview-trigger"
          class="ghost-btn"
          :disabled="!document || chatPromptLoading || chatStreaming"
          @click="openPromptPreviewModal"
        >
          {{ chatPromptLoading ? '加载中...' : '查看提示词' }}
        </button>
      </div>
    </div>

    <div class="workspace-tabs" role="tablist">
      <button
        v-for="item in tabs"
        :key="item.value"
        class="tab-btn"
        :class="{ active: activeTab === item.value }"
        @click="$emit('update:activeTab', item.value)"
      >
        <span>{{ item.icon }}</span>
        <strong>{{ item.label }}</strong>
      </button>
    </div>

    <section v-if="activeTab === 'chat'" class="workspace-card chat-workspace">
      <div class="session-toolbar">
        <div class="session-triggers">
          <div class="trigger-stack">
            <span class="trigger-label">会话</span>
            <button
              data-testid="session-list-trigger"
              class="session-trigger"
              :disabled="chatMutating || chatStreaming"
              @click="toggleSessionList"
            >
              <div class="trigger-copy">
                <strong>{{ currentSessionLabel }}</strong>
                <span>{{ currentSessionMeta }}</span>
              </div>
              <span class="trigger-arrow">▾</span>
            </button>
            <div v-if="sessionListOpen" ref="sessionListRef" class="session-list-panel">
              <button
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
              </button>
              <div v-if="archivedSessions.length === 0" class="session-list-empty">还没有归档会话。</div>
              <button
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
              </button>
            </div>
          </div>

          <div class="trigger-stack">
            <span class="trigger-label">开场白</span>
            <button
              data-testid="greeting-picker-trigger"
              class="session-trigger"
              :disabled="availableGreetings.length === 0 || chatMutating || chatStreaming"
              @click="openGreetingPicker"
            >
              <div class="trigger-copy">
                <strong>{{ selectedGreetingLabel }}</strong>
                <span>{{ selectedGreetingHint }}</span>
              </div>
              <span class="trigger-arrow">▾</span>
            </button>
          </div>
        </div>
        <div class="toolbar-buttons">
          <button class="ghost-btn small" :disabled="!document || chatMutating || chatStreaming" @click="openGreetingPicker">
            重选开场白
          </button>
          <button class="ghost-btn small" :disabled="!session || chatSummarizing || chatStreaming" @click="$emit('summarize-session')">
            {{ chatSummarizing ? '总结中...' : '手动总结' }}
          </button>
          <button class="ghost-btn small" :disabled="!session || chatExporting || chatStreaming" @click="$emit('export-session')">
            {{ chatExporting ? '导出中...' : '导出聊天' }}
          </button>
          <button class="ghost-btn small" :disabled="chatImporting || chatStreaming" @click="pickImport">
            {{ chatImporting ? '导入中...' : '导入聊天' }}
          </button>
        </div>
        <input ref="importInput" hidden type="file" accept=".json" @change="handleImportChange">
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
                <button class="ghost-btn tiny" :disabled="chatStreaming || chatMutating" @click="startEdit(item)">编辑</button>
                <button class="ghost-btn tiny" :disabled="chatStreaming || chatMutating" @click="$emit('delete-message', item.message_id)">删除</button>
                <button class="ghost-btn tiny" :disabled="chatStreaming" @click="$emit('regenerate-message', item.message_id)">重生</button>
              </div>
            </div>

            <div v-if="editingMessageId === item.message_id" class="editor-row">
              <textarea v-model="editingContent" rows="4"></textarea>
              <div class="editor-actions">
                <button class="primary-btn tiny" :disabled="!editingContent.trim() || chatMutating" @click="commitEdit(item.message_id)">保存</button>
                <button class="ghost-btn tiny" @click="cancelEdit">取消</button>
              </div>
            </div>
            <div v-else class="message-body">{{ item.content }}</div>

            <div v-if="item.attachments.length > 0" class="attachment-grid">
              <button
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
              </button>
            </div>
          </article>
        </div>

        <div class="composer-card">
          <div v-if="pendingFiles.length > 0" class="pending-files">
            <button
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
            </button>
          </div>
          <textarea v-model="chatInput" rows="4" placeholder="输入消息，或添加图片后让角色结合画面继续聊天。"></textarea>
          <div class="composer-actions">
            <button class="ghost-btn" :disabled="chatStreaming" @click="pickAttachments">添加图片</button>
            <button class="primary-btn" :disabled="chatStreaming || (!chatInput.trim() && pendingFiles.length === 0)" @click="sendChat">
              {{ chatStreaming ? '回复生成中...' : '发送消息' }}
            </button>
          </div>
          <input ref="attachmentInput" hidden type="file" accept="image/*" multiple @change="handleAttachmentChange">
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
          <button class="ghost-btn small" :disabled="!pendingPatch" @click="$emit('apply-patch')">应用 patch</button>
          <button class="ghost-btn small" :disabled="!canUndoPatch" @click="$emit('undo-patch')">撤销 patch</button>
        </div>
      </div>

      <div class="messages-panel compact">
        <div v-if="agentMessages.length === 0" class="empty-copy">还没有与卡片助手对话。</div>
        <article v-for="(item, index) in agentMessages" :key="`agent-${index}`" class="message-card" :class="item.role">
          <div class="message-head">
            <span class="message-role">{{ item.role === 'assistant' ? '卡片助手' : '你' }}</span>
          </div>
          <pre class="agent-text">{{ item.content }}</pre>
        </article>
      </div>

      <div class="composer-card">
        <textarea v-model="agentInput" rows="3" placeholder="例如：请审查当前角色卡，并建议补充世界书与状态任务。"></textarea>
        <div class="composer-actions">
          <button class="primary-btn" :disabled="agentBusy || !agentInput.trim() || !document" @click="sendAgent">
            {{ agentBusy ? '助手处理中...' : '发送给助手' }}
          </button>
        </div>
      </div>

      <div v-if="pendingPatch" class="prompt-preview-card">
        <h4>待应用 Patch</h4>
        <pre>{{ JSON.stringify(pendingPatch, null, 2) }}</pre>
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
      <div v-else class="empty-copy">发送消息后，这里会显示最新一轮的运行结果。</div>
    </section>

    <BaseModal
      v-model="greetingPickerOpen"
      title="重选开场白"
      size="large"
      custom-class="studio-chat-modal"
    >
      <div class="modal-copy">
        <p>选择一条开场白后，将归档当前会话，并以该开场白重新开启一轮新对话。</p>
      </div>
      <div v-if="availableGreetings.length === 0" class="modal-empty">当前还没有可用开场白。</div>
      <div v-else class="greeting-grid">
        <button
          v-for="item in availableGreetings"
          :key="item.greeting_id"
          type="button"
          class="greeting-card"
          :class="{ active: selectedGreetingId === item.greeting_id }"
          @click="selectedGreetingId = item.greeting_id"
        >
          <div class="greeting-card-head">
            <span class="greeting-badge">{{ item.label }}</span>
            <span v-if="selectedGreetingId === item.greeting_id" class="greeting-check">✓</span>
          </div>
          <p>{{ item.content }}</p>
        </button>
      </div>
      <template #footer>
        <button class="ghost-btn" @click="greetingPickerOpen = false">取消</button>
        <button class="primary-btn" :disabled="!selectedGreetingId || chatMutating || chatStreaming" @click="confirmGreetingSelection">
          确认并重新开场
        </button>
      </template>
    </BaseModal>

    <BaseModal
      v-model="promptPreviewModalOpen"
      title="本轮提示词预览"
      size="large"
      custom-class="studio-chat-modal"
    >
      <div v-if="chatPromptLoading" class="modal-loading">提示词加载中...</div>
      <div v-else-if="promptPreviewError" class="modal-empty">{{ promptPreviewError }}</div>
      <div v-else-if="promptPreview.trim()" class="prompt-preview-body">
        <div class="prompt-tools">
          <button class="ghost-btn small" @click="copyPromptPreview">复制内容</button>
        </div>
        <pre>{{ promptPreview }}</pre>
      </div>
      <div v-else class="modal-empty" data-testid="prompt-preview-empty">
        请先发送至少一条消息后再查看本轮提示词。
      </div>
    </BaseModal>

    <BaseModal
      v-model="imagePreviewOpen"
      :title="imagePreviewAttachment?.filename || '图片预览'"
      size="large"
      custom-class="studio-chat-modal studio-image-modal"
    >
      <div v-if="imagePreviewAttachment" class="image-preview-body">
        <img :src="attachmentUrl(imagePreviewAttachment)" :alt="imagePreviewAttachment.filename">
      </div>
    </BaseModal>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { getCharacterStudioChatAttachmentUrl } from '@/api/characterStudio'
import BaseModal from '@/components/common/BaseModal.vue'
import type {
  CharacterStudioChatAttachment,
  CharacterStudioChatSession,
  CharacterStudioChatSessionSummary,
  CharacterStudioDocument,
  CharacterStudioGreetingOption,
} from '@/types/characterStudio'

const props = defineProps<{
  bookId: string
  document: CharacterStudioDocument | null
  session: CharacterStudioChatSession | null
  archivedSessions: CharacterStudioChatSessionSummary[]
  availableGreetings: CharacterStudioGreetingOption[]
  promptPreview: string
  promptPreviewError: string
  promptPreviewRequestKey: number
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
  pendingPatch: Record<string, unknown> | null
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

const tabs = [
  { value: 'chat', label: '聊天', icon: '💬' },
  { value: 'assistant', label: '卡片助手', icon: '🧠' },
  { value: 'runtime', label: '运行日志', icon: '📟' },
] as const

interface PendingAttachmentCard {
  id: string
  file: File
  previewUrl: string
}

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
const selectedGreetingLabel = computed(() => {
  const selected = props.availableGreetings.find(item => item.greeting_id === selectedGreetingId.value)
  return selected?.label || '选择开场白'
})
const selectedGreetingHint = computed(() => {
  const selected = props.availableGreetings.find(item => item.greeting_id === selectedGreetingId.value)
  return selected?.content.slice(0, 26) || '切换开场白并新建会话'
})

watch(() => props.session?.session_id, () => {
  selectedGreetingId.value = ''
  sessionListOpen.value = false
})

watch(() => props.promptPreviewRequestKey, value => {
  if (value > 0) {
    promptPreviewModalOpen.value = true
  }
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

function commitEdit(messageId: string) {
  if (!editingContent.value.trim()) return
  emit('edit-message', { messageId, content: editingContent.value.trim() })
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
  if (!props.availableGreetings.length) return
  selectedGreetingId.value = props.availableGreetings[0]?.greeting_id || ''
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

function attachmentTypeLabel(mimeType: string) {
  if (mimeType.startsWith('image/')) return '图片'
  return mimeType.split('/').pop() || '附件'
}

function formatSessionTime(value: string) {
  if (!value) return '未知时间'
  return value.slice(5, 16).replace('T', ' ')
}

function summarizeLog(item: Record<string, unknown>) {
  if (item.type === 'regex') return `命中正则: ${String(item.scriptName || item.pattern || '未知脚本')}`
  if (item.type === 'lorebook') return `命中世界书: ${String(item.comment || '未命名条目')}`
  if (item.type === 'task') return `执行任务: ${String(item.name || '未命名任务')}`
  return JSON.stringify(item)
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
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 0;
  height: 100%;
  width: 100% !important;
  max-width: none !important;
  padding: 0 !important;
}

.assistant-head,
.message-head,
.toolbar-buttons,
.head-actions,
.prompt-head,
.composer-actions,
.editor-actions {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  justify-content: space-between;
}

.workspace-head {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 16px;
  align-items: start;
}

.workspace-copy {
  min-width: 0;
}

.kicker {
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #6f84a2;
}

.workspace-head h3,
.assistant-head h4,
.prompt-preview-card h4,
.html-preview-card h4,
.runtime-card h5 {
  margin: 8px 0 0;
  color: #102741;
}

.workspace-head p,
.assistant-head p {
  margin: 8px 0 0;
  color: #607794;
  font-size: 13px;
  line-height: 1.7;
}

.head-actions {
  flex-wrap: wrap;
  justify-content: flex-end;
  align-self: start;
}

.workspace-tabs {
  display: flex;
  gap: 8px;
  padding: 8px;
  border-radius: 18px;
  background: rgba(16, 39, 65, 0.05);
  border: 1px solid rgba(28, 55, 94, 0.08);
  width: 100%;
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
  color: #55708f;
  cursor: pointer;
}

.tab-btn.active {
  background: linear-gradient(135deg, rgba(37, 99, 199, 0.14), rgba(77, 134, 238, 0.1));
  color: #16365b;
  box-shadow: inset 0 0 0 1px rgba(37, 99, 199, 0.16);
}

.workspace-card,
.prompt-preview-card,
.html-preview-card {
  border-radius: 24px;
  padding: 18px;
  background: rgba(252, 253, 255, 0.92);
  border: 1px solid rgba(28, 55, 94, 0.08);
  box-shadow: 0 24px 40px rgba(20, 46, 82, 0.08);
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
}

.assistant-workspace,
.runtime-workspace {
  min-height: 0;
}

.session-toolbar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  margin-bottom: 14px;
  width: 100%;
  align-items: start;
}

.session-triggers {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  min-width: 0;
}

.trigger-stack {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: #516882;
  font-size: 12px;
}

.trigger-stack {
  position: relative;
}

.trigger-label {
  color: #6f84a2;
  font-size: 12px;
}

.session-trigger {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  border: 1px solid rgba(28, 55, 94, 0.1);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(245, 249, 254, 0.92));
  border-radius: 16px;
  padding: 12px 14px;
  color: #183351;
  cursor: pointer;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.5);
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

.trigger-copy strong {
  font-size: 14px;
  color: #14304c;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.trigger-copy span {
  font-size: 12px;
  color: #607794;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.trigger-arrow {
  color: #5f7591;
  flex-shrink: 0;
}

.session-list-panel {
  position: absolute;
  z-index: 15;
  top: calc(100% + 6px);
  left: 0;
  width: min(460px, calc(100vw - 80px));
  max-height: 420px;
  overflow: auto;
  border-radius: 20px;
  padding: 10px;
  background: rgba(255, 255, 255, 0.98);
  border: 1px solid rgba(28, 55, 94, 0.08);
  box-shadow: 0 18px 38px rgba(20, 46, 82, 0.18);
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
  background: rgba(37, 99, 199, 0.08);
}

.session-list-item.current {
  border-bottom: 1px solid rgba(28, 55, 94, 0.08);
  margin-bottom: 6px;
  padding-bottom: 14px;
}

.session-list-empty {
  padding: 12px 14px;
  color: #6d839f;
  font-size: 13px;
}

.item-main {
  min-width: 0;
}

.item-main strong {
  display: block;
  color: #14304c;
  font-size: 14px;
}

.item-main p {
  margin: 6px 0 0;
  color: #607794;
  font-size: 12px;
  line-height: 1.5;
}

.item-meta {
  display: flex;
  flex-direction: column;
  gap: 6px;
  align-items: flex-end;
  color: #6f84a2;
  font-size: 11px;
}

.item-badge {
  display: inline-flex;
  border-radius: 999px;
  padding: 4px 8px;
  background: rgba(37, 99, 199, 0.12);
  color: #1f5fc3;
}

.composer-card textarea,
.editor-row textarea {
  width: 100%;
  border: 1px solid rgba(28, 55, 94, 0.12);
  background: rgba(245, 249, 254, 0.92);
  border-radius: 14px;
  padding: 10px 12px;
  color: #183351;
  font-size: 13px;
}

.toolbar-buttons {
  justify-content: flex-end;
  flex-wrap: wrap;
}

.messages-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1 1 auto;
  min-height: 260px;
  overflow: auto;
  padding-right: 4px;
  padding: 14px;
  border-radius: 20px;
  background: linear-gradient(180deg, rgba(244, 248, 255, 0.95), rgba(238, 244, 252, 0.9));
  border: 1px solid rgba(28, 55, 94, 0.08);
}

.messages-panel.compact {
  flex: 0 0 320px;
  min-height: 320px;
}

.message-card {
  border-radius: 18px;
  padding: 14px;
  border: 1px solid rgba(28, 55, 94, 0.08);
  background: rgba(247, 250, 254, 0.96);
  width: min(100%, 88%);
}

.message-card.user {
  margin-left: auto;
  background: rgba(20, 56, 106, 0.08);
}

.message-card.assistant {
  margin-right: auto;
  background: rgba(37, 99, 199, 0.1);
}

.message-role {
  font-size: 11px;
  color: #5f7591;
}

.message-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.message-body,
.agent-text {
  margin-top: 8px;
  color: #183351;
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
  border: none;
  text-align: left;
  cursor: pointer;
  border-radius: 14px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.88);
  border: 1px solid rgba(28, 55, 94, 0.08);
}

.attachment-frame {
  aspect-ratio: 1 / 1;
  overflow: hidden;
  background: linear-gradient(180deg, rgba(225, 235, 250, 0.72), rgba(241, 246, 255, 0.96));
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
  color: #14304c;
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.attachment-info span {
  font-size: 11px;
  color: #607794;
}

.composer-card {
  margin-top: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px;
  border-radius: 20px;
  background: rgba(244, 248, 255, 0.94);
  border: 1px solid rgba(28, 55, 94, 0.08);
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
  border: 1px solid rgba(28, 55, 94, 0.08);
  background: rgba(255, 255, 255, 0.94);
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
  background: rgba(37, 99, 199, 0.08);
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
  color: #14304c;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pending-image-copy span {
  font-size: 11px;
  color: #607794;
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
  background: rgba(217, 55, 55, 0.12);
  color: #b83535;
  cursor: pointer;
}

.runtime-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  width: 100%;
}

.runtime-card {
  border-radius: 18px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.86);
  border: 1px solid rgba(25, 55, 94, 0.08);
}

.runtime-card pre,
.prompt-preview-card pre {
  margin: 10px 0 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  color: #183351;
  max-height: 280px;
  overflow: auto;
}

.log-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 10px;
}

.log-item {
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(20, 56, 106, 0.06);
  color: #234977;
  font-size: 12px;
  line-height: 1.6;
}

.preview-frame {
  width: 100%;
  height: 260px;
  border: 1px solid rgba(28, 55, 94, 0.08);
  border-radius: 16px;
  margin-top: 12px;
  background: #fff;
}

.empty-copy {
  color: #6d839f;
  font-size: 13px;
  line-height: 1.7;
}

.modal-copy p,
.modal-empty,
.modal-loading {
  color: #607794;
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
  border: 1px solid rgba(28, 55, 94, 0.08);
  border-radius: 18px;
  padding: 16px;
  background: rgba(244, 248, 255, 0.84);
  text-align: left;
  cursor: pointer;
}

.greeting-card.active {
  border-color: rgba(37, 99, 199, 0.28);
  box-shadow: inset 0 0 0 1px rgba(37, 99, 199, 0.16);
  background: rgba(237, 244, 255, 0.96);
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
  background: rgba(37, 99, 199, 0.1);
  color: #1f5fc3;
  font-size: 11px;
}

.greeting-check {
  color: #1f5fc3;
  font-weight: 700;
}

.greeting-card p {
  margin: 12px 0 0;
  color: #183351;
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
}

.prompt-preview-body pre {
  margin: 0;
  padding: 16px;
  border-radius: 16px;
  background: rgba(244, 248, 255, 0.92);
  border: 1px solid rgba(28, 55, 94, 0.08);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 60vh;
  overflow: auto;
  color: #183351;
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
  background: rgba(244, 248, 255, 0.9);
}

.ghost-btn,
.primary-btn {
  border: none;
  border-radius: 14px;
  cursor: pointer;
}

.ghost-btn {
  padding: 10px 14px;
  background: rgba(20, 56, 106, 0.07);
  color: #234977;
}

.primary-btn {
  padding: 11px 16px;
  background: linear-gradient(135deg, #2563c7, #4d86ee);
  color: #fff;
  box-shadow: 0 12px 24px rgba(37, 99, 199, 0.18);
}

.ghost-btn:disabled,
.primary-btn:disabled {
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

@media (max-width: 1100px) {
  .workspace-head {
    grid-template-columns: 1fr;
  }

  .head-actions {
    justify-content: flex-start;
  }

  .runtime-grid {
    grid-template-columns: 1fr;
  }

  .session-toolbar {
    grid-template-columns: 1fr;
  }

  .session-triggers,
  .greeting-grid {
    grid-template-columns: 1fr;
  }

  .toolbar-buttons {
    justify-content: flex-start;
  }
}

@media (max-width: 760px) {
  .tab-btn {
    flex: initial;
    justify-content: flex-start;
  }

  .workspace-tabs {
    overflow-x: auto;
  }

  .message-card {
    width: 100%;
  }
}
</style>
