<template>
  <aside class="chat-shell">
    <div class="workspace-head">
      <div class="workspace-copy">
        <div class="kicker">聊天工作区</div>
        <h3>聊天 / 助手 / 运行日志</h3>
        <p>在同一个区域里完成继续聊天、卡片助手修卡和命中调试。</p>
      </div>
      <div class="head-actions">
        <button class="ghost-btn" :disabled="!document || chatMutating || chatStreaming" @click="$emit('new-session', selectedGreetingId || undefined)">
          新对话
        </button>
        <button class="ghost-btn" :disabled="!document || chatPromptLoading || chatStreaming" @click="togglePromptPreview">
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
        <label class="toolbar-field">
          <span>归档会话</span>
          <select :value="session?.session_id || ''" :disabled="archivedSessions.length === 0 || chatMutating || chatStreaming" @change="switchSession">
            <option :value="session?.session_id || ''">当前会话</option>
            <option v-for="item in archivedSessions" :key="item.session_id" :value="item.session_id">
              {{ item.title }} · {{ item.message_count }} 条
            </option>
          </select>
        </label>
        <label class="toolbar-field">
          <span>重选开场白</span>
          <select v-model="selectedGreetingId" :disabled="availableGreetings.length === 0 || chatMutating || chatStreaming">
            <option value="">保持当前主问候</option>
            <option v-for="item in availableGreetings" :key="item.greeting_id" :value="item.greeting_id">
              {{ item.label }}
            </option>
          </select>
        </label>
        <div class="toolbar-buttons">
          <button class="ghost-btn small" :disabled="!document || chatMutating || chatStreaming" @click="$emit('new-session', selectedGreetingId || undefined)">
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

      <div v-if="showPromptPreview && promptPreview" class="prompt-preview-card">
        <div class="prompt-head">
          <h4>本轮提示词预览</h4>
          <button class="ghost-btn small" @click="showPromptPreview = false">收起</button>
        </div>
        <pre>{{ promptPreview }}</pre>
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
              <figure v-for="attachment in item.attachments" :key="attachment.attachment_id" class="attachment-card">
                <img
                  v-if="attachment.mime_type.startsWith('image/')"
                  :src="attachmentUrl(attachment)"
                  :alt="attachment.filename"
                >
                <figcaption>{{ attachment.filename }}</figcaption>
              </figure>
            </div>
          </article>
        </div>

      <div class="composer-card">
        <div v-if="pendingFiles.length > 0" class="pending-files">
          <span v-for="(file, index) in pendingFiles" :key="`${file.name}-${index}`" class="file-pill">
            {{ file.name }}
            <button type="button" @click="removePendingFile(index)">×</button>
            </span>
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
  </aside>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { getCharacterStudioChatAttachmentUrl } from '@/api/characterStudio'
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

const chatInput = ref('')
const agentInput = ref('')
const pendingFiles = ref<File[]>([])
const selectedGreetingId = ref('')
const showPromptPreview = ref(false)
const editingMessageId = ref('')
const editingContent = ref('')
const attachmentInput = ref<HTMLInputElement | null>(null)
const importInput = ref<HTMLInputElement | null>(null)

const latestRuntimeMessage = computed(() => {
  const messages = props.session?.messages || []
  return [...messages].reverse().find(item => item.role === 'assistant' && item.runtime_log.length > 0) || null
})

watch(() => props.promptPreview, value => {
  if (value) {
    showPromptPreview.value = true
  }
})

watch(() => props.session?.session_id, () => {
  selectedGreetingId.value = ''
})

function pickAttachments() {
  attachmentInput.value?.click()
}

function handleAttachmentChange(event: Event) {
  const target = event.target as HTMLInputElement
  const files = Array.from(target.files || [])
  pendingFiles.value = [...pendingFiles.value, ...files]
  target.value = ''
}

function removePendingFile(index: number) {
  pendingFiles.value.splice(index, 1)
}

function sendChat() {
  const content = chatInput.value.trim()
  if (!content && pendingFiles.value.length === 0) return
  emit('send-chat', { content, attachments: [...pendingFiles.value] })
  chatInput.value = ''
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

function switchSession(event: Event) {
  const target = event.target as HTMLSelectElement
  if (!target.value || target.value === props.session?.session_id) return
  emit('switch-session', target.value)
}

function togglePromptPreview() {
  showPromptPreview.value = !showPromptPreview.value
  if (showPromptPreview.value) {
    emit('load-prompt-preview')
  }
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

function summarizeLog(item: Record<string, unknown>) {
  if (item.type === 'regex') return `命中正则: ${String(item.scriptName || item.pattern || '未知脚本')}`
  if (item.type === 'lorebook') return `命中世界书: ${String(item.comment || '未命名条目')}`
  if (item.type === 'task') return `执行任务: ${String(item.name || '未命名任务')}`
  return JSON.stringify(item)
}
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
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 14px;
  width: 100%;
}

.toolbar-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: #516882;
  font-size: 12px;
}

.toolbar-field select,
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
  grid-column: 1 / -1;
  justify-content: flex-start;
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
  border-radius: 14px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.88);
  border: 1px solid rgba(28, 55, 94, 0.08);
}

.attachment-card img {
  display: block;
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: cover;
}

.attachment-card figcaption {
  padding: 8px 10px;
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
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.file-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 6px 10px;
  background: rgba(37, 99, 199, 0.1);
  color: #1f5fc3;
  font-size: 12px;
}

.file-pill button {
  border: none;
  background: transparent;
  color: inherit;
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

  .toolbar-buttons {
    grid-column: auto;
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
