<template>
  <aside class="runtime-shell" :class="{ collapsed }">
    <div class="runtime-rail" v-if="collapsed">
      <button class="rail-btn" data-testid="toggle-preview" title="展开运行时侧栏" @click="$emit('toggle-collapsed')">❯</button>
      <div class="rail-stats">
        <span>{{ session?.messages.length || 0 }}</span>
        <small>消息</small>
      </div>
      <div class="rail-stats">
        <span>{{ agentMessages.length }}</span>
        <small>Agent</small>
      </div>
    </div>

    <template v-else>
      <div class="runtime-header">
        <div>
          <div class="kicker">运行时侧栏</div>
          <h3>预览与助手</h3>
          <p>边编辑、边试聊、边用助手修卡，不让主编辑区被运行时信息打断。</p>
        </div>
        <button class="collapse-btn" data-testid="toggle-preview" @click="$emit('toggle-collapsed')">收起</button>
      </div>

      <section class="runtime-card">
        <div class="card-head">
          <div>
            <h4>聊天预览</h4>
            <p>直接在当前文档上试聊，观察问候语、世界书命中、正则日志和变量变化。</p>
          </div>
          <button class="ghost-btn" :disabled="!document" @click="$emit('reset-preview')">重置会话</button>
        </div>

        <div v-if="!document" class="empty-copy">选择角色文档后可试聊。</div>
        <template v-else>
          <div class="chat-shell">
            <div v-if="!session || session.messages.length === 0" class="empty-copy">还没有开始预览会话。</div>
            <div v-else class="message-list">
              <article v-for="(item, index) in session.messages" :key="`preview-${index}`" class="message-card" :class="item.role">
                <div class="message-head">
                  <span class="message-role">{{ item.role === 'assistant' ? (document.identity.name || '角色') : '你' }}</span>
                </div>
                <div class="message-body">{{ item.content }}</div>
              </article>
            </div>
          </div>

          <div class="composer">
            <textarea v-model="previewInput" rows="3" placeholder="输入一段用户消息，立即在当前角色文档上试聊。"></textarea>
            <button class="primary-btn" :disabled="previewing || !previewInput.trim()" @click="sendPreview">
              {{ previewing ? '响应生成中...' : '发送预览消息' }}
            </button>
          </div>

          <div class="meta-grid">
            <section class="meta-card">
              <h5>状态变量</h5>
              <pre>{{ formattedVariables }}</pre>
            </section>
            <section class="meta-card">
              <h5>命中日志</h5>
              <div v-if="session && session.log.length > 0" class="log-list">
                <div v-for="(item, index) in session.log" :key="`log-${index}`" class="log-item">
                  {{ summarizeLog(item) }}
                </div>
              </div>
              <div v-else class="empty-copy small">当前还没有命中日志。</div>
            </section>
          </div>
        </template>
      </section>

      <section class="runtime-card">
        <div class="card-head">
          <div>
            <h4>卡片助手</h4>
            <p>审查当前角色卡，并通过 `json:patch` 给出可应用的结构化建议。</p>
          </div>
          <div class="assistant-actions">
            <button class="ghost-btn" :disabled="!pendingPatch" @click="$emit('apply-patch')">应用 patch</button>
            <button class="ghost-btn" :disabled="!canUndoPatch" @click="$emit('undo-patch')">撤销 patch</button>
          </div>
        </div>

        <div class="agent-shell">
          <div v-if="agentMessages.length === 0" class="empty-copy">还没有与卡片助手对话。</div>
          <div v-else class="message-list">
            <article v-for="(item, index) in agentMessages" :key="`agent-${index}`" class="message-card" :class="item.role">
              <div class="message-head">
                <span class="message-role">{{ item.role === 'assistant' ? '卡片助手' : '你' }}</span>
              </div>
              <pre class="agent-text">{{ item.content }}</pre>
            </article>
          </div>
        </div>

        <div class="composer">
          <textarea v-model="agentInput" rows="3" placeholder="例如：请审查当前角色卡，并建议需要补充的世界书与问候语。"></textarea>
          <button class="primary-btn" :disabled="agentBusy || !agentInput.trim() || !document" @click="sendAgent">
            {{ agentBusy ? '助手处理中...' : '发送给助手' }}
          </button>
        </div>

        <div v-if="pendingPatch" class="meta-card">
          <h5>待应用 Patch</h5>
          <pre>{{ JSON.stringify(pendingPatch, null, 2) }}</pre>
        </div>

        <div v-if="agentHtmlPreview" class="meta-card">
          <h5>HTML 预览块</h5>
          <iframe class="preview-frame" :srcdoc="agentHtmlPreview" sandbox="allow-scripts"></iframe>
        </div>
      </section>
    </template>
  </aside>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import type { CharacterStudioDocument, PreviewSessionState } from '@/types/characterStudio'

const props = defineProps<{
  document: CharacterStudioDocument | null
  session: PreviewSessionState | null
  previewing: boolean
  agentBusy: boolean
  agentMessages: Array<{ role: 'user' | 'assistant'; content: string }>
  pendingPatch: Record<string, unknown> | null
  canUndoPatch: boolean
  agentHtmlPreview: string
  collapsed: boolean
}>()

const emit = defineEmits<{
  (e: 'send-preview', value: string): void
  (e: 'reset-preview'): void
  (e: 'send-agent', value: string): void
  (e: 'apply-patch'): void
  (e: 'undo-patch'): void
  (e: 'toggle-collapsed'): void
}>()

const previewInput = ref('')
const agentInput = ref('')

const formattedVariables = computed(() => JSON.stringify(props.session?.variables || {}, null, 2))

function sendPreview() {
  const value = previewInput.value.trim()
  if (!value) return
  emit('send-preview', value)
  previewInput.value = ''
}

function sendAgent() {
  const value = agentInput.value.trim()
  if (!value) return
  emit('send-agent', value)
  agentInput.value = ''
}

function summarizeLog(item: Record<string, unknown>) {
  if (item.type === 'regex') return `命中正则: ${String(item.scriptName || item.pattern || '未知脚本')}`
  if (item.type === 'lorebook') return `命中世界书: ${String(item.comment || '未命名条目')}`
  if (item.type === 'task') return `执行任务: ${String(item.name || '未命名任务')}`
  return JSON.stringify(item)
}
</script>

<style scoped>
.runtime-shell {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 0;
  width: 100%;
  align-self: stretch;
}

.runtime-shell.collapsed {
  align-items: stretch;
}

.runtime-rail {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  border-radius: 26px;
  padding: 18px 10px;
  background: rgba(252, 253, 255, 0.9);
  border: 1px solid rgba(28, 55, 94, 0.08);
  box-shadow: 0 24px 40px rgba(20, 46, 82, 0.08);
  min-height: 420px;
}

.rail-btn {
  width: 42px;
  height: 42px;
  border: none;
  border-radius: 14px;
  background: rgba(37, 99, 199, 0.1);
  color: #1f5fc3;
  cursor: pointer;
  font-size: 18px;
}

.rail-stats {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  color: #607794;
}

.rail-stats span {
  font-size: 20px;
  font-weight: 700;
  color: #14304c;
}

.rail-stats small {
  font-size: 10px;
}

.runtime-header,
.card-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.runtime-header {
  padding: 18px 18px 0;
}

.kicker {
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #6f84a2;
}

.runtime-header h3,
.card-head h4,
.meta-card h5 {
  margin: 8px 0 0;
  color: #102741;
}

.runtime-header p,
.card-head p {
  margin: 8px 0 0;
  color: #607794;
  font-size: 13px;
  line-height: 1.7;
}

.runtime-card {
  border-radius: 26px;
  padding: 18px;
  background: rgba(252, 253, 255, 0.9);
  border: 1px solid rgba(28, 55, 94, 0.08);
  box-shadow: 0 24px 40px rgba(20, 46, 82, 0.08);
}

.assistant-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.collapse-btn,
.ghost-btn,
.primary-btn {
  border: none;
  border-radius: 14px;
  cursor: pointer;
}

.collapse-btn,
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

.chat-shell,
.agent-shell {
  margin-top: 16px;
  max-height: 360px;
  overflow: auto;
  border-radius: 18px;
  padding: 14px;
  background: rgba(245, 249, 254, 0.92);
  border: 1px solid rgba(28, 55, 94, 0.08);
}

.message-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.message-card {
  border-radius: 16px;
  padding: 12px;
}

.message-card.user {
  background: rgba(20, 56, 106, 0.08);
}

.message-card.assistant {
  background: rgba(37, 99, 199, 0.1);
}

.message-head {
  display: flex;
  justify-content: space-between;
}

.message-role {
  font-size: 11px;
  color: #5f7591;
}

.message-body,
.agent-text {
  margin: 8px 0 0;
  color: #183351;
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
}

.composer {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 16px;
}

.composer textarea {
  border: 1px solid rgba(28, 55, 94, 0.12);
  background: rgba(245, 249, 254, 0.92);
  border-radius: 16px;
  padding: 12px 14px;
  color: #183351;
  font-size: 13px;
  resize: vertical;
  line-height: 1.7;
}

.meta-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.meta-card {
  margin-top: 16px;
  border-radius: 18px;
  padding: 14px;
  background: rgba(245, 249, 254, 0.92);
  border: 1px solid rgba(28, 55, 94, 0.08);
}

.meta-card pre {
  margin: 10px 0 0;
  color: #183351;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}

.log-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 10px;
}

.log-item {
  border-radius: 12px;
  padding: 10px;
  background: rgba(37, 99, 199, 0.08);
  color: #1f5fc3;
  font-size: 12px;
}

.preview-frame {
  width: 100%;
  min-height: 220px;
  border: 1px solid rgba(28, 55, 94, 0.1);
  border-radius: 14px;
  background: #fff;
}

.empty-copy {
  color: #6d839f;
  font-size: 13px;
  line-height: 1.6;
}

.small {
  margin-top: 10px;
}

@media (max-width: 1180px) {
  .meta-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 900px) {
  .runtime-header,
  .card-head {
    flex-direction: column;
  }
}
</style>
