<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import type { CharacterStudioAgentPatchV2, CharacterStudioDocument } from '@/types/characterStudio'

type AgentMessage = { role: 'user' | 'assistant'; content: string }
type PatchSummarySection = { key: string; title: string; items: string[] }

defineProps<{
  agentBusy: boolean
  agentHtmlPreview: string
  agentInput: string
  agentMessages: AgentMessage[]
  canUndoPatch: boolean
  document: CharacterStudioDocument | null
  patchSummarySections: PatchSummarySection[]
  pendingPatch: CharacterStudioAgentPatchV2 | null
}>()

defineEmits<{
  (event: 'applyPatch'): void
  (event: 'sendAgent'): void
  (event: 'undoPatch'): void
  (event: 'update:agentInput', value: string | number | boolean): void
}>()
</script>

<template>
  <section class="workspace-card assistant-workspace">
    <div class="assistant-head">
      <div>
        <h4>卡片助手</h4>
        <p>围绕角色卡本体给出结构化建议，可应用 patch 或撤销。</p>
      </div>
      <div class="assistant-actions">
        <UiButton
          variant="toolbar"
          class="action-ghost"
          :disabled="!pendingPatch"
          size="sm"
          @click="$emit('applyPatch')"
        >
          应用 patch
        </UiButton>
        <UiButton
          variant="toolbar"
          class="action-ghost"
          :disabled="!canUndoPatch"
          size="sm"
          @click="$emit('undoPatch')"
        >
          撤销 patch
        </UiButton>
      </div>
    </div>

    <div class="assistant-main">
      <div class="messages-panel assistant-messages">
        <div v-if="agentMessages.length === 0" class="empty-copy">还没有与卡片助手对话。</div>
        <article
          v-for="(item, index) in agentMessages"
          :key="`agent-${index}`"
          class="message-card"
          :class="item.role"
        >
          <div class="message-head">
            <span class="message-role">{{ item.role === 'assistant' ? '卡片助手' : '你' }}</span>
          </div>
          <pre class="agent-text">{{ item.content }}</pre>
        </article>
      </div>

      <div class="composer-card assistant-composer">
        <div class="composer-main">
          <UiTextarea
            :model-value="agentInput"
            class="chat-composer-input"
            rows="1"
            placeholder="例如：请审查当前角色卡，并建议补充世界书与状态任务。"
            @update:model-value="$emit('update:agentInput', $event)"
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
              @click="$emit('sendAgent')"
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
</template>

<style scoped>
.workspace-card,
.prompt-preview-card,
.html-preview-card {
  width: 100%;
  padding: 14px;
  border: 1px solid var(--studio-border-default);
  border-radius: 24px;
  background: var(--character-studio-preview-shell-surface-raised);
  box-shadow: 0 24px 40px var(--studio-shadow-floating);
}

.workspace-card {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  min-height: 0;
}

.assistant-workspace {
  gap: 12px;
  min-height: 0;
}

.assistant-head,
.message-head,
.composer-actions {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.assistant-head h4,
.prompt-preview-card h4,
.html-preview-card h4 {
  margin: 8px 0 0;
  color: var(--character-studio-preview-shell-text-primary);
}

.assistant-head p {
  margin: 8px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.7;
}

.assistant-main {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
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
  background: linear-gradient(180deg, var(--character-studio-preview-workspace-surface-base), var(--character-studio-preview-workspace-surface-raised));
}

.assistant-messages {
  flex: 1 1 auto;
  min-height: 0;
}

.message-card {
  width: min(100%, 88%);
  padding: 14px;
  border: 1px solid var(--studio-border-default);
  border-radius: 18px;
  background: var(--character-studio-preview-workspace-surface-muted);
}

.message-card.user {
  margin-left: auto;
  background: var(--character-studio-preview-workspace-surface-subtle);
}

.message-card.assistant {
  margin-right: auto;
  background: var(--studio-surface-tint);
}

.message-role {
  color: var(--character-studio-preview-workspace-text-primary);
  font-size: 11px;
}

.agent-text {
  margin-top: 8px;
  color: var(--studio-text-strong);
  font-family: inherit;
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
}

.composer-card {
  --ui-textarea-border: 1px solid var(--studio-border-strong);
  --ui-textarea-background: var(--studio-surface-soft);
  --ui-textarea-radius: 14px;
  --ui-textarea-padding: 10px 12px;
  --ui-textarea-color: var(--studio-text-strong);
  --ui-textarea-font-size: 13px;

  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 2px;
  padding: 10px 12px;
  border: 1px solid var(--studio-border-default);
  border-radius: 20px;
  background: var(--character-studio-preview-workspace-surface-overlay);
}

.assistant-composer {
  flex: 0 0 auto;
  margin-top: 0;
}

.composer-main {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: stretch;
  gap: 10px;
}

.chat-composer-input {
  min-height: 64px;
  resize: vertical;
}

.compact-actions {
  align-items: stretch;
  justify-content: flex-end;
  flex-direction: column;
  gap: 6px;
}

.icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  min-width: 44px;
  height: 44px;
  padding: 0;
  font-size: 22px;
  line-height: 1;
}

.patch-summary {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 12px;
}

.patch-summary-section {
  padding: 14px;
  border: 1px solid var(--studio-border-default);
  border-radius: 16px;
  background: var(--character-studio-preview-workspace-surface-soft);
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
  color: var(--studio-text-muted);
  font-size: 12px;
}

.patch-summary-list {
  margin: 10px 0 0;
  padding-left: 18px;
  color: var(--studio-text-default);
  font-size: 13px;
  line-height: 1.7;
}

.patch-raw-details {
  margin-top: 12px;
}

.patch-raw-details summary {
  color: var(--studio-text-muted);
  font-size: 12px;
  cursor: pointer;
}

.prompt-preview-card pre {
  flex: 1 1 auto;
  max-height: 280px;
  min-height: 0;
  margin: 10px 0 0;
  overflow: auto;
  color: var(--studio-text-strong);
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}

.preview-frame {
  width: 100%;
  height: 260px;
  margin-top: 12px;
  border: 1px solid var(--studio-border-default);
  border-radius: 16px;
  background: var(--color-surface-base);
}

.empty-copy {
  color: var(--studio-text-subtle);
  font-size: 13px;
  line-height: 1.7;
}
</style>
