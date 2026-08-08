<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductMessageBubble from '@/components/product/ProductMessageBubble.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import StudioPreviewWorkspaceHeader from './StudioPreviewWorkspaceHeader.vue'
import StudioPreviewWorkspacePanel from './StudioPreviewWorkspacePanel.vue'
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
  (event: 'update:agentInput', value: string): void
}>()
</script>

<template>
  <StudioPreviewWorkspacePanel class="agent-workspace">
    <StudioPreviewWorkspaceHeader
      title="卡片助手"
      description="围绕角色卡本体给出结构化建议，可应用 patch 或撤销。"
    >
      <template #actions>
        <ProductActionRow appearance="accent" class="agent-workspace__actions" aria-label="卡片助手 patch 操作" justify="start" variant="toolbar">
          <UiButton
            variant="secondary"
            :disabled="!pendingPatch"
            size="sm"
            @click="$emit('applyPatch')"
          >
            应用 patch
          </UiButton>
          <UiButton
            variant="secondary"
            :disabled="!canUndoPatch"
            size="sm"
            @click="$emit('undoPatch')"
          >
            撤销 patch
          </UiButton>
        </ProductActionRow>
      </template>
    </StudioPreviewWorkspaceHeader>

    <div class="agent-workspace__main">
      <div class="agent-workspace__messages">
        <ProductEmptyState
          v-if="agentMessages.length === 0"
          icon-name="sparkles"
          role="note"
          size="compact"
          title="还没有与卡片助手对话"
        />
        <ProductMessageBubble
          v-for="(item, index) in agentMessages"
          :key="`agent-${index}`"
          class="agent-workspace__message"
          :role="item.role"
          :avatar-icon-name="item.role === 'assistant' ? 'sparkles' : 'users'"
          :avatar-label="item.role === 'assistant' ? '卡片助手' : '你'"
          :aria-label="`${item.role === 'assistant' ? '卡片助手' : '你'}的助手消息`"
          data-testid="studio-agent-message"
          :data-message-role="item.role"
        >
          <template #meta>
            <span class="agent-workspace__message-role">{{ item.role === 'assistant' ? '卡片助手' : '你' }}</span>
          </template>
          <pre class="agent-workspace__message-text">{{ item.content }}</pre>
        </ProductMessageBubble>
      </div>

      <div class="agent-workspace__composer">
        <div class="agent-workspace__composer-main">
          <UiTextarea
            :model-value="agentInput"
            class="agent-workspace__composer-input"
            variant="studio"
            rows="1"
            aria-label="卡片助手消息内容"
            placeholder="例如：请审查当前角色卡，并建议补充世界书与状态任务。"
            @update:model-value="$emit('update:agentInput', $event)"
          />
          <ProductActionRow appearance="accent" class="agent-workspace__composer-actions" aria-label="卡片助手消息操作" justify="start">
            <UiIconButton
              variant="primary"
              size="lg"
              data-testid="assistant-send-trigger"
              type="button"
              :label="agentBusy ? '助手处理中...' : '发送给助手'"
              :disabled="agentBusy || !agentInput.trim() || !document"
              @click="$emit('sendAgent')"
            >
              <UiIcon :name="agentBusy ? 'loading' : 'send'" size="18" />
            </UiIconButton>
          </ProductActionRow>
        </div>
      </div>
    </div>

    <div v-if="pendingPatch" class="agent-workspace__patch-card">
      <h4 class="agent-workspace__patch-card-title">待应用 Patch</h4>
      <div v-if="patchSummarySections.length > 0" class="agent-workspace__patch-summary">
        <section
          v-for="section in patchSummarySections"
          :key="section.key"
          class="agent-workspace__patch-summary-section"
        >
          <div class="agent-workspace__patch-summary-head">
            <strong class="agent-workspace__patch-summary-title">{{ section.title }}</strong>
            <span class="agent-workspace__patch-summary-count">{{ section.items.length }} 项</span>
          </div>
          <ul class="agent-workspace__patch-summary-list">
            <li v-for="(item, index) in section.items" :key="`${section.key}-${index}`">{{ item }}</li>
          </ul>
        </section>
      </div>
      <details class="agent-workspace__patch-raw-details">
        <summary class="agent-workspace__patch-raw-summary">查看原始 JSON</summary>
        <pre class="agent-workspace__patch-raw-json">{{ JSON.stringify(pendingPatch, null, 2) }}</pre>
      </details>
    </div>

    <div v-if="agentHtmlPreview" class="agent-workspace__html-preview-card">
      <h4 class="agent-workspace__html-preview-title">HTML 预览块</h4>
      <iframe class="agent-workspace__preview-frame" :srcdoc="agentHtmlPreview" sandbox="allow-scripts"></iframe>
    </div>
  </StudioPreviewWorkspacePanel>
</template>

<style scoped>
.agent-workspace__patch-card,
.agent-workspace__html-preview-card {
  width: 100%;
  padding: 14px;
  border: 1px solid var(--studio-border-default);
  border-radius: 24px;
  background: color-mix(in srgb, var(--color-surface-card) 92%, transparent);
  box-shadow: 0 24px 40px var(--studio-shadow-floating);
}

.agent-workspace {
  gap: 12px;
  min-height: 0;
}

.agent-workspace__composer-actions {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: flex-end;
  gap: 6px;
}

.agent-workspace__patch-card-title,
.agent-workspace__html-preview-title {
  margin: 8px 0 0;
  color: var(--color-text-heading);
}

.agent-workspace__main {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
}

.agent-workspace__messages {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
  padding: 12px;
  overflow: auto;
  border: 1px solid var(--studio-border-default);
  border-radius: 20px;
  background: linear-gradient(180deg, color-mix(in srgb, var(--color-surface-app) 95%, transparent), color-mix(in srgb, var(--color-surface-neutral-muted) 90%, transparent));
}

.agent-workspace__message-role {
  color: inherit;
  font-size: 11px;
  opacity: 0.72;
}

.agent-workspace__message-text {
  margin: 0;
  color: inherit;
  font-family: inherit;
  font-size: 13px;
  line-height: 1.7;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

.agent-workspace__composer {
  display: flex;
  flex-direction: column;
  flex: 0 0 auto;
  gap: 6px;
  margin-top: 0;
  padding: 10px 12px;
  border: 1px solid var(--studio-border-default);
  border-radius: 20px;
  background: color-mix(in srgb, var(--color-surface-app) 94%, transparent);
}

.agent-workspace__composer-main {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: stretch;
  gap: 10px;
}

.agent-workspace__composer-input {
  min-height: 64px;
  resize: vertical;
}

.agent-workspace__patch-summary {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 12px;
}

.agent-workspace__patch-summary-section {
  padding: 14px;
  border: 1px solid var(--studio-border-default);
  border-radius: 16px;
  background: color-mix(in srgb, var(--color-surface-app) 88%, transparent);
}

.agent-workspace__patch-summary-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.agent-workspace__patch-summary-title {
  color: var(--color-text-heading);
}

.agent-workspace__patch-summary-count {
  color: var(--studio-text-muted);
  font-size: 12px;
}

.agent-workspace__patch-summary-list {
  margin: 10px 0 0;
  padding-left: 18px;
  color: var(--studio-text-default);
  font-size: 13px;
  line-height: 1.7;
  overflow-wrap: anywhere;
}

.agent-workspace__patch-raw-details {
  margin-top: 12px;
}

.agent-workspace__patch-raw-summary {
  color: var(--studio-text-muted);
  font-size: 12px;
  cursor: pointer;
}

.agent-workspace__patch-raw-json {
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

.agent-workspace__preview-frame {
  width: 100%;
  height: 260px;
  margin-top: 12px;
  border: 1px solid var(--studio-border-default);
  border-radius: 16px;
  background: var(--color-surface-base);
}

</style>
