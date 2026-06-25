<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import { formatSessionTime } from '../characterStudioPreviewHelpers'
import type { CharacterStudioChatSessionSummary } from '@/types/characterStudio'

defineProps<{
  archivedSessions: CharacterStudioChatSessionSummary[]
  canUseGreeting: boolean
  chatExporting: boolean
  chatImporting: boolean
  chatMutating: boolean
  chatPromptLoading: boolean
  chatStreaming: boolean
  chatSummarizing: boolean
  currentGreetingLabel: string
  currentSessionExcerpt: string
  currentSessionId: string
  currentSessionLabel: string
  currentSessionMeta: string
  hasDocument: boolean
  hasSession: boolean
}>()

const emit = defineEmits<{
  (event: 'choose-session', sessionId: string): void
  (event: 'export-session'): void
  (event: 'import-session', file: File): void
  (event: 'new-session'): void
  (event: 'open-greeting-picker'): void
  (event: 'open-prompt-preview'): void
  (event: 'summarize-session'): void
}>()

const sessionListOpen = ref(false)
const sessionListRef = ref<HTMLElement | null>(null)
const importInput = ref<HTMLInputElement | null>(null)

function toggleSessionList() {
  sessionListOpen.value = !sessionListOpen.value
}

function closeSessionList() {
  sessionListOpen.value = false
}

function chooseSession(sessionId: string) {
  emit('choose-session', sessionId)
  closeSessionList()
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
})
</script>

<template>
  <div class="session-toolbar" :class="{ 'session-toolbar--empty': !hasDocument }">
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
            :class="{ active: currentSessionId }"
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
            v-for="item in archivedSessions"
            :key="item.session_id"
            variant="toolbar"
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
          :disabled="!canUseGreeting || chatMutating || chatStreaming"
          @click="$emit('open-greeting-picker')"
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
      <UiButton variant="toolbar" class="action-ghost" :disabled="!hasDocument || chatMutating || chatStreaming" size="sm" @click="$emit('new-session')">
        新对话
      </UiButton>
      <UiButton
        variant="toolbar"
        data-testid="prompt-preview-trigger"
        class="action-ghost"
        :disabled="!hasDocument || chatPromptLoading || chatStreaming"
        size="sm"
        @click="$emit('open-prompt-preview')"
      >
        {{ chatPromptLoading ? '加载中...' : '查看提示词' }}
      </UiButton>
      <UiButton variant="toolbar" class="action-ghost" :disabled="!hasDocument || chatMutating || chatStreaming" size="sm" @click="$emit('open-greeting-picker')">
        重选开场白
      </UiButton>
      <UiButton variant="toolbar" class="action-ghost" :disabled="!hasSession || chatSummarizing || chatStreaming" size="sm" @click="$emit('summarize-session')">
        {{ chatSummarizing ? '总结中...' : '手动总结' }}
      </UiButton>
      <UiButton variant="toolbar" class="action-ghost" :disabled="!hasSession || chatExporting || chatStreaming" size="sm" @click="$emit('export-session')">
        {{ chatExporting ? '导出中...' : '导出聊天' }}
      </UiButton>
      <UiButton variant="toolbar" class="action-ghost" :disabled="chatImporting || chatStreaming" size="sm" @click="pickImport">
        {{ chatImporting ? '导入中...' : '导入聊天' }}
      </UiButton>
    </div>
    <UiFileInput ref="importInput" hidden accept=".json" @change="handleImportChange" />
  </div>
</template>

<style scoped>
.session-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px 12px;
  width: 100%;
  margin-bottom: 4px;
}

.session-triggers {
  display: flex;
  flex: 1 1 440px;
  flex-wrap: wrap;
  gap: 10px;
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
  padding: 12px 14px;
  border: 1px solid var(--character-studio-preview-trigger-border);
  border-radius: 16px;
  background: linear-gradient(180deg, var(--character-studio-preview-trigger-background), var(--studio-surface-soft));
  box-shadow: inset 0 1px 0 var(--character-studio-preview-trigger-highlight);
  color: var(--studio-text-strong);
  cursor: pointer;
}

.session-trigger-inline {
  min-height: 46px;
  padding: 10px 14px;
}

.session-trigger:disabled {
  cursor: not-allowed;
  opacity: 0.62;
}

.trigger-copy {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
  text-align: left;
}

.trigger-copy-inline {
  flex-direction: row;
  align-items: center;
  gap: 8px;
}

.trigger-copy strong {
  overflow: hidden;
  color: var(--character-studio-preview-session-title-text);
  font-size: 14px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.trigger-tag,
.trigger-meta {
  color: var(--character-studio-preview-supporting-text);
  font-size: 11px;
  white-space: nowrap;
}

.trigger-tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 4px 8px;
  border-radius: 999px;
  background: var(--character-studio-preview-active-tab-background);
}

.trigger-meta {
  overflow: hidden;
  text-overflow: ellipsis;
}

.trigger-arrow {
  color: var(--character-studio-preview-supporting-text);
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
  padding: 10px;
  border: 1px solid var(--studio-border-default);
  border-radius: 20px;
  background: var(--character-studio-preview-pending-attachment-background);
  box-shadow: 0 18px 38px var(--character-studio-preview-popover-shadow);
}

.session-list-item {
  width: 100%;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  padding: 12px 14px;
  border: none;
  border-radius: 16px;
  background: transparent;
  text-align: left;
  cursor: pointer;
}

.session-list-item:hover,
.session-list-item.active {
  background: var(--character-studio-preview-runtime-log-background);
}

.session-list-item.current {
  padding-bottom: 14px;
  margin-bottom: 6px;
  border-bottom: 1px solid var(--studio-border-default);
}

.session-list-empty {
  padding: 12px 14px;
  color: var(--studio-text-subtle);
  font-size: 13px;
}

.item-main {
  min-width: 0;
}

.item-main strong {
  display: block;
  color: var(--character-studio-preview-session-title-text);
  font-size: 14px;
}

.item-main p {
  margin: 6px 0 0;
  color: var(--studio-text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.item-meta {
  display: flex;
  flex-direction: column;
  gap: 6px;
  align-items: flex-end;
  color: var(--character-studio-preview-disabled-text);
  font-size: 11px;
}

.item-badge {
  display: inline-flex;
  padding: 4px 8px;
  border-radius: 999px;
  background: var(--studio-surface-tint-muted);
  color: var(--color-text-primary-strong);
}

.toolbar-buttons {
  display: flex;
  flex: 0 0 auto;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 12px;
}

.action-ghost {
  padding: 10px 14px;
  border: none;
  border-radius: 14px;
  background: var(--studio-surface-muted);
  color: var(--studio-text-default);
  cursor: pointer;
}

.action-ghost:disabled {
  cursor: not-allowed;
  box-shadow: none;
  opacity: 0.68;
}

.session-toolbar--empty .action-ghost {
  padding: 6px 12px;
  font-size: 12px;
  line-height: 1.6;
}

@media (--breakpoint-studio-down) {
  .session-toolbar {
    align-items: stretch;
    flex-direction: column;
  }

  .toolbar-buttons {
    justify-content: flex-start;
  }
}
</style>
