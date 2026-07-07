<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
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
const sessionDropdownRef = ref<HTMLElement | null>(null)
const importInput = ref<InstanceType<typeof UiFileInput> | null>(null)
const SESSION_LIST_PANEL_ID = 'studio-session-list-panel'

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

function handleImportChange(files: File[]) {
  const file = files[0]
  if (!file) return
  emit('import-session', file)
  importInput.value?.clear()
}

function handleDocumentClick(event: MouseEvent) {
  if (!sessionListOpen.value) return
  if (sessionDropdownRef.value?.contains(event.target as Node)) return
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
    <div class="session-toolbar__triggers">
      <div ref="sessionDropdownRef" class="session-toolbar__trigger-stack session-toolbar__trigger-stack--wide">
        <UiButton
          variant="toolbar"
          data-testid="session-list-trigger"
          class="session-toolbar__trigger session-toolbar__trigger--inline"
          :disabled="chatMutating || chatStreaming"
          aria-haspopup="menu"
          :aria-controls="SESSION_LIST_PANEL_ID"
          :aria-expanded="sessionListOpen ? 'true' : 'false'"
          @click="toggleSessionList"
        >
          <div class="session-toolbar__trigger-copy session-toolbar__trigger-copy--inline">
            <span class="session-toolbar__trigger-tag">会话</span>
            <strong class="session-toolbar__trigger-title">{{ currentSessionLabel }}</strong>
            <span class="session-toolbar__trigger-meta">{{ currentSessionMeta }}</span>
          </div>
          <span class="session-toolbar__trigger-arrow">
            <UiIcon name="chevron-down" size="14" />
          </span>
        </UiButton>
        <div
          v-if="sessionListOpen"
          :id="SESSION_LIST_PANEL_ID"
          class="session-toolbar__session-list"
          role="menu"
          aria-label="聊天会话列表"
          @keydown.escape.stop.prevent="closeSessionList"
        >
          <UiButton
            variant="toolbar"
            class="session-toolbar__session-item session-toolbar__session-item--current"
            :class="{ 'session-toolbar__session-item--active': currentSessionId }"
            role="menuitem"
            aria-current="true"
            @click="closeSessionList"
          >
            <div class="session-toolbar__session-item-main">
              <strong class="session-toolbar__session-title">{{ currentSessionLabel }}</strong>
              <p class="session-toolbar__session-excerpt">{{ currentSessionExcerpt || '当前活跃会话' }}</p>
            </div>
            <div class="session-toolbar__session-item-meta">
              <span>{{ currentSessionMeta }}</span>
              <span class="session-toolbar__session-item-badge">当前</span>
            </div>
          </UiButton>
          <ProductStatusBanner
            v-if="archivedSessions.length === 0"
            icon-name="message"
            role="note"
            tone="neutral"
            title="暂无归档会话"
          >
            还没有归档会话。
          </ProductStatusBanner>
          <UiButton
            v-for="item in archivedSessions"
            :key="item.session_id"
            variant="toolbar"
            class="session-toolbar__session-item"
            role="menuitem"
            @click="chooseSession(item.session_id)"
          >
            <div class="session-toolbar__session-item-main">
              <strong class="session-toolbar__session-title">{{ item.title }}</strong>
              <p class="session-toolbar__session-excerpt">{{ item.last_message_excerpt || '暂无摘要' }}</p>
            </div>
            <div class="session-toolbar__session-item-meta">
              <span>{{ item.message_count }} 条</span>
              <span>{{ formatSessionTime(item.updated_at) }}</span>
            </div>
          </UiButton>
        </div>
      </div>

      <div class="session-toolbar__trigger-stack">
        <UiButton
          variant="toolbar"
          data-testid="greeting-picker-trigger"
          class="session-toolbar__trigger session-toolbar__trigger--inline"
          :disabled="!canUseGreeting || chatMutating || chatStreaming"
          @click="$emit('open-greeting-picker')"
        >
          <div class="session-toolbar__trigger-copy session-toolbar__trigger-copy--inline">
            <span class="session-toolbar__trigger-tag">开场白</span>
            <strong class="session-toolbar__trigger-title">{{ currentGreetingLabel }}</strong>
          </div>
          <span class="session-toolbar__trigger-arrow">
            <UiIcon name="chevron-down" size="14" />
          </span>
        </UiButton>
      </div>
    </div>
    <ProductActionRow class="session-toolbar__actions" aria-label="聊天会话操作" justify="start" variant="toolbar">
      <UiButton variant="secondary" :disabled="!hasDocument || chatMutating || chatStreaming" size="sm" @click="$emit('new-session')">
        新对话
      </UiButton>
      <UiButton
        variant="secondary"
        data-testid="prompt-preview-trigger"
        :disabled="!hasDocument || chatPromptLoading || chatStreaming"
        size="sm"
        @click="$emit('open-prompt-preview')"
      >
        {{ chatPromptLoading ? '加载中...' : '查看提示词' }}
      </UiButton>
      <UiButton variant="secondary" :disabled="!hasDocument || chatMutating || chatStreaming" size="sm" @click="$emit('open-greeting-picker')">
        重选开场白
      </UiButton>
      <UiButton variant="secondary" :disabled="!hasSession || chatSummarizing || chatStreaming" size="sm" @click="$emit('summarize-session')">
        {{ chatSummarizing ? '总结中...' : '手动总结' }}
      </UiButton>
      <UiButton variant="secondary" :disabled="!hasSession || chatExporting || chatStreaming" size="sm" @click="$emit('export-session')">
        {{ chatExporting ? '导出中...' : '导出聊天' }}
      </UiButton>
      <UiButton variant="secondary" :disabled="chatImporting || chatStreaming" size="sm" @click="pickImport">
        {{ chatImporting ? '导入中...' : '导入聊天' }}
      </UiButton>
    </ProductActionRow>
    <UiFileInput ref="importInput" hidden accept=".json" @files-change="handleImportChange" />
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

.session-toolbar__triggers {
  display: flex;
  flex: 1 1 440px;
  flex-wrap: wrap;
  gap: 10px;
  min-width: 0;
}

.session-toolbar__trigger-stack {
  position: relative;
  flex: 1 1 220px;
  min-width: 0;
}

.session-toolbar__trigger-stack--wide {
  flex: 1 1 260px;
}

.session-toolbar__trigger {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 12px 14px;
  border: 1px solid color-mix(in srgb, var(--color-border-default) 70%, transparent);
  border-radius: 16px;
  background: linear-gradient(180deg, color-mix(in srgb, var(--color-surface-card) 96%, transparent), var(--studio-surface-soft));
  box-shadow: inset 0 1px 0 color-mix(in srgb, var(--color-surface-card) 50%, transparent);
  color: var(--studio-text-strong);
  cursor: pointer;
}

.session-toolbar__trigger--inline {
  min-height: 46px;
  padding: 10px 14px;
}

.session-toolbar__trigger:disabled {
  cursor: not-allowed;
  opacity: 0.62;
}

.session-toolbar__trigger-copy {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
  text-align: left;
}

.session-toolbar__trigger-copy--inline {
  flex-direction: row;
  align-items: center;
  gap: 8px;
}

.session-toolbar__trigger-title {
  overflow: hidden;
  color: var(--studio-text-strong);
  font-size: 14px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-toolbar__trigger-tag,
.session-toolbar__trigger-meta {
  color: var(--color-text-supporting);
  font-size: 11px;
  white-space: nowrap;
}

.session-toolbar__trigger-tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 4px 8px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--color-action-brand) 10%, transparent);
}

.session-toolbar__trigger-meta {
  overflow: hidden;
  text-overflow: ellipsis;
}

.session-toolbar__trigger-arrow {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-supporting);
  flex-shrink: 0;
}

.session-toolbar__session-list {
  position: absolute;
  z-index: var(--z-local-overlay);
  top: calc(100% + 6px);
  left: 0;
  width: min(460px, 100%);
  max-width: 100%;
  max-height: 420px;
  overflow: auto;
  padding: 10px;
  border: 1px solid var(--studio-border-default);
  border-radius: 20px;
  background: color-mix(in srgb, var(--color-surface-card) 88%, transparent);
  box-shadow: 0 18px 38px var(--shadow-medium);
}

.session-toolbar__session-item {
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

.session-toolbar__session-item:hover,
.session-toolbar__session-item--active {
  background: color-mix(in srgb, var(--color-action-brand) 6%, transparent);
}

.session-toolbar__session-item--current {
  padding-bottom: 14px;
  margin-bottom: 6px;
  border-bottom: 1px solid var(--studio-border-default);
}

.session-toolbar__session-item-main {
  min-width: 0;
}

.session-toolbar__session-title {
  display: block;
  color: var(--studio-text-strong);
  font-size: 14px;
}

.session-toolbar__session-excerpt {
  margin: 6px 0 0;
  color: var(--studio-text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.session-toolbar__session-item-meta {
  display: flex;
  flex-direction: column;
  gap: 6px;
  align-items: flex-end;
  color: var(--color-text-muted);
  font-size: 11px;
}

.session-toolbar__session-item-badge {
  display: inline-flex;
  padding: 4px 8px;
  border-radius: 999px;
  background: var(--studio-surface-tint-muted);
  color: var(--color-text-link-strong);
}

.session-toolbar__actions {
  flex: 0 0 auto;
}

@media (--breakpoint-studio-down) {
  .session-toolbar {
    align-items: stretch;
    flex-direction: column;
  }

  .session-toolbar__actions {
    justify-content: flex-start;
  }
}
</style>
