<template>
  <AppShell class="studio-page" variant="studio" viewport-mode="immersive">
    <StudioTopbar
      :book-title="currentBookTitle"
      :document-title="store.currentDocument?.meta.title || ''"
      :document-origin="currentDocumentOrigin"
      :has-document="!!store.currentDocument"
      :busy="store.hasBusyAction"
      :busy-label="store.activeActionLabel"
      :save-pending="store.isSaving"
      :validate-pending="store.isValidating"
      @back="goBack"
      @save="saveNow"
      @validate="validate"
      @open-resource="store.resourcePanelOpen = true"
      @open-export="store.activeEditorTab = 'export'"
    />

    <ProductEmptyState
      v-if="!bookId"
      class="studio-page__missing-context-state"
      eyebrow="缺少上下文"
      icon-name="alert-triangle"
      title="未检测到书籍参数"
      description="请从漫画分析页进入角色工坊，或在 URL 中携带 `book` 参数。角色工坊需要当前书籍的分析上下文。"
    />

    <div v-else class="studio-page__workspace-root">
      <ProductStatusBanner
        v-if="store.errorMessage"
        class="studio-page__workspace-error-banner"
        tone="danger"
        aria-live="assertive"
      >
        {{ store.errorMessage }}
        <template #actions>
          <UiButton variant="secondary" size="sm" @click="store.clearErrorMessage()">知道了</UiButton>
        </template>
      </ProductStatusBanner>

      <ProductSplitWorkspace
        v-model:left-pane-width="leftPaneWidth"
        class="studio-page__workspace-shell"
        aria-label="角色工坊工作区"
        resizer-label="调整编辑区和预览区宽度"
        :min="PANE_WIDTH_MIN"
        :max="PANE_WIDTH_MAX"
        :step="PANE_WIDTH_STEP"
        left-scroll-test-id="editor-scroll"
        right-scroll-test-id="chat-scroll"
      >
        <template #left>
          <div class="studio-page__workspace-slot-content">
            <CharacterStudioEditor
              :document="store.currentDocument"
              :avatar-url="avatarUrl"
              :diagnostics="store.diagnostics"
              :pending-state="store.editorPendingState"
              :active-tab="store.activeEditorTab"
              :active-script-tab="store.activeScriptTab"
              @update:document="store.updateCurrentDocument($event)"
              @update:active-tab="store.activeEditorTab = $event"
              @update:active-script-tab="store.activeScriptTab = $event"
              @save="saveNow"
              @generate="generateSection"
              @validate="validate"
              @delete="deleteCurrent"
              @import-worldbook="importWorldbook"
              @download="download"
            />
          </div>
        </template>

        <template #right>
          <div class="studio-page__workspace-slot-content">
            <CharacterStudioPreview
              :book-id="props.bookId || ''"
              :document="store.currentDocument"
              :session="store.activeChatSession"
              :archived-sessions="store.archivedChatSessions"
              :available-greetings="store.availableChatGreetings"
              :prompt-preview="store.chatPromptPreview"
              :prompt-preview-error="store.chatPromptPreviewError"
              :active-tab="store.activeWorkspaceTab"
              :chat-loading="store.isChatLoading"
              :chat-streaming="store.isChatStreaming"
              :chat-abortable="Boolean(store.activeChatOperationId)"
              :chat-mutating="store.isChatMutating"
              :chat-summarizing="store.isChatSummarizing"
              :chat-exporting="store.isChatExporting"
              :chat-importing="store.isChatImporting"
              :chat-prompt-loading="store.isChatPromptLoading"
              :agent-busy="store.isAgentBusy"
              :agent-messages="store.agentMessages"
              :pending-patch="store.pendingAgentPatch"
              :can-undo-patch="store.canUndoPatch"
              :agent-html-preview="store.agentHtmlPreview"
              @abort-chat="abortChat"
              @update:active-tab="store.activeWorkspaceTab = $event"
              @send-chat="sendChat"
              @edit-message="editChatMessage"
              @delete-message="deleteChatMessage"
              @regenerate-message="regenerateChatMessage"
              @new-session="createChatSession"
              @switch-session="switchChatSession"
              @delete-session="deleteArchivedChatSession"
              @summarize-session="summarizeChatSession"
              @export-session="exportChatSession"
              @import-session="importChatSession"
              @load-prompt-preview="loadPromptPreviewFromChat"
              @send-agent="sendAgent"
              @apply-patch="store.applyPendingPatch()"
              @undo-patch="store.undoLastPatch()"
            />
          </div>
        </template>
      </ProductSplitWorkspace>
    </div>

    <template #overlay>
      <div
        v-if="store.resourcePanelOpen"
        class="studio-page__resource-overlay"
        data-testid="resource-overlay"
        @click.self="store.resourcePanelOpen = false"
      >
        <div class="studio-page__resource-dialog" data-testid="resource-dialog">
          <CharacterStudioSidebar
            :documents="store.filteredDocuments"
            :candidates="store.filteredCandidates"
            :search="store.selectedLibrarySearch"
            :current-document-id="store.currentDocument?.id || ''"
            :has-timeline="store.hasTimeline"
            :workspace-loading="store.isWorkspaceLoading"
            :creating-manual="store.isCreatingManual"
            :importing-file="store.isImportingFile"
            :opening-document-id="store.openingDocumentId"
            :creating-candidate-name="store.creatingCandidateName"
            @update:search="store.selectedLibrarySearch = $event"
            @open-document="openDocument"
            @create-manual="createManual"
            @create-from-candidate="createFromCandidate"
            @import-file="importFile"
          />
        </div>
      </div>
    </template>
  </AppShell>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import AppShell from '@/components/ui/AppShell.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductSplitWorkspace from '@/components/product/ProductSplitWorkspace.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { getCharacterStudioAvatarUrl } from '@/api/characterStudio'
import { useCharacterStudioStore } from '@/stores/characterStudioStore'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { confirmProductAction } from '@/composables/useProductConfirm'
import type { CharacterStudioChatSessionSummary } from '@/types/characterStudio'
import CharacterStudioSidebar from '@/components/insight/studio/CharacterStudioSidebar.vue'
import CharacterStudioEditor from '@/components/insight/studio/CharacterStudioEditor.vue'
import CharacterStudioPreview from '@/components/insight/studio/CharacterStudioPreview.vue'
import StudioTopbar from '@/components/insight/studio/StudioTopbar.vue'

const props = defineProps<{
  bookId?: string
  docId?: string
}>()

const router = useRouter()
const store = useCharacterStudioStore()
const bookshelfStore = useBookshelfStore()
const PANE_WIDTH_MIN = 35
const PANE_WIDTH_MAX = 70
const PANE_WIDTH_STEP = 2
const leftPaneWidth = ref(52)
let hydrateRequestId = 0

const currentBookTitle = computed(() => {
  if (!props.bookId) return ''
  const book = bookshelfStore.books.find(item => item.id === props.bookId)
  return book?.title || props.bookId
})

const currentDocumentOrigin = computed(() => {
  const origin = store.currentDocument?.origin.type
  if (origin === 'analysis') return '分析生成'
  if (origin === 'imported') return '外部导入'
  if (origin === 'manual') return '手工创建'
  return ''
})

const avatarUrl = computed(() => {
  if (!props.bookId || !store.currentDocument?.id || !store.currentDocument.avatarUrl) return ''
  return getCharacterStudioAvatarUrl(store.currentDocument.id)
})

async function runAction(action: () => Promise<void>) {
  try {
    await action()
    return true
  } catch {
    return false
  }
}

function isActiveHydration(requestId: number, bookId: string, docId?: string): boolean {
  return requestId === hydrateRequestId && props.bookId === bookId && props.docId === docId
}

async function hydrateWorkspace(nextBookId: string) {
  const requestId = ++hydrateRequestId
  const requestedDocId = props.docId
  try {
    if (!bookshelfStore.books.length) {
      await bookshelfStore.loadBooks()
      if (!isActiveHydration(requestId, nextBookId, requestedDocId)) return
    }
    if (!bookshelfStore.getBookById(nextBookId)) {
      await bookshelfStore.loadBookDetail(nextBookId)
      if (!isActiveHydration(requestId, nextBookId, requestedDocId)) return
    }
    await store.loadWorkspace(nextBookId)
    if (!isActiveHydration(requestId, nextBookId, requestedDocId)) return

    if (requestedDocId) {
      const openedRequested = await runAction(() => store.openDocument(requestedDocId))
      if (!isActiveHydration(requestId, nextBookId, requestedDocId)) return
      if (openedRequested) return
      if (store.documents.length === 0) {
        void router.replace({ name: 'character-studio', query: { book: nextBookId } })
        return
      }
    }
    if (store.documents.length > 0) {
      const fallbackDocId = store.documents[0]!.id
      const openedFallback = await runAction(() => store.openDocument(fallbackDocId))
      if (!isActiveHydration(requestId, nextBookId, requestedDocId)) return
      if (openedFallback) {
        void router.replace({ name: 'character-studio', query: { book: nextBookId, doc: fallbackDocId } })
      }
    }
  } catch {
    // 错误由 store 统一承载
  }
}

function goBack() {
  if (!props.bookId) {
    void router.push({ name: 'insight' })
    return
  }
  void router.push({ name: 'insight', query: { book: props.bookId } })
}

async function openDocument(docId: string) {
  const ok = await runAction(() => store.openDocument(docId))
  if (!ok) return
  store.resourcePanelOpen = false
  if (!props.bookId) return
  void router.replace({ name: 'character-studio', query: { book: props.bookId, doc: docId } })
}

async function createManual() {
  const ok = await runAction(() => store.createManualDocument())
  if (!ok) return
  store.resourcePanelOpen = false
  if (!props.bookId || !store.currentDocument) return
  void router.replace({ name: 'character-studio', query: { book: props.bookId, doc: store.currentDocument.id } })
}

async function createFromCandidate(candidateName: string) {
  const ok = await runAction(() => store.createDocumentFromCandidate(candidateName))
  if (!ok) return
  store.resourcePanelOpen = false
  if (!props.bookId || !store.currentDocument) return
  void router.replace({ name: 'character-studio', query: { book: props.bookId, doc: store.currentDocument.id } })
}

async function importFile(file: File) {
  const ok = await runAction(() => store.importFile(file))
  if (!ok) return
  store.resourcePanelOpen = false
  if (!props.bookId || !store.currentDocument) return
  void router.replace({ name: 'character-studio', query: { book: props.bookId, doc: store.currentDocument.id } })
}

async function importWorldbook(file: File) {
  await runAction(() => store.importWorldbook(file))
}

async function saveNow() {
  await runAction(() => store.persistCurrentDocument())
}

async function validate() {
  await runAction(() => store.validateCurrentDocument())
}

async function generateSection(section: string) {
  await runAction(() => store.generateSection(section))
}

async function deleteCurrent() {
  const ok = await runAction(() => store.deleteCurrentDocument())
  if (!ok) return
  if (!props.bookId) return
  void router.replace({ name: 'character-studio', query: { book: props.bookId } })
}

async function download(format: string) {
  await runAction(() => store.downloadCurrent(format))
}

async function sendAgent(message: string) {
  store.activeWorkspaceTab = 'assistant'
  await runAction(() => store.sendAgentMessage(message))
}

async function createChatSession(greetingId?: string) {
  store.activeWorkspaceTab = 'chat'
  await runAction(() => store.createChatSession(greetingId))
}

async function switchChatSession(sessionId: string) {
  store.activeWorkspaceTab = 'chat'
  await runAction(() => store.switchChatSession(sessionId))
}

async function abortChat() {
  store.activeWorkspaceTab = 'chat'
  await runAction(() => store.abortActiveChatOperation())
}

async function deleteArchivedChatSession(
  session: CharacterStudioChatSessionSummary,
) {
  const confirmed = await confirmProductAction({
    title: '永久删除归档会话',
    message: `确定永久删除“${session.title}”吗？聊天消息和附件引用将一并删除，无法恢复。`,
    confirmText: '永久删除',
    tone: 'danger',
  })
  if (!confirmed) return
  await runAction(() => store.deleteArchivedChatSession(
    session.session_id,
    session.revision,
  ))
}

async function sendChat(payload: { content: string; attachments: File[] }) {
  store.activeWorkspaceTab = 'chat'
  await runAction(() => store.sendChatMessage(payload.content, payload.attachments))
}

async function editChatMessage(payload: { messageId: string; content: string }) {
  store.activeWorkspaceTab = 'chat'
  await runAction(() => store.editChatMessage(payload.messageId, payload.content))
}

async function deleteChatMessage(messageId: string) {
  store.activeWorkspaceTab = 'chat'
  await runAction(() => store.deleteChatMessage(messageId))
}

async function regenerateChatMessage(messageId: string) {
  store.activeWorkspaceTab = 'chat'
  await runAction(() => store.regenerateChatMessage(messageId))
}

async function summarizeChatSession() {
  store.activeWorkspaceTab = 'runtime'
  await runAction(() => store.summarizeChatSession())
}

async function exportChatSession() {
  await runAction(() => store.exportChatSession())
}

async function importChatSession(file: File) {
  store.activeWorkspaceTab = 'chat'
  await runAction(() => store.importChatSession(file))
}

async function loadPromptPreviewFromChat() {
  await runAction(() => store.loadChatPromptPreview())
}

onMounted(async () => {
  if (props.bookId) {
    await hydrateWorkspace(props.bookId)
  }
})

onUnmounted(() => {
  hydrateRequestId += 1
})

watch(() => props.bookId, async nextBookId => {
  if (!nextBookId) {
    hydrateRequestId += 1
    return
  }
  await hydrateWorkspace(nextBookId)
})

watch(() => props.docId, async nextDocId => {
  if (!nextDocId || nextDocId === store.currentDocument?.id) return
  await runAction(() => store.openDocument(nextDocId))
})
</script>

<style scoped>
.studio-page {
  --studio-surface-soft: color-mix(in srgb, var(--color-surface-page) 92%, transparent);
  --studio-surface-tint: color-mix(in srgb, var(--color-action-primary) 10%, transparent);
  --studio-surface-tint-muted: color-mix(in srgb, var(--color-action-primary) 6%, transparent);
  --studio-surface-tint-strong: color-mix(in srgb, var(--color-action-primary) 14%, transparent);
  --studio-surface-muted: color-mix(in srgb, var(--color-text-heading) 7%, transparent);
  --studio-text-strong: var(--color-text-heading);
  --studio-text-default: var(--color-text-default);
  --studio-text-muted: var(--color-text-supporting);
  --studio-text-subtle: var(--color-text-subtle);
  --studio-border-default: color-mix(in srgb, var(--color-text-heading) 8%, transparent);
  --studio-border-strong: color-mix(in srgb, var(--color-text-heading) 12%, transparent);
  --studio-shadow-floating: var(--shadow-medium);
  --studio-form-control-border: 1px solid var(--studio-border-strong);
  --studio-form-control-background: var(--studio-surface-soft);
  --studio-form-control-color: var(--studio-text-strong);
  --studio-form-control-font-size: 13px;
  --ui-selector-control-background: var(--studio-form-control-background);
  --ui-selector-control-text: var(--studio-form-control-color);
  --ui-selector-control-border: var(--studio-border-strong);
  --ui-selector-control-font-size: var(--studio-form-control-font-size);
  --ui-select-padding: 10px 12px;
  --ui-select-radius: 14px;
  --ui-select-lg-padding: 12px 14px;
  --ui-select-lg-radius: 16px;
  --ui-input-studio-border: var(--studio-form-control-border);
  --ui-input-studio-background: var(--studio-form-control-background);
  --ui-input-studio-color: var(--studio-form-control-color);
  --ui-input-studio-font-size: var(--studio-form-control-font-size);
  --ui-textarea-studio-border: var(--studio-form-control-border);
  --ui-textarea-studio-background: var(--studio-form-control-background);
  --ui-textarea-studio-color: var(--studio-form-control-color);
  --ui-textarea-studio-font-size: var(--studio-form-control-font-size);
  --ui-textarea-studio-line-height: 1.7;
  --studio-view-accent-primary: color-mix(in srgb, var(--color-action-primary) 8%, transparent);
  --studio-view-accent-secondary: var(--color-surface-page);
  --studio-view-accent-muted: color-mix(in srgb, var(--color-surface-page) 55%, var(--color-surface-base));
  --studio-view-accent-strong: var(--color-surface-quiet);
  --studio-view-surface-raised: color-mix(in srgb, var(--color-overlay-backdrop-solid) 38%, transparent);
  --studio-view-text-primary: var(--color-text-heading);

  margin: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background:
    radial-gradient(circle at top right, var(--studio-view-accent-primary), transparent 24%),
    linear-gradient(180deg, var(--studio-view-accent-secondary) 0%, var(--studio-view-accent-muted) 48%, var(--studio-view-accent-strong) 100%);
  color: var(--studio-view-text-primary);
}

.studio-page__workspace-root {
  display: flex;
  flex: 1;
  min-height: 0;
  flex-direction: column;
}

.studio-page__workspace-shell {
  flex: 1;
  min-height: 0;
  padding: 18px 20px 20px;
}

.studio-page__workspace-slot-content {
  min-height: 100%;
}

.studio-page__workspace-slot-content > .studio-editor,
.studio-page__workspace-slot-content > .character-studio-preview {
  height: auto;
  min-height: 100%;
  overflow: visible;
}

.studio-page__resource-overlay {
  width: 100%;
  height: 100%;
  background: var(--studio-view-surface-raised);
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 82px 20px 20px;
  overflow-y: auto;
}

.studio-page__resource-dialog {
  width: min(1180px, 100%);
  height: calc(100dvh - 120px);
  max-height: calc(100dvh - 120px);
  min-height: 0;
  display: flex;
  overflow: hidden;
  flex-shrink: 0;
}

.studio-page__resource-dialog > * {
  flex: 1 1 auto;
  min-height: 0;
}

.studio-page__workspace-error-banner {
  margin: 14px 20px 0;
}

@media (--breakpoint-studio-down) {
  .studio-page__workspace-shell {
    padding: 14px;
  }
}
</style>
