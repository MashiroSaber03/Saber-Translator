<template>
  <div class="studio-page">
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

    <div v-if="!bookId" class="empty-state">
      <div class="empty-badge">缺少上下文</div>
      <h2>未检测到书籍参数</h2>
      <p>请从漫画分析页进入角色工坊，或在 URL 中携带 `book` 参数。角色工坊仍然依赖当前书籍的分析上下文。</p>
    </div>

    <div v-else class="workspace-root">
      <div v-if="store.errorMessage" class="workspace-error">
        <span>⚠ {{ store.errorMessage }}</span>
        <button class="error-dismiss" @click="store.clearErrorMessage()">知道了</button>
      </div>

      <div v-if="store.resourcePanelOpen" class="resource-overlay" @click.self="store.resourcePanelOpen = false">
        <div class="resource-dialog">
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

      <div class="workspace-shell" :style="workspaceStyle">
        <section class="editor-pane">
          <div class="column-scroll" data-testid="editor-scroll">
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
        </section>

        <div class="pane-resizer" @mousedown="startResize"></div>

        <section class="chat-pane">
          <div class="column-scroll" data-testid="chat-scroll">
            <CharacterStudioPreview
              :book-id="props.bookId || ''"
              :document="store.currentDocument"
              :session="store.activeChatSession"
              :archived-sessions="store.archivedChatSessions"
              :prompt-preview="store.chatPromptPreview"
              :prompt-preview-error="store.chatPromptPreviewError"
              :active-tab="store.activeWorkspaceTab"
              :chat-loading="store.isChatLoading"
              :chat-streaming="store.isChatStreaming"
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
              @update:active-tab="store.activeWorkspaceTab = $event"
              @send-chat="sendChat"
              @edit-message="editChatMessage"
              @delete-message="deleteChatMessage"
              @regenerate-message="regenerateChatMessage"
              @new-session="createChatSession"
              @switch-session="switchChatSession"
              @summarize-session="summarizeChatSession"
              @export-session="exportChatSession"
              @import-session="importChatSession"
              @load-prompt-preview="loadPromptPreviewFromChat"
              @send-agent="sendAgent"
              @apply-patch="store.applyPendingPatch()"
              @undo-patch="store.undoLastPatch()"
            />
          </div>
        </section>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { getCharacterStudioAvatarUrl } from '@/api/characterStudio'
import { useCharacterStudioStore } from '@/stores/characterStudioStore'
import { useBookshelfStore } from '@/stores/bookshelfStore'
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
const leftPaneWidth = ref(52)
const resizing = ref(false)

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
  if (!props.bookId || !store.currentDocument?.id || !store.currentDocument.avatar.asset_path) return ''
  return getCharacterStudioAvatarUrl(props.bookId, store.currentDocument.id)
})

const workspaceStyle = computed(() => ({
  gridTemplateColumns: `${leftPaneWidth.value}fr 8px ${100 - leftPaneWidth.value}fr`,
}))

function handleMouseMove(event: MouseEvent) {
  if (!resizing.value) return
  const width = window.innerWidth || 1
  const next = Math.min(70, Math.max(35, (event.clientX / width) * 100))
  leftPaneWidth.value = next
}

function handleMouseUp() {
  resizing.value = false
  document.body.classList.remove('studio-resizing')
}

function startResize() {
  resizing.value = true
  document.body.classList.add('studio-resizing')
}

async function runAction(action: () => Promise<void>) {
  try {
    await action()
    return true
  } catch {
    return false
  }
}

async function hydrateWorkspace(nextBookId: string) {
  try {
    if (!bookshelfStore.books.length) {
      await bookshelfStore.fetchBooks()
    }
    await store.loadWorkspace(nextBookId)
    if (props.docId) {
      const openedRequested = await runAction(() => store.openDocument(props.docId!))
      if (openedRequested) return
      if (store.documents.length === 0) {
        void router.replace({ name: 'character-studio', query: { book: nextBookId } })
        return
      }
    }
    if (store.documents.length > 0) {
      const fallbackDocId = store.documents[0]!.id
      const openedFallback = await runAction(() => store.openDocument(fallbackDocId))
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

async function summarizeChatSession(cutoffMessageId?: string) {
  store.activeWorkspaceTab = 'runtime'
  await runAction(() => store.summarizeChatSession(cutoffMessageId))
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
  window.addEventListener('mousemove', handleMouseMove)
  window.addEventListener('mouseup', handleMouseUp)
  if (props.bookId) {
    await hydrateWorkspace(props.bookId)
  }
})

onUnmounted(() => {
  window.removeEventListener('mousemove', handleMouseMove)
  window.removeEventListener('mouseup', handleMouseUp)
  document.body.classList.remove('studio-resizing')
})

watch(() => props.bookId, async nextBookId => {
  if (!nextBookId) return
  await hydrateWorkspace(nextBookId)
})

watch(() => props.docId, async nextDocId => {
  if (!nextDocId || nextDocId === store.currentDocument?.id) return
  await runAction(() => store.openDocument(nextDocId))
})
</script>

<style scoped>
.studio-page {
  height: 100vh;
  margin: 0 -20px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background:
    radial-gradient(circle at top right, rgba(86, 138, 225, 0.08), transparent 24%),
    linear-gradient(180deg, #f4f7fb 0%, #eef3f9 48%, #f6f8fb 100%);
  color: #122b47;
}

.workspace-root {
  display: flex;
  flex: 1;
  min-height: 0;
  flex-direction: column;
}

.workspace-shell {
  display: grid;
  flex: 1;
  min-height: 0;
  padding: 18px 20px 20px;
}

.editor-pane,
.chat-pane {
  min-width: 0;
  min-height: 0;
}

.chat-pane {
  width: auto !important;
  padding: 0 !important;
}

.column-scroll {
  height: 100%;
  min-height: 0;
}

.pane-resizer {
  width: 8px;
  cursor: col-resize;
  border-radius: 999px;
  background: linear-gradient(180deg, rgba(37, 99, 199, 0.1), rgba(37, 99, 199, 0.22));
}

.resource-overlay {
  position: fixed;
  inset: 0;
  z-index: 60;
  background: rgba(9, 25, 49, 0.38);
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 82px 20px 20px;
}

.resource-dialog {
  width: min(1180px, 100%);
  max-height: calc(100vh - 120px);
  overflow: hidden;
}

.workspace-error {
  margin: 14px 20px 0;
  border-radius: 16px;
  padding: 12px 16px;
  background: rgba(255, 244, 244, 0.92);
  border: 1px solid rgba(217, 55, 55, 0.12);
  color: #b83535;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.error-dismiss {
  border: none;
  border-radius: 12px;
  padding: 8px 12px;
  cursor: pointer;
  background: rgba(217, 55, 55, 0.12);
  color: inherit;
}

.empty-state {
  margin: auto;
  max-width: 560px;
  text-align: center;
  padding: 48px 32px;
  border-radius: 28px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(28, 55, 94, 0.08);
  box-shadow: 0 26px 42px rgba(20, 46, 82, 0.08);
}

.empty-badge {
  display: inline-flex;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(37, 99, 199, 0.12);
  color: #1f5fc3;
  font-size: 12px;
}

.empty-state h2 {
  margin: 16px 0 0;
  font-size: 30px;
}

.empty-state p {
  margin: 12px 0 0;
  color: #607794;
  line-height: 1.7;
}

:global(body.studio-resizing) {
  cursor: col-resize;
  user-select: none;
}

@media (max-width: 1100px) {
  .workspace-shell {
    grid-template-columns: 1fr !important;
    gap: 16px;
  }

  .pane-resizer {
    display: none;
  }
}
</style>
