<template>
  <div class="studio-page">
    <StudioTopbar
      :subtitle="topbarSubtitle"
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
      @open-export="store.activeEditorTab = 'export'"
      @toggle-left-drawer="store.leftDrawerOpen = !store.leftDrawerOpen"
      @toggle-right-drawer="store.rightDrawerOpen = !store.rightDrawerOpen"
    />

    <div v-if="!bookId" class="empty-state">
      <div class="empty-badge">缺少上下文</div>
      <h2>未检测到书籍参数</h2>
      <p>请从漫画分析页进入角色工坊，或在 URL 中携带 `book` 参数。角色工坊仍然依赖当前书籍的分析上下文。</p>
    </div>

    <div v-else class="workspace-shell">
      <div v-if="store.errorMessage" class="workspace-error">
        <span>⚠ {{ store.errorMessage }}</span>
        <button class="error-dismiss" @click="store.clearErrorMessage()">知道了</button>
      </div>
      <div v-if="store.leftDrawerOpen" class="drawer-mask left-mask" @click="store.leftDrawerOpen = false"></div>
      <div v-if="store.rightDrawerOpen" class="drawer-mask right-mask" @click="store.rightDrawerOpen = false"></div>

      <aside class="left-column" :class="{ 'drawer-open': store.leftDrawerOpen }">
        <div class="column-scroll" data-testid="left-scroll">
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
      </aside>

      <section class="editor-column">
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

      <aside class="right-column" :class="{ 'drawer-open': store.rightDrawerOpen }">
        <div class="column-scroll" data-testid="right-scroll">
          <CharacterStudioPreview
            :document="store.currentDocument"
            :session="store.previewSession"
            :previewing="store.isPreviewing"
            :agent-busy="store.isAgentBusy"
            :resetting-preview="store.isResettingPreview"
            :agent-messages="store.agentMessages"
            :pending-patch="store.pendingAgentPatch"
            :can-undo-patch="store.canUndoPatch"
            :agent-html-preview="store.agentHtmlPreview"
            @send-preview="sendPreview"
            @reset-preview="resetPreview"
            @send-agent="sendAgent"
            @apply-patch="store.applyPendingPatch()"
            @undo-patch="store.undoLastPatch()"
          />
        </div>
      </aside>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, watch } from 'vue'
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

const topbarSubtitle = computed(() => {
  if (!store.currentDocument) {
    return store.hasTimeline
      ? '选择一个角色文档，或从分析候选创建新角色。'
      : '当前书还没有增强时间线，但仍可空白新建或导入角色卡。'
  }
  return '编辑区优先，运行时预览收纳在右侧侧栏，适合长时间编卡。'
})

const avatarUrl = computed(() => {
  if (!props.bookId || !store.currentDocument?.id || !store.currentDocument.avatar.asset_path) return ''
  return getCharacterStudioAvatarUrl(props.bookId, store.currentDocument.id)
})

function closeDrawers() {
  store.leftDrawerOpen = false
  store.rightDrawerOpen = false
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
      await store.openDocument(props.docId)
    } else if (store.documents.length > 0) {
      await store.openDocument(store.documents[0]!.id)
    }
  } catch {
    // 错误已在 store 中记录并展示到页面
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
  closeDrawers()
  if (!props.bookId) return
  void router.replace({ name: 'character-studio', query: { book: props.bookId, doc: docId } })
}

async function createManual() {
  const ok = await runAction(() => store.createManualDocument())
  if (!ok) return
  closeDrawers()
  if (!props.bookId || !store.currentDocument) return
  void router.replace({ name: 'character-studio', query: { book: props.bookId, doc: store.currentDocument.id } })
}

async function createFromCandidate(candidateName: string) {
  const ok = await runAction(() => store.createDocumentFromCandidate(candidateName))
  if (!ok) return
  closeDrawers()
  if (!props.bookId || !store.currentDocument) return
  void router.replace({ name: 'character-studio', query: { book: props.bookId, doc: store.currentDocument.id } })
}

async function importFile(file: File) {
  const ok = await runAction(() => store.importFile(file))
  if (!ok) return
  closeDrawers()
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

async function sendPreview(message: string) {
  await runAction(() => store.sendPreviewMessage(message))
}

async function resetPreview() {
  await runAction(() => store.resetPreview())
}

async function sendAgent(message: string) {
  await runAction(() => store.sendAgentMessage(message))
}

async function download(format: string) {
  await runAction(() => store.downloadCurrent(format))
}

onMounted(async () => {
  const handleResize = () => {
    if (window.innerWidth > 900) {
      store.leftDrawerOpen = false
      store.rightDrawerOpen = false
    }
  }
  window.addEventListener('resize', handleResize)
  ;(window as Window & { __characterStudioResizeHandler__?: () => void }).__characterStudioResizeHandler__ = handleResize
  if (props.bookId) {
    await hydrateWorkspace(props.bookId)
  }
})

onUnmounted(() => {
  const handler = (window as Window & { __characterStudioResizeHandler__?: () => void }).__characterStudioResizeHandler__
  if (handler) {
    window.removeEventListener('resize', handler)
    delete (window as Window & { __characterStudioResizeHandler__?: () => void }).__characterStudioResizeHandler__
  }
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
  --studio-scrollbar-size: 10px;
  --studio-scrollbar-thumb: rgba(122, 148, 186, 0.42);
  --studio-scrollbar-thumb-hover: rgba(88, 118, 164, 0.72);
  --studio-scrollbar-track: rgba(213, 223, 239, 0.32);
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

.workspace-shell {
  display: grid;
  grid-template-columns: 304px minmax(0, 1fr) 392px;
  gap: 20px;
  padding: 20px 24px 28px;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.workspace-error {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding: 14px 16px;
  border-radius: 18px;
  background: rgba(255, 244, 244, 0.92);
  border: 1px solid rgba(217, 55, 55, 0.18);
  color: #9d2f2f;
}

.error-dismiss {
  border: none;
  border-radius: 12px;
  padding: 8px 12px;
  background: rgba(217, 55, 55, 0.12);
  color: #9d2f2f;
  cursor: pointer;
}

.left-column,
.editor-column,
.right-column {
  min-width: 0;
  width: 100%;
  min-height: 0;
  height: 100%;
  justify-self: stretch;
  overflow: hidden;
}

.column-scroll {
  height: 100%;
  min-height: 0;
}

.column-scroll,
.studio-page :deep(.studio-editor),
.studio-page :deep(.runtime-shell),
.studio-page :deep(.sidebar-content),
.studio-page :deep(.chat-shell),
.studio-page :deep(.agent-shell) {
  scrollbar-width: thin;
  scrollbar-color: var(--studio-scrollbar-thumb) transparent;
}

.column-scroll::-webkit-scrollbar,
.studio-page :deep(.studio-editor::-webkit-scrollbar),
.studio-page :deep(.runtime-shell::-webkit-scrollbar),
.studio-page :deep(.sidebar-content::-webkit-scrollbar),
.studio-page :deep(.chat-shell::-webkit-scrollbar),
.studio-page :deep(.agent-shell::-webkit-scrollbar) {
  width: var(--studio-scrollbar-size);
  height: var(--studio-scrollbar-size);
}

.column-scroll::-webkit-scrollbar-track,
.studio-page :deep(.studio-editor::-webkit-scrollbar-track),
.studio-page :deep(.runtime-shell::-webkit-scrollbar-track),
.studio-page :deep(.sidebar-content::-webkit-scrollbar-track),
.studio-page :deep(.chat-shell::-webkit-scrollbar-track),
.studio-page :deep(.agent-shell::-webkit-scrollbar-track) {
  background: transparent;
}

.column-scroll::-webkit-scrollbar-thumb,
.studio-page :deep(.studio-editor::-webkit-scrollbar-thumb),
.studio-page :deep(.runtime-shell::-webkit-scrollbar-thumb),
.studio-page :deep(.sidebar-content::-webkit-scrollbar-thumb),
.studio-page :deep(.chat-shell::-webkit-scrollbar-thumb),
.studio-page :deep(.agent-shell::-webkit-scrollbar-thumb) {
  border-radius: 999px;
  background: linear-gradient(180deg, rgba(147, 170, 204, 0.5), rgba(112, 138, 180, 0.44));
  border: 2px solid transparent;
  background-clip: padding-box;
}

.column-scroll:hover::-webkit-scrollbar-thumb,
.studio-page :deep(.studio-editor:hover::-webkit-scrollbar-thumb),
.studio-page :deep(.runtime-shell:hover::-webkit-scrollbar-thumb),
.studio-page :deep(.sidebar-content:hover::-webkit-scrollbar-thumb),
.studio-page :deep(.chat-shell:hover::-webkit-scrollbar-thumb),
.studio-page :deep(.agent-shell:hover::-webkit-scrollbar-thumb) {
  background: linear-gradient(180deg, rgba(118, 146, 188, 0.78), rgba(83, 112, 159, 0.74));
  background-clip: padding-box;
}

.column-scroll::-webkit-scrollbar-corner,
.studio-page :deep(.studio-editor::-webkit-scrollbar-corner),
.studio-page :deep(.runtime-shell::-webkit-scrollbar-corner),
.studio-page :deep(.sidebar-content::-webkit-scrollbar-corner),
.studio-page :deep(.chat-shell::-webkit-scrollbar-corner),
.studio-page :deep(.agent-shell::-webkit-scrollbar-corner) {
  background: transparent;
}

.right-column {
  transition: width 0.24s ease, transform 0.24s ease;
}

.right-column.collapsed {
  width: 88px;
}

.empty-state {
  max-width: 560px;
  margin: 80px auto;
  padding: 36px;
  border-radius: 28px;
  background: rgba(255, 255, 255, 0.88);
  box-shadow: 0 26px 42px rgba(20, 46, 82, 0.08);
}

.empty-badge {
  display: inline-flex;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(37, 99, 199, 0.1);
  color: #1f5fc3;
  font-size: 12px;
  font-weight: 600;
}

.empty-state h2 {
  margin: 16px 0 0;
  font-size: 28px;
}

.empty-state p {
  margin: 12px 0 0;
  color: #607794;
  line-height: 1.8;
}

.drawer-mask {
  position: fixed;
  inset: 0;
  background: rgba(16, 34, 58, 0.34);
  z-index: 20;
}

@media (max-width: 1440px) {
  .workspace-shell {
    grid-template-columns: 288px minmax(0, 1fr) 368px;
  }

}

@media (max-width: 1180px) {
  .workspace-shell {
    grid-template-columns: 280px minmax(0, 1fr) 360px;
  }
}

@media (max-width: 900px) {
  .workspace-shell {
    grid-template-columns: 1fr;
    padding: 16px;
  }

  .left-column,
  .right-column {
    position: fixed;
    top: 88px;
    bottom: 12px;
    width: min(88vw, 360px);
    z-index: 30;
    transition: transform 0.26s ease;
  }

  .left-column {
    left: 12px;
    transform: translateX(calc(-100% - 18px));
  }

  .right-column {
    right: 12px;
    transform: translateX(calc(100% + 18px));
  }

  .left-column.drawer-open,
  .right-column.drawer-open {
    transform: translateX(0);
  }
}
</style>
