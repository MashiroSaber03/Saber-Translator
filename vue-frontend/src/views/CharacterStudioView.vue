<template>
  <div class="studio-page">
    <StudioTopbar
      :subtitle="topbarSubtitle"
      :book-title="currentBookTitle"
      :document-title="store.currentDocument?.meta.title || ''"
      :document-origin="currentDocumentOrigin"
      :has-document="!!store.currentDocument"
      :preview-collapsed="effectivePreviewCollapsed"
      @back="goBack"
      @save="saveNow"
      @validate="validate"
      @open-export="store.activeEditorTab = 'export'"
      @toggle-preview="togglePreview"
      @toggle-left-drawer="store.leftDrawerOpen = !store.leftDrawerOpen"
      @toggle-right-drawer="store.rightDrawerOpen = !store.rightDrawerOpen"
    />

    <div v-if="!bookId" class="empty-state">
      <div class="empty-badge">缺少上下文</div>
      <h2>未检测到书籍参数</h2>
      <p>请从漫画分析页进入角色工坊，或在 URL 中携带 `book` 参数。角色工坊仍然依赖当前书籍的分析上下文。</p>
    </div>

    <div v-else class="workspace-shell" :class="{ 'preview-collapsed': effectivePreviewCollapsed, 'compact-desktop': isCompactDesktop }">
      <div v-if="store.leftDrawerOpen" class="drawer-mask left-mask" @click="store.leftDrawerOpen = false"></div>
      <div v-if="store.rightDrawerOpen && isCompactDesktop" class="drawer-mask right-mask" @click="store.rightDrawerOpen = false"></div>

      <aside class="left-column" :class="{ 'drawer-open': store.leftDrawerOpen }">
        <CharacterStudioSidebar
          :documents="store.filteredDocuments"
          :candidates="store.filteredCandidates"
          :search="store.selectedLibrarySearch"
          :current-document-id="store.currentDocument?.id || ''"
          :has-timeline="store.hasTimeline"
          @update:search="store.selectedLibrarySearch = $event"
          @open-document="openDocument"
          @create-manual="createManual"
          @create-from-candidate="createFromCandidate"
          @import-file="importFile"
        />
      </aside>

      <section class="editor-column">
        <CharacterStudioEditor
          :document="store.currentDocument"
          :avatar-url="avatarUrl"
          :saving="store.isSaving"
          :diagnostics="store.diagnostics"
          :active-tab="store.activeEditorTab"
          :active-script-tab="store.activeScriptTab"
          @update:document="store.currentDocument = $event"
          @update:active-tab="store.activeEditorTab = $event"
          @update:active-script-tab="store.activeScriptTab = $event"
          @save="saveNow"
          @generate="generateSection"
          @validate="validate"
          @delete="deleteCurrent"
          @import-worldbook="importWorldbook"
          @download="download"
        />
      </section>

      <aside class="right-column" :class="{ collapsed: effectivePreviewCollapsed, 'drawer-open': store.rightDrawerOpen }">
        <CharacterStudioPreview
          :document="store.currentDocument"
          :session="store.previewSession"
          :previewing="store.isPreviewing"
          :agent-busy="store.isAgentBusy"
          :agent-messages="store.agentMessages"
          :pending-patch="store.pendingAgentPatch"
          :can-undo-patch="store.canUndoPatch"
          :agent-html-preview="store.agentHtmlPreview"
          :collapsed="effectivePreviewCollapsed"
          @send-preview="sendPreview"
          @reset-preview="resetPreview"
          @send-agent="sendAgent"
          @apply-patch="store.applyPendingPatch()"
          @undo-patch="store.undoLastPatch()"
          @toggle-collapsed="togglePreview"
        />
      </aside>
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
const viewportWidth = ref(typeof window === 'undefined' ? 1440 : window.innerWidth)

const isCompactDesktop = computed(() => viewportWidth.value <= 1280 && viewportWidth.value > 900)
const effectivePreviewCollapsed = computed(() => (
  isCompactDesktop.value ? !store.rightDrawerOpen : store.previewCollapsed
))

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

async function hydrateWorkspace(nextBookId: string) {
  if (!bookshelfStore.books.length) {
    await bookshelfStore.fetchBooks()
  }
  await store.loadWorkspace(nextBookId)
  if (props.docId) {
    await store.openDocument(props.docId)
  } else if (store.documents.length > 0) {
    await store.openDocument(store.documents[0]!.id)
  }
}

function goBack() {
  if (!props.bookId) {
    void router.push({ name: 'insight' })
    return
  }
  void router.push({ name: 'insight', query: { book: props.bookId } })
}

function togglePreview() {
  if (isCompactDesktop.value) {
    store.rightDrawerOpen = !store.rightDrawerOpen
    return
  }
  store.previewCollapsed = !store.previewCollapsed
  if (store.previewCollapsed) {
    store.rightDrawerOpen = false
  }
}

async function openDocument(docId: string) {
  await store.openDocument(docId)
  closeDrawers()
  if (!props.bookId) return
  void router.replace({ name: 'character-studio', query: { book: props.bookId, doc: docId } })
}

async function createManual() {
  await store.createManualDocument()
  closeDrawers()
  if (!props.bookId || !store.currentDocument) return
  void router.replace({ name: 'character-studio', query: { book: props.bookId, doc: store.currentDocument.id } })
}

async function createFromCandidate(candidateName: string) {
  await store.createDocumentFromCandidate(candidateName)
  closeDrawers()
  if (!props.bookId || !store.currentDocument) return
  void router.replace({ name: 'character-studio', query: { book: props.bookId, doc: store.currentDocument.id } })
}

async function importFile(file: File) {
  await store.importFile(file)
  closeDrawers()
  if (!props.bookId || !store.currentDocument) return
  void router.replace({ name: 'character-studio', query: { book: props.bookId, doc: store.currentDocument.id } })
}

async function importWorldbook(file: File) {
  await store.importWorldbook(file)
}

async function saveNow() {
  await store.persistCurrentDocument()
}

async function validate() {
  await store.validateCurrentDocument()
}

async function generateSection(section: string) {
  await store.generateSection(section)
}

async function deleteCurrent() {
  await store.deleteCurrentDocument()
  if (!props.bookId) return
  void router.replace({ name: 'character-studio', query: { book: props.bookId } })
}

async function sendPreview(message: string) {
  await store.sendPreviewMessage(message)
}

async function resetPreview() {
  await store.resetPreview()
}

async function sendAgent(message: string) {
  await store.sendAgentMessage(message)
}

async function download(format: string) {
  await store.downloadCurrent(format)
}

onMounted(async () => {
  const handleResize = () => {
    viewportWidth.value = window.innerWidth
    if (window.innerWidth > 900) {
      store.leftDrawerOpen = false
    }
    if (window.innerWidth > 1280) {
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
  await store.openDocument(nextDocId)
})
</script>

<style scoped>
.studio-page {
  min-height: 100vh;
  margin: 0 -20px;
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
  min-height: calc(100vh - 112px);
}

.left-column,
.editor-column,
.right-column {
  min-width: 0;
  width: 100%;
  justify-self: stretch;
}

.left-column,
.right-column {
  align-self: start;
  position: sticky;
  top: 108px;
  max-height: calc(100vh - 132px);
}

.right-column {
  transition: width 0.24s ease, transform 0.24s ease;
}

.right-column.collapsed {
  width: 88px;
}

.workspace-shell.preview-collapsed {
  grid-template-columns: 304px minmax(0, 1fr) 88px;
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

  .workspace-shell.preview-collapsed {
    grid-template-columns: 288px minmax(0, 1fr) 88px;
  }
}

@media (max-width: 1180px) {
  .workspace-shell.compact-desktop {
    grid-template-columns: 280px minmax(0, 1fr) 76px;
  }

  .workspace-shell.compact-desktop.preview-collapsed {
    grid-template-columns: 280px minmax(0, 1fr) 76px;
  }

  .workspace-shell.compact-desktop .right-column {
    grid-column: auto;
    position: sticky;
    top: 108px;
    max-height: calc(100vh - 132px);
  }

  .workspace-shell.compact-desktop .right-column.collapsed {
    width: 76px;
  }

  .workspace-shell.compact-desktop .right-column.drawer-open {
    position: fixed;
    right: 16px;
    top: 108px;
    bottom: 16px;
    width: min(440px, calc(100vw - 32px));
    max-height: none;
    z-index: 30;
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
