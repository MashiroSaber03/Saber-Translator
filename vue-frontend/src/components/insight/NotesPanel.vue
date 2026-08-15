<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import { useInsightStore } from '@/stores/insightStore'
import type { NoteData, NoteType } from '@/types/insight'
import { confirmProductAction } from '@/composables/useProductConfirm'
import { showToast } from '@/utils/toast'
import NoteEditorModal from './notes/NoteEditorModal.vue'
import NotesList from './notes/NotesList.vue'
import NotesToolbar from './notes/NotesToolbar.vue'

const noteFilterOptions: Array<{ label: string; value: NoteType | 'all' }> = [
  { label: '全部', value: 'all' },
  { label: '文本笔记', value: 'text' },
  { label: '问答笔记', value: 'qa' },
]

const insightStore = useInsightStore()
const showNoteModal = ref(false)
const editingNote = ref<NoteData | null>(null)
const newNoteTitle = ref('')
const newNoteContent = ref('')
const newNotePageNum = ref<number | null>(null)
const newNoteTags = ref('')
const isLoadingNoteDetail = ref(false)
const loadingNoteId = ref<string | null>(null)
const isSavingNote = ref(false)
const deletingNoteIds = ref<Set<string>>(new Set())
let editorRequestSequence = 0

const noteTypeFilter = computed({
  get: () => insightStore.noteTypeFilter,
  set: value => {
    void insightStore.setNoteTypeFilter(value)
  },
})

function resetDraft(): void {
  newNoteTitle.value = ''
  newNoteContent.value = ''
  newNotePageNum.value = insightStore.selectedPageNum
  newNoteTags.value = ''
}

function openNoteModal(): void {
  editingNote.value = null
  resetDraft()
  showNoteModal.value = true
}

async function openEditModal(note: NoteData): Promise<void> {
  const bookId = insightStore.currentBookId
  if (!bookId || isLoadingNoteDetail.value || isSavingNote.value) return
  const requestId = ++editorRequestSequence
  isLoadingNoteDetail.value = true
  loadingNoteId.value = note.id
  try {
    const detail = await insightStore.loadNoteDetail(note.id)
    if (
      !detail
      || requestId !== editorRequestSequence
      || insightStore.currentBookId !== bookId
    ) return
    editingNote.value = detail
    newNoteTitle.value = detail.title
    newNoteContent.value = detail.content
    newNotePageNum.value = detail.pageNum || null
    newNoteTags.value = detail.tags.join(', ')
    showNoteModal.value = true
  } finally {
    if (requestId === editorRequestSequence) {
      isLoadingNoteDetail.value = false
      loadingNoteId.value = null
    }
  }
}

function closeNoteModal(): void {
  if (isSavingNote.value) return
  resetEditor()
}

function resetEditor(): void {
  editorRequestSequence += 1
  isLoadingNoteDetail.value = false
  loadingNoteId.value = null
  isSavingNote.value = false
  showNoteModal.value = false
  editingNote.value = null
  newNoteTitle.value = ''
  newNoteContent.value = ''
  newNoteTags.value = ''
}

function parseTags(tagsStr: string): string[] {
  if (!tagsStr.trim()) return []
  return tagsStr.split(/[,，]/).map(tag => tag.trim()).filter(Boolean)
}

async function saveNote(): Promise<void> {
  const bookId = insightStore.currentBookId
  const editedNote = editingNote.value
  if (
    !bookId
    || isSavingNote.value
    || (!newNoteContent.value.trim() && editedNote?.type !== 'qa')
  ) return
  const requestId = ++editorRequestSequence
  isSavingNote.value = true

  const tags = parseTags(newNoteTags.value)
  try {
    if (editedNote) {
      await insightStore.updateNote(
        editedNote.id,
        editedNote.type === 'qa'
          ? { title: newNoteTitle.value || undefined }
          : {
              title: newNoteTitle.value || undefined,
              content: newNoteContent.value,
              pageNum: newNotePageNum.value || undefined,
              tags,
            }
      )
    } else {
      await insightStore.addNote({
        title: newNoteTitle.value || undefined,
        content: newNoteContent.value,
        type: 'text',
        pageNum: newNotePageNum.value || undefined,
        tags,
      })
    }
    if (
      requestId === editorRequestSequence
      && insightStore.currentBookId === bookId
      && editingNote.value === editedNote
    ) {
      resetEditor()
    }
  } catch (error) {
    if (requestId === editorRequestSequence && insightStore.currentBookId === bookId) {
      showToast(
        insightStore.notesError
          ?? (error instanceof Error ? error.message : '保存笔记失败'),
        'error',
      )
    }
  } finally {
    if (requestId === editorRequestSequence) isSavingNote.value = false
  }
}

async function deleteNote(noteId: string): Promise<void> {
  const bookId = insightStore.currentBookId
  if (!bookId || deletingNoteIds.value.has(noteId)) return
  deletingNoteIds.value = new Set(deletingNoteIds.value).add(noteId)
  try {
    const confirmed = await confirmProductAction({
      title: '删除笔记',
      message: '确定要删除这条笔记吗？',
      confirmText: '删除',
      cancelText: '取消',
      tone: 'danger',
    })
    if (!confirmed || insightStore.currentBookId !== bookId) return
    await insightStore.deleteNote(noteId)
  } catch (error) {
    if (insightStore.currentBookId === bookId) {
      showToast(
        insightStore.notesError
          ?? (error instanceof Error ? error.message : '删除笔记失败'),
        'error',
      )
    }
  } finally {
    const next = new Set(deletingNoteIds.value)
    next.delete(noteId)
    deletingNoteIds.value = next
  }
}

function goToPage(pageNum: number): void {
  insightStore.selectPage(pageNum)
}

function retryNotes(): void {
  void insightStore.loadNotesFromAPI()
}

watch(
  () => insightStore.currentBookId,
  () => {
    resetEditor()
    deletingNoteIds.value = new Set()
  },
)
</script>

<template>
  <div class="notes-panel">
    <NotesToolbar
      :filter="noteTypeFilter"
      :filter-options="noteFilterOptions"
      @update:filter="noteTypeFilter = $event"
    />

    <ProductStatusBanner
      v-if="insightStore.notesLoading"
      class="notes-panel__status"
      tone="neutral"
      title="正在加载笔记"
      aria-live="polite"
    >
      正在读取当前书籍的笔记。
    </ProductStatusBanner>

    <ProductStatusBanner
      v-else-if="insightStore.notesError"
      class="notes-panel__status"
      tone="danger"
      title="笔记操作失败"
      aria-live="assertive"
    >
      {{ insightStore.notesError }}
      <template #actions>
        <UiButton variant="secondary" size="sm" @click="retryNotes">重新加载</UiButton>
      </template>
    </ProductStatusBanner>

    <NotesList
      v-if="
        !insightStore.notesLoading
          && (!insightStore.notesError || insightStore.notes.length > 0)
      "
      :notes="insightStore.notes"
      :busy-note-ids="Array.from(deletingNoteIds)"
      :editing-note-id="loadingNoteId"
      @delete="deleteNote"
      @edit="openEditModal"
      @show-page="goToPage"
    />

    <UiButton
      v-if="!insightStore.notesError && insightStore.notesNextCursor"
      variant="secondary"
      size="sm"
      class="notes-panel__load-more"
      :disabled="insightStore.notesLoadingMore"
      @click="insightStore.loadMoreNotes"
    >
      {{ insightStore.notesLoadingMore ? '加载中...' : '加载更多笔记' }}
    </UiButton>

    <UiButton
      variant="secondary"
      size="sm"
      class="notes-panel__add-button"
      :disabled="isLoadingNoteDetail || isSavingNote"
      @click="openNoteModal"
    >
      <UiIcon name="plus" size="14" />
      添加笔记
    </UiButton>

    <NoteEditorModal
      :visible="showNoteModal"
      :editing-note="editingNote"
      :is-saving="isSavingNote"
      :max-page="insightStore.totalPageCount || undefined"
      v-model:note-title="newNoteTitle"
      v-model:note-content="newNoteContent"
      v-model:note-page-num="newNotePageNum"
      v-model:note-tags="newNoteTags"
      @close="closeNoteModal"
      @save="saveNote"
      @show-page="goToPage"
    />
  </div>
</template>

<style scoped>
.notes-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 20px 18px;
}

.notes-panel__add-button {
  flex: 0 0 auto;
  width: 100%;
}

.notes-panel__status {
  margin-bottom: 12px;
}

.notes-panel__load-more {
  flex: 0 0 auto;
  width: 100%;
  margin-bottom: 8px;
}
</style>
