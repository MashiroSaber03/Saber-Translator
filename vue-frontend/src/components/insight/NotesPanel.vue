<script setup lang="ts">
import { computed, ref } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import { useInsightStore } from '@/stores/insightStore'
import type { NoteData, NoteType } from '@/types/insight'
import { confirmProductAction } from '@/composables/useProductConfirm'
import NoteEditorModal from './notes/NoteEditorModal.vue'
import NotesList from './notes/NotesList.vue'
import NotesToolbar from './notes/NotesToolbar.vue'

const noteFilterOptions: Array<{ label: string; value: NoteType | 'all' }> = [
  { label: '全部', value: 'all' },
  { label: '文本笔记', value: 'text' },
  { label: '问答笔记', value: 'qa' },
]

const noteTypeOptions: Array<{ label: string; value: NoteType }> = [
  { label: '文本笔记', value: 'text' },
  { label: '问答笔记', value: 'qa' },
]

const insightStore = useInsightStore()
const showNoteModal = ref(false)
const editingNote = ref<NoteData | null>(null)
const newNoteTitle = ref('')
const newNoteContent = ref('')
const newNoteType = ref<NoteType>('text')
const newNotePageNum = ref<number | null>(null)
const newNoteTags = ref('')

const filteredNotes = computed(() => insightStore.filteredNotes)
const noteTypeFilter = computed({
  get: () => insightStore.noteTypeFilter,
  set: value => insightStore.setNoteTypeFilter(value),
})

function resetDraft(): void {
  newNoteTitle.value = ''
  newNoteContent.value = ''
  newNoteType.value = 'text'
  newNotePageNum.value = insightStore.selectedPageNum
  newNoteTags.value = ''
}

function openNoteModal(): void {
  editingNote.value = null
  resetDraft()
  showNoteModal.value = true
}

async function openEditModal(note: NoteData): Promise<void> {
  const detail = await insightStore.loadNoteDetail(note.id)
  if (!detail) return
  editingNote.value = detail
  newNoteTitle.value = detail.title || ''
  newNoteContent.value = detail.content
  newNoteType.value = detail.type
  newNotePageNum.value = detail.pageNum || null
  newNoteTags.value = (detail.tags || []).join(', ')
  showNoteModal.value = true
}

function closeNoteModal(): void {
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
  if (!newNoteContent.value.trim() && editingNote.value?.type !== 'qa') return

  const tags = parseTags(newNoteTags.value)
  const notePayload = {
    title: newNoteTitle.value || undefined,
    content: newNoteContent.value,
    type: newNoteType.value,
    pageNum: newNotePageNum.value || undefined,
    tags: tags.length > 0 ? tags : undefined,
  }

  if (editingNote.value) {
    await insightStore.updateNote(editingNote.value.id, notePayload)
  } else {
    await insightStore.addNote(notePayload)
  }

  closeNoteModal()
}

async function deleteNote(noteId: string): Promise<void> {
  const confirmed = await confirmProductAction({
    title: '删除笔记',
    message: '确定要删除这条笔记吗？',
    confirmText: '删除',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed) return
  await insightStore.deleteNote(noteId)
}

function goToPage(pageNum: number): void {
  insightStore.selectPage(pageNum)
}
</script>

<template>
  <div class="notes-panel">
    <NotesToolbar
      :filter="noteTypeFilter"
      :filter-options="noteFilterOptions"
      @update:filter="noteTypeFilter = $event"
    />

    <NotesList
      :notes="filteredNotes"
      @delete="deleteNote"
      @edit="openEditModal"
      @show-page="goToPage"
    />

    <UiButton
      v-if="insightStore.notesNextCursor"
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
      @click="openNoteModal"
    >
      <UiIcon name="plus" size="14" />
      添加笔记
    </UiButton>

    <NoteEditorModal
      :visible="showNoteModal"
      :editing-note="editingNote"
      :note-type-options="noteTypeOptions"
      v-model:note-title="newNoteTitle"
      v-model:note-content="newNoteContent"
      v-model:note-type="newNoteType"
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

.notes-panel__load-more {
  flex: 0 0 auto;
  width: 100%;
  margin-bottom: 8px;
}
</style>
