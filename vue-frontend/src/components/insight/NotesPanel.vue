<script setup lang="ts">
import { computed, ref } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import { useInsightStore, type NoteData, type NoteType } from '@/stores/insightStore'
import NoteEditorModal from './notes/NoteEditorModal.vue'
import NotesList from './notes/NotesList.vue'
import NotesToolbar from './notes/NotesToolbar.vue'

const noteFilterOptions = [
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

function openEditModal(note: NoteData): void {
  editingNote.value = note
  newNoteTitle.value = note.title || ''
  newNoteContent.value = note.content
  newNoteType.value = note.type
  newNotePageNum.value = note.pageNum || null
  newNoteTags.value = (note.tags || []).join(', ')
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
    await insightStore.addNote({
      ...notePayload,
      id: Date.now().toString(),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    })
  }

  closeNoteModal()
}

async function deleteNote(noteId: string): Promise<void> {
  if (!confirm('确定要删除这条笔记吗？')) return
  await insightStore.deleteNote(noteId)
}

function goToPage(pageNum: number): void {
  insightStore.selectPage(pageNum)
}
</script>

<template>
  <div class="workspace-section notes-section">
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
      variant="secondary"
      size="sm"
      class="btn-block"
      @click="openNoteModal"
    >
      + 添加笔记
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
.workspace-section.notes-section {
  --ui-button-padding: 10px 18px;
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--insight-action-primary);
  --ui-button-primary-hover-background: var(--insight-action-primary-strong);
  --ui-button-secondary-background: var(--insight-surface-tertiary);
  --ui-button-secondary-color: var(--insight-text-primary);
  --ui-button-secondary-border: 1px solid var(--color-border-muted);
  --ui-button-secondary-hover-background: var(--color-border-muted);
  --ui-button-sm-padding: 8px 14px;
  --ui-button-sm-font-size: 13px;
  --ui-button-disabled-opacity: 0.6;

  padding: 20px 18px;
}

.btn-block {
  width: 100%;
}
</style>
