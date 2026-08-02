<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { createTag, getBookDetail } from '@/api/bookshelf'
import { showToast } from '@/utils/toast'
import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'
import BookDeleteConfirmContent from './book-detail/BookDeleteConfirmContent.vue'
import BookDetailSummary from './book-detail/BookDetailSummary.vue'
import ChapterFormContent from './book-detail/ChapterFormContent.vue'
import ChapterList from './book-detail/ChapterList.vue'
import QuickTagPicker from './book-detail/QuickTagPicker.vue'
import { createTranslationBatch } from '@/api/v2/translation'
import { useTaskCenterStore } from '@/stores/taskCenterStore'

const emit = defineEmits<{
  close: []
  edit: [bookId: string]
}>()

const router = useRouter()
const bookshelfStore = useBookshelfStore()
const taskCenterStore = useTaskCenterStore()

const showChapterModal = ref(false)
const editingChapterId = ref<string | null>(null)
const chapterTitle = ref('')
const selectedChapterIds = ref(new Set<string>())

const showDeleteConfirm = ref(false)
const deleteTarget = ref<'book' | 'chapter'>('book')
const deleteChapterId = ref<string | null>(null)

const currentBook = computed(() => bookshelfStore.currentBook)
const chapters = computed(() => currentBook.value?.chapters || [])
const allTags = computed(() => bookshelfStore.tags)

function formatDate(dateStr?: string): string {
  if (!dateStr) return '-'
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function getTagColor(tagName: string): string {
  const tagInfo = allTags.value.find(t => t.name === tagName)
  return tagInfo?.color || 'var(--color-action-brand)'
}

function editCurrentBook() {
  if (currentBook.value) {
    emit('edit', currentBook.value.id)
    emit('close')
  }
}

function deleteCurrentBook() {
  deleteTarget.value = 'book'
  showDeleteConfirm.value = true
}

async function confirmDelete() {
  try {
    if (deleteTarget.value === 'book' && currentBook.value) {
      const success = await bookshelfStore.deleteBookApi(currentBook.value.id)
      if (success) {
        showToast('书籍已删除', 'success')
        emit('close')
      } else {
        showToast('删除失败', 'error')
      }
    } else if (deleteTarget.value === 'chapter' && deleteChapterId.value && currentBook.value) {
      const success = await bookshelfStore.deleteChapterApi(currentBook.value.id, deleteChapterId.value)
      if (success) {
        showToast('章节已删除', 'success')
      } else {
        showToast('删除失败', 'error')
      }
    }
  } catch (error) {
    if (
      error
      && typeof error === 'object'
      && 'status' in error
      && error.status === 423
    ) {
      showToast('存在进行中的任务或导入，请先在任务中心处理', 'warning')
      taskCenterStore.open({
        bookId: currentBook.value?.id,
        chapterId: deleteChapterId.value || undefined,
      })
    } else {
      showToast(error instanceof Error ? error.message : '删除失败', 'error')
    }
  }
  showDeleteConfirm.value = false
  deleteChapterId.value = null
}

function selectChapter(chapterId: string, selected: boolean) {
  const next = new Set(selectedChapterIds.value)
  if (selected) next.add(chapterId)
  else next.delete(chapterId)
  selectedChapterIds.value = next
}

function selectAllChapters(chapterIds: string[]) {
  selectedChapterIds.value = new Set(chapterIds)
}

async function translateSelectedChapters() {
  const chapterIds = [...selectedChapterIds.value]
  if (!chapterIds.length) return
  try {
    const result = await createTranslationBatch(chapterIds, { mode: 'standard' })
    selectedChapterIds.value = new Set()
    await taskCenterStore.refresh()
    await refreshBookDetail()
    const skipped = result.skipped.length
    showToast(
      skipped
        ? `已创建 ${result.jobIds.length} 个任务，跳过 ${skipped} 个章节`
        : `已创建 ${result.jobIds.length} 个后端翻译任务`,
      skipped ? 'warning' : 'success',
    )
    taskCenterStore.open({ batchId: result.batchId })
  } catch (error) {
    showToast(error instanceof Error ? error.message : '创建批量翻译任务失败', 'error')
  }
}

function openCreateChapterModal() {
  editingChapterId.value = null
  chapterTitle.value = ''
  showChapterModal.value = true
}

function openEditChapterModal(chapterId: string) {
  const chapter = chapters.value.find(c => c.id === chapterId)
  if (chapter) {
    editingChapterId.value = chapterId
    chapterTitle.value = chapter.title
    showChapterModal.value = true
  }
}

async function saveChapter() {
  if (!chapterTitle.value.trim() || !currentBook.value) {
    showToast('请输入章节名称', 'warning')
    return
  }

  try {
    if (editingChapterId.value) {
      const success = await bookshelfStore.updateChapterApi(
        currentBook.value.id,
        editingChapterId.value,
        chapterTitle.value.trim()
      )
      if (success) {
        showToast('章节更新成功', 'success')
        showChapterModal.value = false
      } else {
        showToast('更新失败', 'error')
      }
    } else {
      const chapter = await bookshelfStore.createChapterApi(currentBook.value.id, chapterTitle.value.trim())
      if (chapter) {
        showToast('章节创建成功', 'success')
        showChapterModal.value = false
      } else {
        showToast('创建失败', 'error')
      }
    }
  } catch (error) {
    const status = (
      error
      && typeof error === 'object'
      && 'status' in error
    ) ? Number(error.status) : 0
    if (status === 423) {
      showToast('本章存在进行中的任务，请先在任务中心取消或等待任务结束', 'warning')
      taskCenterStore.open({
        bookId: currentBook.value.id,
        chapterId: editingChapterId.value || undefined,
      })
    } else {
      showToast(error instanceof Error ? error.message : '保存失败', 'error')
    }
  }
}

function deleteChapter(chapterId: string) {
  deleteTarget.value = 'chapter'
  deleteChapterId.value = chapterId
  showDeleteConfirm.value = true
}

function goToTranslate(chapterId: string) {
  if (currentBook.value) {
    router.push({
      path: '/translate',
      query: {
        book: currentBook.value.id,
        chapter: chapterId,
      },
    })
  }
}

function goToReader(chapterId: string) {
  if (currentBook.value) {
    router.push({
      path: '/reader',
      query: {
        book: currentBook.value.id,
        chapter: chapterId,
      },
    })
  }
}

function goToInsight() {
  if (currentBook.value) {
    router.push({
      path: '/insight',
      query: {
        book: currentBook.value.id,
      },
    })
  }
}

async function handleChapterReorder(chapterIds: string[]): Promise<boolean> {
  if (!currentBook.value) return false
  try {
    const success = await bookshelfStore.reorderChaptersApi(currentBook.value.id, chapterIds)
    if (success) {
      showToast('章节排序已更新', 'success')
      return true
    } else {
      showToast('排序保存失败', 'error')
      await refreshBookDetail()
      return false
    }
  } catch (error) {
    showToast('排序保存失败', 'error')
    await refreshBookDetail()
    return false
  }
}

async function refreshBookDetail() {
  if (!currentBook.value) return
  try {
    const book = await getBookDetail(currentBook.value.id)
    bookshelfStore.updateBook(currentBook.value.id, book)
  } catch {
    // Sort rollback is best-effort; the visible order remains unchanged if refresh fails.
  }
}

const draggedChapterIndex = ref<number | null>(null)
const dragOverChapterIndex = ref<number | null>(null)

function handleChapterDragStart(event: DragEvent, index: number) {
  draggedChapterIndex.value = index
  if (event.dataTransfer) {
    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData('text/plain', index.toString())
  }
}

function handleChapterDragOver(event: DragEvent, index: number) {
  event.preventDefault()
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = 'move'
  }
  dragOverChapterIndex.value = index
}

function handleChapterDragLeave() {
  dragOverChapterIndex.value = null
}

async function handleChapterDrop(event: DragEvent, targetIndex: number) {
  event.preventDefault()

  if (draggedChapterIndex.value === null || draggedChapterIndex.value === targetIndex || !currentBook.value) {
    resetChapterDragState()
    return
  }

  const newOrder = [...chapters.value]
  const [removed] = newOrder.splice(draggedChapterIndex.value, 1)
  if (!removed) {
    resetChapterDragState()
    return
  }
  newOrder.splice(targetIndex, 0, removed)

  const chapterIds = newOrder.map(c => c.id)
  await handleChapterReorder(chapterIds)

  resetChapterDragState()
}

function handleChapterDragEnd() {
  resetChapterDragState()
}

function resetChapterDragState() {
  draggedChapterIndex.value = null
  dragOverChapterIndex.value = null
}

const showAddTagModal = ref(false)
const quickTagFilter = ref('')

const filteredAvailableTags = computed(() => {
  const currentTags = currentBook.value?.tags || []
  const filter = quickTagFilter.value.trim().toLowerCase()

  return allTags.value.filter(t =>
    !currentTags.includes(t.name) &&
    (filter === '' || t.name.toLowerCase().includes(filter))
  )
})

const showCreateNewTagOption = computed(() => {
  const filter = quickTagFilter.value.trim()
  if (!filter) return false

  return !allTags.value.some(t => t.name.toLowerCase() === filter.toLowerCase())
})

function openAddTagModal() {
  quickTagFilter.value = ''
  showAddTagModal.value = true
}

function closeAddTagModal() {
  showAddTagModal.value = false
  quickTagFilter.value = ''
}

async function handleQuickTagInputEnter() {
  const tagName = quickTagFilter.value.trim()
  if (tagName) {
    await quickAddTagToBook(tagName)
    quickTagFilter.value = ''
  }
}

const isTagLoading = ref(false)

async function removeTag(tagName: string) {
  if (!currentBook.value || isTagLoading.value) return

  isTagLoading.value = true

  try {
    const currentTags = currentBook.value.tags || []
    const newTags = currentTags.filter(t => t !== tagName)

    const success = await bookshelfStore.updateBookApi(currentBook.value.id, {
      tags: newTags
    })

    if (success) {
      showToast('标签已移除', 'success')
    } else {
      showToast('移除标签失败', 'error')
    }
  } catch {
    showToast('操作失败', 'error')
  } finally {
    isTagLoading.value = false
  }
}

async function quickAddTagToBook(tagName: string) {
  if (!currentBook.value || !tagName || isTagLoading.value) return

  if (currentBook.value.tags?.includes(tagName)) {
    showToast('该标签已存在', 'info')
    return
  }

  isTagLoading.value = true

  try {
    if (!allTags.value.some(t => t.name === tagName)) {
      await createTag(tagName)
    }

    const currentTags = currentBook.value.tags || []
    const newTags = [...currentTags, tagName]

    const success = await bookshelfStore.updateBookApi(currentBook.value.id, {
      tags: newTags
    })

    if (success) {
      showToast('标签已添加', 'success')
    } else {
      showToast('添加标签失败', 'error')
    }
  } catch {
    showToast('操作失败', 'error')
  } finally {
    isTagLoading.value = false
  }
}
</script>

<template>
  <BaseModal
    title="书籍详情"
    size="large"
    custom-class="book-detail-modal"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="emit('close')"
  >
    <div v-if="currentBook" class="book-detail-container">
      <BookDetailSummary
        :book="currentBook"
        :chapter-count="chapters.length"
        :format-date="formatDate"
        :get-tag-color="getTagColor"
        @add-tag="openAddTagModal"
        @delete="deleteCurrentBook"
        @edit="editCurrentBook"
        @insight="goToInsight"
        @remove-tag="removeTag"
      />

      <ChapterList
        :chapters="chapters"
        :drag-over-chapter-index="dragOverChapterIndex"
        :dragged-chapter-index="draggedChapterIndex"
        :selected-chapter-ids="selectedChapterIds"
        @create="openCreateChapterModal"
        @delete="deleteChapter"
        @drag-end="handleChapterDragEnd"
        @drag-leave="handleChapterDragLeave"
        @drag-over="handleChapterDragOver"
        @drag-start="handleChapterDragStart"
        @drop="handleChapterDrop"
        @edit="openEditChapterModal"
        @read="goToReader"
        @translate="goToTranslate"
        @select="selectChapter"
        @select-all="selectAllChapters"
        @translate-selected="translateSelectedChapters"
      />
    </div>
  </BaseModal>

  <BaseModal
    v-model="showChapterModal"
    :title="editingChapterId ? '编辑章节' : '新建章节'"
    size="small"
    :close-on-overlay="true"
    :close-on-esc="true"
  >
    <ChapterFormContent v-model="chapterTitle" @save="saveChapter" />
    <template #footer>
      <ProductActionRow
        aria-label="章节表单操作"
        variant="dialog"
      >
        <UiButton type="button" variant="secondary" @click="showChapterModal = false">取消</UiButton>
        <UiButton type="button" variant="primary" @click="saveChapter">保存</UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>

  <BaseModal
    v-model="showAddTagModal"
    title="快速添加标签"
    size="small"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="closeAddTagModal"
  >
    <QuickTagPicker
      v-model:filter="quickTagFilter"
      :available-tags="filteredAvailableTags"
      :show-create-new-tag-option="showCreateNewTagOption"
      @add="quickAddTagToBook"
      @submit="handleQuickTagInputEnter"
    />
    <template #footer>
      <ProductActionRow
        aria-label="快速标签操作"
        variant="dialog"
      >
        <UiButton type="button" variant="secondary" @click="closeAddTagModal">关闭</UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>

  <BaseModal
    v-model="showDeleteConfirm"
    title="确认删除"
    size="small"
    custom-class="confirm-modal"
    body-text-align="center"
    :close-on-overlay="true"
    :close-on-esc="true"
  >
    <BookDeleteConfirmContent :target="deleteTarget" />
    <template #footer>
      <ProductActionRow
        aria-label="书籍详情删除操作"
        variant="dialog"
      >
        <UiButton type="button" variant="secondary" @click="showDeleteConfirm = false">取消</UiButton>
        <UiButton type="button" variant="danger" @click="confirmDelete">删除</UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<style scoped>
.book-detail-container {
  display: flex;
  flex-direction: column;
  gap: 24px;
}
</style>
