<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { ApiClientError } from '@/api/client'
import { createChaptersExportJob, getBookDetail } from '@/api/bookshelf'
import { showToast, useToast } from '@/utils/toast'
import { triggerUrlDownload, withDownloadFileName } from '@/utils/browserDownload'
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
import { useSettingsStore } from '@/stores/settings'

const emit = defineEmits<{
  close: []
  edit: [bookId: string]
}>()

withDefaults(defineProps<{
  characterStudioAllowed?: boolean
  insightAllowed?: boolean
  translationAllowed?: boolean
}>(), {
  characterStudioAllowed: true,
  insightAllowed: true,
  translationAllowed: true,
})

const router = useRouter()
const bookshelfStore = useBookshelfStore()
const taskCenterStore = useTaskCenterStore()
const settingsStore = useSettingsStore()
const toast = useToast()

const showChapterModal = ref(false)
const editingChapterId = ref<string | null>(null)
const chapterTitle = ref('')
const selectedChapterIds = ref(new Set<string>())

const showDeleteConfirm = ref(false)
const deleteTarget = ref<'book' | 'chapter'>('book')
const deleteChapterId = ref<string | null>(null)
const isDeleting = ref(false)
const isChapterSaving = ref(false)
const isBatchDownloading = ref(false)
const isBatchTranslating = ref(false)
const isReordering = ref(false)

const currentBook = computed(() => bookshelfStore.currentBook)
const chapters = computed(() => currentBook.value?.chapters || [])
const allTags = computed(() => bookshelfStore.tags)

interface DeleteRequest {
  bookId: string
  chapterId?: string
}

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

function lockedJobId(error: ApiClientError): string | undefined {
  const details = error.details
  if (!details || typeof details !== 'object' || Array.isArray(details)) return undefined
  const jobs = 'jobs' in details ? details.jobs : undefined
  if (!Array.isArray(jobs)) return undefined
  for (const value of jobs) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) continue
    const jobId = 'jobId' in value ? value.jobId : undefined
    if (typeof jobId === 'string' && jobId) return jobId
  }
  return undefined
}

async function executeDelete(request: DeleteRequest): Promise<void> {
  if (request.chapterId) {
    await bookshelfStore.deleteChapterApi(request.bookId, request.chapterId)
  } else {
    await bookshelfStore.deleteBookApi(request.bookId)
  }
}

async function confirmDelete() {
  if (isDeleting.value) return
  const bookId = currentBook.value?.id
  if (!bookId) return
  const request: DeleteRequest = {
    bookId,
    chapterId: deleteTarget.value === 'chapter'
      ? deleteChapterId.value || undefined
      : undefined,
  }
  if (deleteTarget.value === 'chapter' && !request.chapterId) return

  isDeleting.value = true
  try {
    await executeDelete(request)

    if (!request.chapterId) {
      showToast('书籍已删除', 'success')
      emit('close')
    } else {
      const nextSelection = new Set(selectedChapterIds.value)
      nextSelection.delete(request.chapterId)
      selectedChapterIds.value = nextSelection
      showToast('章节已删除', 'success')
    }
  } catch (error) {
    if (error instanceof ApiClientError && error.status === 423) {
      showToast('仍有正在执行的任务或导入，请先在任务中心处理', 'warning')
      taskCenterStore.open({
        jobId: lockedJobId(error),
        bookId: request.bookId,
        chapterId: request.chapterId,
      })
    } else {
      showToast(error instanceof Error ? error.message : '删除失败', 'error')
    }
  } finally {
    isDeleting.value = false
    showDeleteConfirm.value = false
    deleteChapterId.value = null
  }
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
  if (!chapterIds.length || isBatchTranslating.value || isBatchDownloading.value) return
  isBatchTranslating.value = true
  try {
    const result = await createTranslationBatch({ chapterIds }, { mode: 'standard' })
    selectedChapterIds.value = new Set()
    await taskCenterStore.refresh().catch(() => undefined)
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
  } finally {
    isBatchTranslating.value = false
  }
}

async function downloadSelectedChapters() {
  const chapterIds = [...selectedChapterIds.value]
  if (!chapterIds.length || isBatchDownloading.value || isBatchTranslating.value) return
  isBatchDownloading.value = true
  let queuedToastId: number | null = null
  try {
    const accepted = await createChaptersExportJob(
      chapterIds,
      settingsStore.exportPreferences.preserveOriginalFilenames,
    )
    const jobId = accepted.jobIds[0]
    if (!jobId) throw new Error('后端没有返回批量章节导出任务')
    await taskCenterStore.refresh()
    taskCenterStore.open({ batchId: accepted.batchId })
    queuedToastId = showToast('章节下载已进入后端队列，可安全关闭书籍详情', 'info', 0)
    const job = await taskCenterStore.waitForJob(jobId)
    const artifact = job.artifacts[0]
    if (!artifact) throw new Error('批量章节导出任务未生成可下载文件')
    triggerUrlDownload(
      withDownloadFileName(artifact.url, `chapters-${chapterIds.length}-export.zip`),
    )
    selectedChapterIds.value = new Set()
    showToast('章节导出完成，下载已开始', 'success')
  } catch (error) {
    showToast(error instanceof Error ? error.message : '章节批量下载失败', 'error')
  } finally {
    if (queuedToastId !== null) toast.removeToast(queuedToastId)
    isBatchDownloading.value = false
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
  if (isChapterSaving.value) return
  if (!chapterTitle.value.trim() || !currentBook.value) {
    showToast('请输入章节名称', 'warning')
    return
  }

  isChapterSaving.value = true
  try {
    if (editingChapterId.value) {
      await bookshelfStore.updateChapterApi(
        currentBook.value.id,
        editingChapterId.value,
        chapterTitle.value.trim()
      )
      showToast('章节更新成功', 'success')
      showChapterModal.value = false
    } else {
      await bookshelfStore.createChapterApi(currentBook.value.id, chapterTitle.value.trim())
      showToast('章节创建成功', 'success')
      showChapterModal.value = false
    }
  } catch (error) {
    if (error instanceof ApiClientError && error.status === 423) {
      showToast('本章存在进行中的任务，请先在任务中心取消或等待任务结束', 'warning')
      taskCenterStore.open({
        bookId: currentBook.value.id,
        chapterId: editingChapterId.value || undefined,
      })
    } else {
      showToast(error instanceof Error ? error.message : '保存失败', 'error')
    }
  } finally {
    isChapterSaving.value = false
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

function goToCharacterStudio() {
  if (currentBook.value) {
    router.push({
      name: 'character-studio',
      query: { book: currentBook.value.id },
    })
  }
}

async function handleChapterReorder(chapterIds: string[]): Promise<boolean> {
  if (!currentBook.value || isReordering.value) return false
  isReordering.value = true
  try {
    await bookshelfStore.reorderChaptersApi(currentBook.value.id, chapterIds)
    showToast('章节排序已更新', 'success')
    return true
  } catch (error) {
    showToast('排序保存失败', 'error')
    await refreshBookDetail()
    return false
  } finally {
    isReordering.value = false
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
    const added = await quickAddTagToBook(tagName)
    if (added) quickTagFilter.value = ''
  }
}

const isTagLoading = ref(false)

async function removeTag(tagName: string) {
  if (!currentBook.value || isTagLoading.value) return

  isTagLoading.value = true

  try {
    const currentTags = currentBook.value.tags || []
    const newTags = currentTags.filter(t => t !== tagName)

    await bookshelfStore.updateBookApi(currentBook.value.id, {
      tags: newTags
    })
    showToast('标签已移除', 'success')
  } catch (error) {
    showToast(error instanceof Error ? error.message : '操作失败', 'error')
  } finally {
    isTagLoading.value = false
  }
}

async function quickAddTagToBook(tagName: string): Promise<boolean> {
  if (!currentBook.value || !tagName || isTagLoading.value) return false

  const existingTag = allTags.value.find(
    tag => tag.name.toLowerCase() === tagName.toLowerCase(),
  )
  const canonicalName = existingTag?.name ?? tagName

  if (currentBook.value.tags?.includes(canonicalName)) {
    showToast('该标签已存在', 'info')
    return false
  }

  isTagLoading.value = true

  try {
    if (!existingTag) {
      await bookshelfStore.createTag(canonicalName)
    }

    const currentTags = currentBook.value.tags || []
    const newTags = [...currentTags, canonicalName]

    await bookshelfStore.updateBookApi(currentBook.value.id, {
      tags: newTags
    })
    showToast('标签已添加', 'success')
    return true
  } catch (error) {
    showToast(error instanceof Error ? error.message : '操作失败', 'error')
    return false
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
        :character-studio-allowed="characterStudioAllowed"
        :chapter-count="chapters.length"
        :format-date="formatDate"
        :get-tag-color="getTagColor"
        :insight-allowed="insightAllowed"
        @add-tag="openAddTagModal"
        @delete="deleteCurrentBook"
        @edit="editCurrentBook"
        @insight="goToInsight"
        @character-studio="goToCharacterStudio"
        @remove-tag="removeTag"
      />

      <ChapterList
        :chapters="chapters"
        :drag-over-chapter-index="dragOverChapterIndex"
        :dragged-chapter-index="draggedChapterIndex"
        :download-pending="isBatchDownloading"
        :selected-chapter-ids="selectedChapterIds"
        :translation-pending="isBatchTranslating"
        :translation-allowed="translationAllowed"
        @create="openCreateChapterModal"
        @delete="deleteChapter"
        @drag-end="handleChapterDragEnd"
        @drag-leave="handleChapterDragLeave"
        @drag-over="handleChapterDragOver"
        @drag-start="handleChapterDragStart"
        @drop="handleChapterDrop"
        @download-selected="downloadSelectedChapters"
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
        <UiButton type="button" variant="primary" :loading="isChapterSaving" @click="saveChapter">保存</UiButton>
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
        <UiButton type="button" variant="danger" :loading="isDeleting" @click="confirmDelete">删除</UiButton>
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
