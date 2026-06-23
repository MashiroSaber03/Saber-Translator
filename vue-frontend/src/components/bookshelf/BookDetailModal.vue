<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { createTag, getBookDetail } from '@/api/bookshelf'
import { showToast } from '@/utils/toast'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import BookDeleteConfirmContent from './book-detail/BookDeleteConfirmContent.vue'
import BookDetailSummary from './book-detail/BookDetailSummary.vue'
import ChapterFormContent from './book-detail/ChapterFormContent.vue'
import ChapterList from './book-detail/ChapterList.vue'
import QuickTagPicker from './book-detail/QuickTagPicker.vue'

const emit = defineEmits<{
  close: []
  edit: [bookId: string]
}>()

const router = useRouter()
const bookshelfStore = useBookshelfStore()

// 章节模态框状态
const showChapterModal = ref(false)
const editingChapterId = ref<string | null>(null)
const chapterTitle = ref('')

// 确认删除状态
const showDeleteConfirm = ref(false)
const deleteTarget = ref<'book' | 'chapter'>('book')
const deleteChapterId = ref<string | null>(null)

// 计算属性
const currentBook = computed(() => bookshelfStore.currentBook)
const chapters = computed(() => currentBook.value?.chapters || [])
const allTags = computed(() => bookshelfStore.tags)

// 格式化日期
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

// 获取标签颜色
function getTagColor(tagName: string): string {
  const tagInfo = allTags.value.find(t => t.name === tagName)
  return tagInfo?.color || '#667eea'
}

// 编辑当前书籍
function editCurrentBook() {
  if (currentBook.value) {
    emit('edit', currentBook.value.id)
    emit('close')
  }
}

// 删除当前书籍
function deleteCurrentBook() {
  deleteTarget.value = 'book'
  showDeleteConfirm.value = true
}

// 确认删除
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
    showToast('删除失败', 'error')
  }
  showDeleteConfirm.value = false
  deleteChapterId.value = null
}

// 打开新建章节模态框
function openCreateChapterModal() {
  editingChapterId.value = null
  chapterTitle.value = ''
  showChapterModal.value = true
}

// 打开编辑章节模态框
function openEditChapterModal(chapterId: string) {
  const chapter = chapters.value.find(c => c.id === chapterId)
  if (chapter) {
    editingChapterId.value = chapterId
    chapterTitle.value = chapter.title
    showChapterModal.value = true
  }
}

// 保存章节
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
    showToast('保存失败', 'error')
  }
}

// 删除章节
function deleteChapter(chapterId: string) {
  deleteTarget.value = 'chapter'
  deleteChapterId.value = chapterId
  showDeleteConfirm.value = true
}

// 跳转到翻译页面
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

// 跳转到阅读器
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

// 跳转到漫画分析
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

// 处理章节排序
async function handleChapterReorder(chapterIds: string[]): Promise<boolean> {
  if (!currentBook.value) return false
  try {
    const success = await bookshelfStore.reorderChaptersApi(currentBook.value.id, chapterIds)
    if (success) {
      showToast('章节排序已更新', 'success')
      return true
    } else {
      showToast('排序保存失败', 'error')
      // 刷新以恢复原始顺序
      await refreshBookDetail()
      return false
    }
  } catch (error) {
    showToast('排序保存失败', 'error')
    // 刷新以恢复原始顺序
    await refreshBookDetail()
    return false
  }
}

// 刷新当前书籍详情（用于排序失败后恢复原顺序）
async function refreshBookDetail() {
  if (!currentBook.value) return
  try {
    const response = await getBookDetail(currentBook.value.id)
    if (response.success && response.book) {
      bookshelfStore.updateBook(currentBook.value.id, response.book)
    }
  } catch (error) {
    console.error('刷新书籍详情失败:', error)
  }
}

// 章节拖拽排序状态
const draggedChapterIndex = ref<number | null>(null)
const dragOverChapterIndex = ref<number | null>(null)

// 章节拖拽开始
function handleChapterDragStart(event: DragEvent, index: number) {
  draggedChapterIndex.value = index
  if (event.dataTransfer) {
    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData('text/plain', index.toString())
  }
}

// 章节拖拽经过
function handleChapterDragOver(event: DragEvent, index: number) {
  event.preventDefault()
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = 'move'
  }
  dragOverChapterIndex.value = index
}

// 章节拖拽离开
function handleChapterDragLeave() {
  dragOverChapterIndex.value = null
}

// 章节放置
async function handleChapterDrop(event: DragEvent, targetIndex: number) {
  event.preventDefault()

  if (draggedChapterIndex.value === null || draggedChapterIndex.value === targetIndex || !currentBook.value) {
    resetChapterDragState()
    return
  }

  // 重新排序
  const newOrder = [...chapters.value]
  const [removed] = newOrder.splice(draggedChapterIndex.value, 1)
  if (!removed) {
    resetChapterDragState()
    return
  }
  newOrder.splice(targetIndex, 0, removed)

  // 发送新顺序到后端
  const chapterIds = newOrder.map(c => c.id)
  await handleChapterReorder(chapterIds)

  resetChapterDragState()
}

// 章节拖拽结束
function handleChapterDragEnd() {
  resetChapterDragState()
}

function resetChapterDragState() {
  draggedChapterIndex.value = null
  dragOverChapterIndex.value = null
}

// 添加标签弹窗状态
const showAddTagModal = ref(false)
const quickTagFilter = ref('')

// 过滤后的可用标签列表（排除已添加的标签）
const filteredAvailableTags = computed(() => {
  const currentTags = currentBook.value?.tags || []
  const filter = quickTagFilter.value.trim().toLowerCase()

  return allTags.value.filter(t =>
    !currentTags.includes(t.name) &&
    (filter === '' || t.name.toLowerCase().includes(filter))
  )
})

// 是否显示创建新标签选项
const showCreateNewTagOption = computed(() => {
  const filter = quickTagFilter.value.trim()
  if (!filter) return false

  // 如果过滤词不完全匹配任何已有标签，则显示创建选项
  return !allTags.value.some(t => t.name.toLowerCase() === filter.toLowerCase())
})

// 打开添加标签弹窗
function openAddTagModal() {
  quickTagFilter.value = ''
  showAddTagModal.value = true
}

// 关闭添加标签弹窗
function closeAddTagModal() {
  showAddTagModal.value = false
  quickTagFilter.value = ''
}

// 处理输入框回车事件
async function handleQuickTagInputEnter() {
  const tagName = quickTagFilter.value.trim()
  if (tagName) {
    await quickAddTagToBook(tagName)
    quickTagFilter.value = ''
  }
}

// 标签操作加载状态
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
      // 标签写入后刷新书籍与标签索引。
      await bookshelfStore.loadBooks()
      await bookshelfStore.loadTags()
    } else {
      showToast('移除标签失败', 'error')
    }
  } catch (error) {
    showToast('操作失败', 'error')
    console.error('移除标签失败:', error)
  } finally {
    isTagLoading.value = false
  }
}

async function quickAddTagToBook(tagName: string) {
  if (!currentBook.value || !tagName || isTagLoading.value) return

  // 检查是否已存在
  if (currentBook.value.tags?.includes(tagName)) {
    showToast('该标签已存在', 'info')
    return
  }

  isTagLoading.value = true

  try {
    if (!allTags.value.some(t => t.name === tagName)) {
      const createResponse = await createTag(tagName)
      if (createResponse.success) {
        // 刷新标签列表
        await bookshelfStore.loadTags()
      } else {
        showToast('创建标签失败', 'error')
        return
      }
    }

    // 获取当前 tags 并追加新标签
    const currentTags = currentBook.value.tags || []
    const newTags = [...currentTags, tagName]

    const success = await bookshelfStore.updateBookApi(currentBook.value.id, {
      tags: newTags
    })

    if (success) {
      showToast('标签已添加', 'success')
      // 刷新书籍列表和标签列表
      await bookshelfStore.loadBooks()
      await bookshelfStore.loadTags()
    } else {
      showToast('添加标签失败', 'error')
    }
  } catch (error) {
    showToast('操作失败', 'error')
    console.error('快速添加标签失败:', error)
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
      />
    </div>
  </BaseModal>

  <!-- 章节编辑模态框 -->
  <BaseModal
    v-model="showChapterModal"
    :title="editingChapterId ? '编辑章节' : '新建章节'"
    size="small"
    :close-on-overlay="true"
    :close-on-esc="true"
  >
    <ChapterFormContent v-model="chapterTitle" @save="saveChapter" />
    <template #footer>
      <UiButton type="button" variant="secondary" @click="showChapterModal = false">取消</UiButton>
      <UiButton type="button" variant="primary" @click="saveChapter">保存</UiButton>
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
      <UiButton type="button" variant="secondary" @click="closeAddTagModal">关闭</UiButton>
    </template>
  </BaseModal>

  <!-- 删除确认模态框 -->
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
      <UiButton type="button" variant="secondary" @click="showDeleteConfirm = false">取消</UiButton>
      <UiButton type="button" variant="danger" @click="confirmDelete">删除</UiButton>
    </template>
  </BaseModal>
</template>

<style scoped>
.book-detail-container {
  --book-detail-modal-border-default: rgba(102, 126, 234, .4);
  --book-detail-modal-border-strong: rgba(102, 126, 234, .6);
  --book-detail-modal-shadow-default: rgba(0, 0, 0, .15);
  --book-detail-modal-shadow-raised: rgba(102, 126, 234, .4);
  --book-detail-modal-shadow-floating: rgba(40, 167, 69, .4);
  --book-detail-modal-shadow-strong: rgba(102, 126, 234, .15);
  --book-detail-modal-shadow-soft: rgba(102, 126, 234, .15);
  --book-detail-modal-surface-base: #7b8eef;
  --book-detail-modal-surface-raised: #8a5cb5;
  --book-detail-modal-surface-muted: #34ce57;
  --book-detail-modal-surface-subtle: #38d9a9;
  --book-detail-modal-surface-hover: rgba(102, 126, 234, .1);
  --book-detail-modal-surface-active: rgba(118, 75, 162, .1);
  --book-detail-modal-surface-selected: rgba(102, 126, 234, .2);
  --book-detail-modal-surface-overlay: rgba(118, 75, 162, .2);
  --book-detail-modal-text-primary: #667eea;

  display: flex;
  flex-direction: column;
  gap: 24px;
}
</style>
