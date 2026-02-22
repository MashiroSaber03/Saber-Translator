<script setup lang="ts">
/**
 * 书籍详情模态框组件
 * 使用与原版bookshelf.html完全相同的HTML结构和CSS类名
 */

import { ref, computed, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { getBookDetail } from '@/api/bookshelf'
import { showToast } from '@/utils/toast'
import BaseModal from '@/components/common/BaseModal.vue'

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
// 【复刻原版 bookshelf.js handleChapterDrop】
async function handleChapterReorder(chapterIds: string[]): Promise<boolean> {
  if (!currentBook.value) return false
  try {
    const success = await bookshelfStore.reorderChaptersApi(currentBook.value.id, chapterIds)
    if (success) {
      showToast('章节排序已更新', 'success')
      return true
    } else {
      showToast('排序保存失败', 'error')
      // 【复刻原版】刷新以恢复原始顺序
      await refreshBookDetail()
      return false
    }
  } catch (error) {
    showToast('排序保存失败', 'error')
    // 【复刻原版】刷新以恢复原始顺序
    await refreshBookDetail()
    return false
  }
}

// 【复刻原版】刷新当前书籍详情（用于排序失败后恢复原顺序）
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
    draggedChapterIndex.value = null
    dragOverChapterIndex.value = null
    return
  }

  // 重新排序
  const newOrder = [...chapters.value]
  const [removed] = newOrder.splice(draggedChapterIndex.value, 1)
  if (!removed) return
  newOrder.splice(targetIndex, 0, removed)

  // 发送新顺序到后端
  const chapterIds = newOrder.map(c => c.id)
  await handleChapterReorder(chapterIds)

  draggedChapterIndex.value = null
  dragOverChapterIndex.value = null
}

// 章节拖拽结束
function handleChapterDragEnd() {
  draggedChapterIndex.value = null
  dragOverChapterIndex.value = null
}

// 添加标签弹窗状态
const showAddTagModal = ref(false)
const quickTagFilter = ref('')
const quickTagInputRef = ref<HTMLInputElement | null>(null)

// 【复刻原版】过滤后的可用标签列表（排除已添加的标签）
const filteredAvailableTags = computed(() => {
  const currentTags = currentBook.value?.tags || []
  const filter = quickTagFilter.value.trim().toLowerCase()
  
  return allTags.value.filter(t => 
    !currentTags.includes(t.name) &&
    (filter === '' || t.name.toLowerCase().includes(filter))
  )
})

// 【复刻原版】是否显示创建新标签选项
const showCreateNewTagOption = computed(() => {
  const filter = quickTagFilter.value.trim()
  if (!filter) return false
  
  // 如果过滤词不完全匹配任何已有标签，则显示创建选项
  return !allTags.value.some(t => t.name.toLowerCase() === filter.toLowerCase())
})

// 【复刻原版】打开添加标签弹窗
function openAddTagModal() {
  quickTagFilter.value = ''
  showAddTagModal.value = true
  
  // 聚焦输入框
  nextTick(() => {
    quickTagInputRef.value?.focus()
  })
}

// 【复刻原版】关闭添加标签弹窗
function closeAddTagModal() {
  showAddTagModal.value = false
  quickTagFilter.value = ''
}

// 【复刻原版】处理输入框回车事件
async function handleQuickTagInputEnter() {
  const tagName = quickTagFilter.value.trim()
  if (tagName) {
    await quickAddTagToBook(tagName)
    quickTagFilter.value = ''
  }
}

// 标签操作加载状态
const isTagLoading = ref(false)

// 从书籍移除标签（用于详情页面的标签删除按钮）
// 【复刻原版 bookshelf.js removeTagFromCurrentBook】
// 步骤: 1. 获取当前书籍 tags  2. 过滤掉要删除的标签  3. PUT 更新整个 tags 数组
async function removeTag(tagName: string) {
  if (!currentBook.value || isTagLoading.value) return
  
  isTagLoading.value = true
  
  try {
    // 【复刻原版】获取当前的 tags 数组并过滤
    const currentTags = currentBook.value.tags || []
    const newTags = currentTags.filter(t => t !== tagName)
    
    // 【复刻原版】通过 updateBookApi 更新整个 tags 数组
    const success = await bookshelfStore.updateBookApi(currentBook.value.id, {
      tags: newTags
    })
    
    if (success) {
      // updateBookApi 已经自动更新了本地状态,不需要手动调用 updateBook
      showToast('标签已移除', 'success')
      // 【复刻原版】刷新书籍列表和标签列表
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

// 【复刻原版】快速添加标签到书籍（支持创建新标签）
// 步骤: 1. 如需创建新标签则创建  2. 获取当前 tags  3. 追加新标签  4. PUT 更新整个 tags 数组
async function quickAddTagToBook(tagName: string) {
  if (!currentBook.value || !tagName || isTagLoading.value) return
  
  // 检查是否已存在
  if (currentBook.value.tags?.includes(tagName)) {
    showToast('该标签已存在', 'info')
    return
  }
  
  isTagLoading.value = true
  
  try {
    const { createTag } = await import('@/api/bookshelf')
    
    // 如果是新标签，先创建
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
    
    // 【复刻原版】获取当前 tags 并追加新标签
    const currentTags = currentBook.value.tags || []
    const newTags = [...currentTags, tagName]
    
    // 【复刻原版】通过 updateBookApi 更新整个 tags 数组
    const success = await bookshelfStore.updateBookApi(currentBook.value.id, {
      tags: newTags
    })
    
    if (success) {
      // updateBookApi 已经自动更新了本地状态,不需要手动调用 updateBook
      showToast('标签已添加', 'success')
      // 【复刻原版】刷新书籍列表和标签列表
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
  <!-- 书籍详情模态框 - 基于 BaseModal -->
  <BaseModal
    title="书籍详情"
    size="large"
    custom-class="book-detail-modal"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="emit('close')"
  >
        <div v-if="currentBook" class="book-detail-container">
          <!-- 书籍信息 - 与原版相同的垂直布局 -->
          <div class="book-info-section">
            <div class="book-cover-large">
              <img
                v-if="currentBook.cover"
                :src="currentBook.cover"
                alt="封面"
              >
              <div v-else class="book-cover-placeholder">📖</div>
            </div>
            <div class="book-meta">
              <h3>{{ currentBook.title }}</h3>
              <p class="meta-item">
                <span>标签：</span>
                <span v-if="currentBook.tags && currentBook.tags.length > 0" class="detail-tags">
                  <span
                    v-for="tag in currentBook.tags"
                    :key="tag"
                    class="detail-tag"
                    :style="{ background: getTagColor(tag) }"
                  >
                    {{ tag }}
                    <span class="remove-detail-tag" @click.stop="removeTag(tag)">×</span>
                  </span>
                </span>
                <span v-else class="no-tags-hint">暂无标签</span>
                <button class="btn-add-tag" title="添加标签" @click="openAddTagModal">+</button>
              </p>
              <p class="meta-item"><span>章节数：</span><span>{{ chapters.length }}</span></p>
              <p class="meta-item"><span>创建时间：</span><span>{{ formatDate(currentBook.created_at || currentBook.createdAt) }}</span></p>
              <p class="meta-item"><span>最后更新：</span><span>{{ formatDate(currentBook.updated_at || currentBook.updatedAt) }}</span></p>
              <div class="book-actions">
                <button class="btn btn-sm btn-primary" @click="goToInsight">● 漫画分析</button>
                <button class="btn btn-sm btn-secondary" @click="editCurrentBook">编辑书籍</button>
                <button class="btn btn-sm btn-danger" @click="deleteCurrentBook">删除书籍</button>
              </div>
            </div>
          </div>

          <!-- 章节列表 -->
          <div class="chapters-section">
            <div class="section-header">
              <h3>章节列表</h3>
              <button class="btn btn-sm btn-primary" @click="openCreateChapterModal">
                <span class="btn-icon">+</span> 新建章节
              </button>
            </div>
            <div v-if="chapters.length > 0" class="chapters-list">
              <div
                v-for="(chapter, index) in chapters"
                :key="chapter.id"
                class="chapter-item"
                :class="{
                  dragging: draggedChapterIndex === index,
                  'drag-over': dragOverChapterIndex === index && draggedChapterIndex !== index
                }"
                draggable="true"
                @dragstart="handleChapterDragStart($event, index)"
                @dragover="handleChapterDragOver($event, index)"
                @dragleave="handleChapterDragLeave"
                @drop="handleChapterDrop($event, index)"
                @dragend="handleChapterDragEnd"
              >
                <div class="chapter-drag-handle" title="拖拽排序">⋮⋮</div>
                <div class="chapter-info">
                  <span class="chapter-order">#{{ index + 1 }}</span>
                  <span class="chapter-title">{{ chapter.title }}</span>
                  <span class="chapter-meta">{{ chapter.image_count || chapter.imageCount || 0 }} 张图片</span>
                </div>
                <div class="chapter-actions">
                  <button class="chapter-action-btn chapter-enter-btn" @click="goToTranslate(chapter.id)">
                    进入翻译
                  </button>
                  <button
                    class="chapter-action-btn chapter-read-btn"
                    :disabled="(chapter.image_count || chapter.imageCount || 0) === 0"
                    @click="goToReader(chapter.id)"
                  >
                    进入阅读
                  </button>
                  <button class="chapter-action-btn" @click="openEditChapterModal(chapter.id)">
                    编辑
                  </button>
                  <button class="chapter-action-btn danger" @click="deleteChapter(chapter.id)">
                    删除
                  </button>
                </div>
              </div>
            </div>
            <div v-else class="empty-state-small">
              <p>暂无章节，点击上方按钮创建</p>
            </div>
          </div>
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
        <div class="form-group">
          <label for="chapterTitleInput">章节名称 <span class="required">*</span></label>
          <input
            id="chapterTitleInput"
            v-model="chapterTitle"
            type="text"
            placeholder="例如：第1话、序章"
            @keypress.enter="saveChapter"
          >
        </div>
        <template #footer>
          <button type="button" class="btn btn-secondary" @click="showChapterModal = false">取消</button>
          <button type="button" class="btn btn-primary" @click="saveChapter">保存</button>
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
          <!-- 【复刻原版】搜索/创建输入框 -->
          <div class="quick-tag-input-wrapper">
            <input
              ref="quickTagInputRef"
              v-model="quickTagFilter"
              type="text"
              class="quick-tag-input"
              placeholder="输入标签名称进行搜索或创建..."
              @keypress.enter="handleQuickTagInputEnter"
            >
          </div>
          
          <!-- 【复刻原版】过滤后的可用标签列表 -->
          <div class="quick-tag-list">
            <!-- 可用标签 -->
            <div
              v-for="tag in filteredAvailableTags"
              :key="tag.name"
              class="quick-tag-item"
              @click="quickAddTagToBook(tag.name)"
            >
              <span class="tag-color-dot" :style="{ background: tag.color || '#667eea' }"></span>
              <span class="quick-tag-name">{{ tag.name }}</span>
              <span class="tag-add-icon">+</span>
            </div>
            
            <!-- 创建新标签选项 -->
            <div
              v-if="showCreateNewTagOption"
              class="quick-tag-item new-tag"
              @click="quickAddTagToBook(quickTagFilter.trim())"
            >
              <span class="tag-icon">+</span>
              <span>创建并添加 "{{ quickTagFilter.trim() }}"</span>
            </div>
            
            <!-- 无可用标签提示 -->
            <p 
              v-if="filteredAvailableTags.length === 0 && !showCreateNewTagOption" 
              class="no-tags-hint"
            >
              {{ quickTagFilter ? '未找到匹配的标签' : '所有标签已添加或暂无标签' }}
            </p>
          </div>
        <template #footer>
          <button type="button" class="btn btn-secondary" @click="closeAddTagModal">关闭</button>
        </template>
  </BaseModal>

  <!-- 删除确认模态框 -->
  <BaseModal
    v-model="showDeleteConfirm"
    title="确认删除"
    size="small"
    :close-on-overlay="true"
    :close-on-esc="true"
  >
          <p>
            {{ deleteTarget === 'book' 
              ? '确定要删除这本书籍吗？所有章节数据将一并删除，此操作不可恢复。' 
              : '确定要删除这个章节吗？此操作不可恢复。' 
            }}
          </p>
        <template #footer>
          <button type="button" class="btn btn-secondary" @click="showDeleteConfirm = false">取消</button>
          <button type="button" class="btn btn-danger" @click="confirmDelete">删除</button>
        </template>
  </BaseModal>
</template>

<style scoped>
/* ==================== 书籍详情模态框样式 - 完整迁移自 bookshelf.css ==================== */

/* 书籍详情容器 */
.book-detail-container {
    display: flex;
    flex-direction: column;
    gap: 24px;
}

/* 书籍信息区域 */
.book-info-section {
    display: flex;
    gap: 24px;
    align-items: flex-start;
}

.book-cover-large {
    width: 140px;
    flex-shrink: 0;
    aspect-ratio: 3 / 4;
    border-radius: 12px;
    overflow: hidden;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

.book-cover-large img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* 书籍详情右侧信息区 - 垂直排列 */
.book-meta {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
}

.book-meta h3 {
    font-size: 1.3rem;
    margin: 0 0 16px 0;
    color: var(--text-primary);
    font-weight: 600;
    line-height: 1.3;
    word-break: break-word;
}

/* 书籍详情元信息项 - 垂直排列 */
.book-meta .meta-item {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.book-meta .meta-item span:first-child {
    color: var(--text-primary);
    font-weight: 500;
    flex-shrink: 0;
    min-width: 70px;
}

.book-meta .detail-tags {
    display: inline-flex;
    gap: 6px;
    flex-wrap: wrap;
}

.book-meta .detail-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75rem;
    color: white;
}

.book-meta .no-tags-hint {
    color: var(--text-secondary);
    font-style: italic;
}

.book-meta .btn-add-tag {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    border: 1px dashed var(--border-color);
    background: transparent;
    color: var(--text-secondary);
    font-size: 0.9rem;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-left: 6px;
}

.book-meta .btn-add-tag:hover {
    border-color: #667eea;
    color: #667eea;
}

/* 操作按钮组 */
.book-actions {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    flex-wrap: wrap;
}

/* 章节区域 */
.chapters-section {
    border-top: 1px solid var(--border-color);
    padding-top: 16px;
}

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    flex-wrap: wrap;
    gap: 12px;
}

.section-header h3 {
    font-size: 1.05rem;
    margin: 0;
    color: var(--text-primary);
    font-weight: 600;
}

.chapters-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 280px;
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
    padding-right: 4px;
}

/* 自定义滚动条 */
.chapters-list::-webkit-scrollbar {
    width: 6px;
}

.chapters-list::-webkit-scrollbar-track {
    background: var(--hover-bg);
    border-radius: 3px;
}

.chapters-list::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 3px;
}

.chapters-list::-webkit-scrollbar-thumb:hover {
    background: var(--text-secondary);
}

.chapter-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--hover-bg);
    border-radius: 8px;
    transition: all 0.2s ease;
    gap: 12px;
}

.chapter-item:hover {
    background: var(--border-color);
}

.chapter-info {
    display: flex;
    align-items: center;
    gap: 12px;
    flex: 1;
    min-width: 0;
}

.chapter-order {
    font-size: 0.8rem;
    color: var(--text-secondary);
    min-width: 32px;
    flex-shrink: 0;
}

.chapter-title {
    font-weight: 500;
    font-size: 0.9rem;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.chapter-meta {
    font-size: 0.75rem;
    color: var(--text-secondary);
}

.chapter-actions {
    display: flex;
    gap: 6px;
    opacity: 1;
    flex-shrink: 0;
}

.chapter-action-btn {
    background: none;
    border: none;
    padding: 6px 10px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.2s;
}

.chapter-action-btn:hover {
    background: var(--card-bg);
    color: var(--text-primary);
}

.chapter-action-btn.danger:hover {
    color: #dc3545;
}

.chapter-enter-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 500;
}

.chapter-enter-btn:hover {
    background: linear-gradient(135deg, #7b8eef 0%, #8a5cb5 100%);
    color: white;
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.chapter-read-btn {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    color: white;
    font-weight: 500;
}

.chapter-read-btn:hover:not(:disabled) {
    background: linear-gradient(135deg, #34ce57 0%, #38d9a9 100%);
    color: white;
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4);
}

.chapter-read-btn:disabled {
    background: var(--border-color);
    color: var(--text-secondary);
    cursor: not-allowed;
    opacity: 0.6;
}

/* 空状态 */
.empty-state-small {
    padding: 40px 20px;
    text-align: center;
    color: var(--text-secondary);
}

/* ==================== 【复刻原版】快速添加标签样式 ==================== */

/* 快速标签输入框包装 */
.quick-tag-input-wrapper {
    margin-bottom: 16px;
}

.quick-tag-input {
    width: 100%;
    padding: 12px 16px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    font-size: 0.95rem;
    background: var(--card-bg);
    color: var(--text-primary);
    transition: all 0.2s;
}

.quick-tag-input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
}

.quick-tag-input::placeholder {
    color: var(--text-secondary);
}

/* 快速标签列表 */
.quick-tag-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 260px;
    overflow-y: auto;
}

/* 快速标签项 */
.quick-tag-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: var(--hover-bg);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
}

.quick-tag-item:hover {
    background: var(--border-color);
    transform: translateX(4px);
}

.quick-tag-item .tag-color-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
}

.quick-tag-item .quick-tag-name {
    flex: 1;
    font-weight: 500;
    color: var(--text-primary);
}

.quick-tag-item .tag-add-icon {
    font-size: 1.2rem;
    font-weight: 600;
    color: #667eea;
    opacity: 0;
    transition: opacity 0.2s;
}

.quick-tag-item:hover .tag-add-icon {
    opacity: 1;
}

/* 创建新标签选项 */
.quick-tag-item.new-tag {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    border: 1px dashed rgba(102, 126, 234, 0.4);
}

.quick-tag-item.new-tag:hover {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
    border-color: rgba(102, 126, 234, 0.6);
}

.quick-tag-item .tag-icon {
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    font-weight: 600;
    color: #667eea;
}

/* 无标签提示 */
.no-tags-hint {
    text-align: center;
    color: var(--text-secondary);
    font-style: italic;
    padding: 24px 16px;
    margin: 0;
}
</style>
