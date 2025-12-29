<script setup lang="ts">
/**
 * 书籍详情模态框组件
 * 使用与原版bookshelf.html完全相同的HTML结构和CSS类名
 */

import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { showToast } from '@/utils/toast'

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
async function handleChapterReorder(chapterIds: string[]) {
  if (!currentBook.value) return
  try {
    const success = await bookshelfStore.reorderChaptersApi(currentBook.value.id, chapterIds)
    if (success) {
      showToast('章节排序已更新', 'success')
    } else {
      showToast('排序失败', 'error')
    }
  } catch (error) {
    showToast('排序失败', 'error')
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

// 打开添加标签弹窗
function openAddTagModal() {
  showAddTagModal.value = true
}

// 标签操作加载状态
const isTagLoading = ref(false)

// 从书籍移除标签（用于详情页面的标签删除按钮）
async function removeTag(tagName: string) {
  if (!currentBook.value || isTagLoading.value) return
  
  isTagLoading.value = true
  
  try {
    // 使用批量移除标签 API（后端只支持批量操作）
    const { batchRemoveTags } = await import('@/api/bookshelf')
    const response = await batchRemoveTags([currentBook.value.id], [tagName])
    if (response.success) {
      bookshelfStore.removeTagFromBook(currentBook.value.id, tagName)
      showToast('标签已移除', 'success')
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

// 添加或移除标签（用于标签选择弹窗）
async function toggleTagOnBook(tagName: string) {
  if (!currentBook.value || isTagLoading.value) return
  
  const isAlreadyAdded = currentBook.value.tags?.includes(tagName)
  isTagLoading.value = true
  
  try {
    // 使用批量 API（后端只支持批量操作，传入标签名称而非 ID）
    const { batchAddTags, batchRemoveTags } = await import('@/api/bookshelf')
    
    if (isAlreadyAdded) {
      // 移除标签
      const response = await batchRemoveTags([currentBook.value.id], [tagName])
      if (response.success) {
        bookshelfStore.removeTagFromBook(currentBook.value.id, tagName)
        showToast('标签已移除', 'success')
      } else {
        showToast('移除标签失败', 'error')
      }
    } else {
      // 添加标签
      const response = await batchAddTags([currentBook.value.id], [tagName])
      if (response.success) {
        bookshelfStore.addTagToBook(currentBook.value.id, tagName)
        showToast('标签已添加', 'success')
      } else {
        showToast('添加标签失败', 'error')
      }
    }
  } catch (error) {
    showToast('操作失败', 'error')
    console.error('标签操作失败:', error)
  } finally {
    isTagLoading.value = false
  }
}
</script>

<template>
  <!-- 书籍详情模态框 - 使用与原版相同的HTML结构 -->
  <div class="modal active">
    <div class="modal-overlay" @click="emit('close')"></div>
    <div class="modal-content modal-large">
      <div class="modal-header">
        <h2>书籍详情</h2>
        <button class="modal-close" @click="emit('close')">&times;</button>
      </div>
      <div class="modal-body">
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
      </div>
    </div>
  </div>

  <!-- 章节编辑模态框 -->
  <Teleport to="body">
    <div v-if="showChapterModal" class="modal active">
      <div class="modal-overlay" @click="showChapterModal = false"></div>
      <div class="modal-content modal-small">
        <div class="modal-header">
          <h2>{{ editingChapterId ? '编辑章节' : '新建章节' }}</h2>
          <button class="modal-close" @click="showChapterModal = false">&times;</button>
        </div>
        <div class="modal-body">
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
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-secondary" @click="showChapterModal = false">取消</button>
          <button type="button" class="btn btn-primary" @click="saveChapter">保存</button>
        </div>
      </div>
    </div>
  </Teleport>

  <!-- 添加标签模态框 -->
  <Teleport to="body">
    <div v-if="showAddTagModal" class="modal active">
      <div class="modal-overlay" @click="showAddTagModal = false"></div>
      <div class="modal-content modal-small">
        <div class="modal-header">
          <h2>添加标签</h2>
          <button class="modal-close" @click="showAddTagModal = false">&times;</button>
        </div>
        <div class="modal-body">
          <div v-if="allTags.length > 0" class="tag-select-list">
            <div
              v-for="tag in allTags"
              :key="tag.id"
              class="tag-select-item"
              :class="{ selected: currentBook?.tags?.includes(tag.name) }"
              @click="toggleTagOnBook(tag.name)"
            >
              <span class="tag-color" :style="{ background: tag.color || '#667eea' }"></span>
              <span class="tag-name">{{ tag.name }}</span>
              <span v-if="currentBook?.tags?.includes(tag.name)" class="tag-check">✓</span>
            </div>
          </div>
          <div v-else class="empty-state-small">
            <p>暂无标签，请先在"管理标签"中创建</p>
          </div>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-secondary" @click="showAddTagModal = false">关闭</button>
        </div>
      </div>
    </div>
  </Teleport>

  <!-- 删除确认模态框 -->
  <Teleport to="body">
    <div v-if="showDeleteConfirm" class="modal active">
      <div class="modal-overlay" @click="showDeleteConfirm = false"></div>
      <div class="modal-content modal-small">
        <div class="modal-header">
          <h2>确认删除</h2>
          <button class="modal-close" @click="showDeleteConfirm = false">&times;</button>
        </div>
        <div class="modal-body">
          <p>
            {{ deleteTarget === 'book' 
              ? '确定要删除这本书籍吗？所有章节数据将一并删除，此操作不可恢复。' 
              : '确定要删除这个章节吗？此操作不可恢复。' 
            }}
          </p>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-secondary" @click="showDeleteConfirm = false">取消</button>
          <button type="button" class="btn btn-danger" @click="confirmDelete">删除</button>
        </div>
      </div>
    </div>
  </Teleport>
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
    color: white !important;
    font-weight: 500;
}

.chapter-enter-btn:hover {
    background: linear-gradient(135deg, #7b8eef 0%, #8a5cb5 100%) !important;
    color: white !important;
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.chapter-read-btn {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    color: white !important;
    font-weight: 500;
}

.chapter-read-btn:hover:not(:disabled) {
    background: linear-gradient(135deg, #34ce57 0%, #38d9a9 100%) !important;
    color: white !important;
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4);
}

.chapter-read-btn:disabled {
    background: var(--border-color);
    color: var(--text-secondary) !important;
    cursor: not-allowed;
    opacity: 0.6;
}

/* 标签选择列表 */
.tag-select-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 300px;
    overflow-y: auto;
}

.tag-select-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: var(--hover-bg);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
}

.tag-select-item:hover {
    background: var(--border-color);
}

.tag-select-item.selected {
    background: rgba(102, 126, 234, 0.15);
    border: 1px solid rgba(102, 126, 234, 0.3);
}

.tag-select-item .tag-color {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    flex-shrink: 0;
}

.tag-select-item .tag-name {
    flex: 1;
    font-weight: 500;
    color: var(--text-primary);
}

.tag-select-item .tag-check {
    color: #667eea;
    font-weight: bold;
}

/* 空状态 */
.empty-state-small {
    padding: 40px 20px;
    text-align: center;
    color: var(--text-secondary);
}
</style>
