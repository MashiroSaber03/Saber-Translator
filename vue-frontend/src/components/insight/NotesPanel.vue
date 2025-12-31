<script setup lang="ts">
/**
 * 笔记面板组件
 * 管理漫画分析过程中的笔记
 */

import { ref, computed } from 'vue'
import { useInsightStore, type NoteType, type NoteData } from '@/stores/insightStore'
import CustomSelect from '@/components/common/CustomSelect.vue'

/** 笔记筛选类型选项 */
const noteFilterOptions = [
  { label: '全部', value: 'all' },
  { label: '文本笔记', value: 'text' },
  { label: '问答笔记', value: 'qa' }
]

/** 笔记类型选项 */
const noteTypeOptions = [
  { label: '文本笔记', value: 'text' },
  { label: '问答笔记', value: 'qa' }
]

// ============================================================
// 状态
// ============================================================

const insightStore = useInsightStore()

/** 是否显示添加笔记模态框 */
const showNoteModal = ref(false)

/** 编辑中的笔记 */
const editingNote = ref<NoteData | null>(null)

/** 新笔记标题 */
const newNoteTitle = ref('')

/** 新笔记内容 */
const newNoteContent = ref('')

/** 新笔记类型 */
const newNoteType = ref<NoteType>('text')

/** 新笔记关联页码 */
const newNotePageNum = ref<number | null>(null)

/** 新笔记标签 */
const newNoteTags = ref('')

// ============================================================
// 计算属性
// ============================================================

/** 过滤后的笔记列表 */
const filteredNotes = computed(() => insightStore.filteredNotes)

/** 当前筛选类型 */
const noteTypeFilter = computed({
  get: () => insightStore.noteTypeFilter,
  set: (value) => insightStore.setNoteTypeFilter(value)
})

// ============================================================
// 方法
// ============================================================

/**
 * 打开添加笔记模态框
 */
function openNoteModal(): void {
  editingNote.value = null
  newNoteTitle.value = ''
  newNoteContent.value = ''
  newNoteType.value = 'text'
  newNotePageNum.value = insightStore.selectedPageNum
  newNoteTags.value = ''
  showNoteModal.value = true
}

/**
 * 打开编辑笔记模态框
 * @param note - 要编辑的笔记
 */
function openEditModal(note: NoteData): void {
  editingNote.value = note
  newNoteTitle.value = note.title || ''
  newNoteContent.value = note.content
  newNoteType.value = note.type
  newNotePageNum.value = note.pageNum || null
  newNoteTags.value = (note.tags || []).join(', ')
  showNoteModal.value = true
}

/**
 * 关闭笔记模态框
 */
function closeNoteModal(): void {
  showNoteModal.value = false
  editingNote.value = null
  newNoteTitle.value = ''
  newNoteContent.value = ''
  newNoteTags.value = ''
}

/**
 * 保存笔记
 */
/**
 * 解析标签字符串为数组
 */
function parseTags(tagsStr: string): string[] {
  if (!tagsStr.trim()) return []
  return tagsStr.split(/[,，]/).map(t => t.trim()).filter(t => t)
}

async function saveNote(): Promise<void> {
  if (!newNoteContent.value.trim()) return

  const tags = parseTags(newNoteTags.value)

  if (editingNote.value) {
    // 更新现有笔记
    await insightStore.updateNote(editingNote.value.id, {
      title: newNoteTitle.value || undefined,
      content: newNoteContent.value,
      type: newNoteType.value,
      pageNum: newNotePageNum.value || undefined,
      tags: tags.length > 0 ? tags : undefined
    })
  } else {
    // 创建新笔记
    const note: NoteData = {
      id: Date.now().toString(),
      type: newNoteType.value,
      title: newNoteTitle.value || undefined,
      content: newNoteContent.value,
      pageNum: newNotePageNum.value || undefined,
      tags: tags.length > 0 ? tags : undefined,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }
    await insightStore.addNote(note)
  }

  closeNoteModal()
}

/**
 * 删除笔记
 * @param noteId - 笔记ID
 */
async function deleteNote(noteId: string): Promise<void> {
  if (!confirm('确定要删除这条笔记吗？')) return
  await insightStore.deleteNote(noteId)
}

/**
 * 跳转到笔记关联的页面
 * @param pageNum - 页码
 */
function goToPage(pageNum: number): void {
  insightStore.selectPage(pageNum)
}

/**
 * 格式化日期
 * @param dateStr - 日期字符串
 */
function formatDate(dateStr: string): string {
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

/**
 * 获取笔记类型图标
 * @param type - 笔记类型
 */
function getNoteTypeIcon(type: NoteType): string {
  return type === 'qa' ? '💬' : '📝'
}
</script>

<template>
  <div class="workspace-section notes-section">
    <div class="section-header-with-actions">
      <h3 class="section-title">📝 笔记</h3>
      <div class="notes-filter">
        <CustomSelect
          v-model="noteTypeFilter"
          :options="noteFilterOptions"
        />
      </div>
    </div>
    
    <!-- 笔记列表 -->
    <div class="notes-list">
      <div v-if="filteredNotes.length === 0" class="placeholder-text">
        暂无笔记
      </div>
      
      <div 
        v-for="note in filteredNotes" 
        :key="note.id"
        class="note-item"
        :class="{ 'qa-note': note.type === 'qa' }"
        @click="openEditModal(note)"
      >
        <div class="note-header">
          <span class="note-type-icon">{{ getNoteTypeIcon(note.type) }}</span>
          <span class="note-date">{{ formatDate(note.createdAt) }}</span>
          <div class="note-actions">
            <button 
              class="btn-icon-sm" 
              title="编辑"
              @click.stop="openEditModal(note)"
            >
              ✏️
            </button>
            <button 
              class="btn-icon-sm" 
              title="删除"
              @click.stop="deleteNote(note.id)"
            >
              🗑️
            </button>
          </div>
        </div>
        <div v-if="note.title" class="note-title">{{ note.title }}</div>
        <!-- 问答笔记显示问题预览 -->
        <div v-if="note.type === 'qa'" class="note-content">
          <div class="qa-preview-text">Q: {{ note.question?.substring(0, 60) }}...</div>
        </div>
        <!-- 文本笔记显示内容 -->
        <div v-else class="note-content">{{ note.content }}</div>
        <div v-if="note.tags && note.tags.length > 0" class="note-tags">
          <span v-for="tag in note.tags" :key="tag" class="note-tag">{{ tag }}</span>
        </div>
        <!-- 问答笔记显示引用页码 -->
        <div v-if="note.type === 'qa' && note.citations && note.citations.length > 0" class="note-citations">
          <span 
            v-for="citation in note.citations.slice(0, 3)" 
            :key="citation.page"
            class="citation-badge"
            @click.stop="goToPage(citation.page)"
          >
            第{{ citation.page }}页
          </span>
          <span v-if="note.citations.length > 3" class="citation-badge">+{{ note.citations.length - 3 }}</span>
        </div>
        <div v-if="note.pageNum" class="note-page-link">
          <button 
            class="btn-link" 
            @click.stop="goToPage(note.pageNum)"
          >
            📄 第 {{ note.pageNum }} 页
          </button>
        </div>
      </div>
    </div>
    
    <!-- 添加笔记按钮 -->
    <button 
      class="btn btn-secondary btn-block btn-sm" 
      @click="openNoteModal"
    >
      + 添加笔记
    </button>
    
    <!-- 笔记模态框 -->
    <div v-if="showNoteModal" class="modal show" @click.self="closeNoteModal">
      <div class="modal-content modal-sm">
        <div class="modal-header">
          <h3>{{ editingNote ? '编辑笔记' : '添加笔记' }}</h3>
          <button class="modal-close" @click="closeNoteModal">&times;</button>
        </div>
        <div class="modal-body">
          <!-- 问答笔记查看模式 -->
          <template v-if="editingNote && editingNote.type === 'qa'">
            <div class="qa-note-view">
              <div class="qa-section">
                <label class="qa-label">问题</label>
                <div class="qa-content">{{ editingNote.question }}</div>
              </div>
              <div class="qa-section">
                <label class="qa-label">回答</label>
                <div class="qa-content qa-answer">{{ editingNote.answer }}</div>
              </div>
              <div v-if="editingNote.citations && editingNote.citations.length > 0" class="qa-section">
                <label class="qa-label">引用页码</label>
                <div class="qa-citations">
                  <span 
                    v-for="citation in editingNote.citations" 
                    :key="citation.page"
                    class="qa-citation-badge"
                    @click="goToPage(citation.page)"
                  >
                    第{{ citation.page }}页
                  </span>
                </div>
              </div>
              <div v-if="editingNote.comment" class="qa-section">
                <label class="qa-label">补充说明</label>
                <div class="qa-content">{{ editingNote.comment }}</div>
              </div>
            </div>
            <div class="form-group">
              <label>笔记标题 <span class="label-optional">(可选)</span></label>
              <input 
                v-model="newNoteTitle" 
                type="text" 
                class="form-input"
                placeholder="修改标题..."
              >
            </div>
          </template>
          <!-- 文本笔记编辑模式 -->
          <template v-else>
            <div class="form-group">
              <label>笔记类型</label>
              <CustomSelect
                v-model="newNoteType"
                :options="noteTypeOptions"
              />
            </div>
            <div class="form-group">
              <label>标题 <span class="label-optional">(可选)</span></label>
              <input 
                v-model="newNoteTitle" 
                type="text" 
                class="form-input"
                placeholder="给笔记起个标题..."
              >
            </div>
            <div class="form-group">
              <label>内容 <span class="label-required">*</span></label>
              <textarea 
                v-model="newNoteContent"
                class="form-textarea"
                rows="5"
                placeholder="写下你的想法..."
              ></textarea>
            </div>
            <div class="form-group">
              <label>关联页码 <span class="label-optional">(可选)</span></label>
              <input 
                v-model.number="newNotePageNum" 
                type="number" 
                class="form-input"
                placeholder="输入页码"
                min="1"
              >
            </div>
            <div class="form-group">
              <label>标签 <span class="label-optional">(可选)</span></label>
              <input 
                v-model="newNoteTags" 
                type="text" 
                class="form-input"
                placeholder="多个标签用逗号分隔，如: 角色,剧情"
              >
            </div>
          </template>
        </div>
        <div class="modal-footer">
          <button class="btn btn-secondary" @click="closeNoteModal">取消</button>
          <button 
            class="btn btn-primary" 
            :disabled="editingNote?.type !== 'qa' && !newNoteContent.trim()"
            @click="saveNote"
          >
            保存
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ==================== NotesPanel 完整样式 ==================== */

/* ==================== CSS变量 ==================== */
.workspace-section {
  --bg-primary: #f8fafc;
  --bg-secondary: #ffffff;
  --bg-tertiary: #f1f5f9;
  --text-primary: #1a202c;
  --text-secondary: #64748b;
  --text-muted: #94a3b8;
  --border-color: #e2e8f0;
  --primary-color: #6366f1;
  --primary-light: #818cf8;
  --primary-dark: #4f46e5;
  --success-color: #22c55e;
  --warning-color: #f59e0b;
  --error-color: #ef4444;
}

/* ==================== 工作区通用样式 ==================== */
.workspace-section {
  padding: 16px;
  border-bottom: 1px solid var(--border-color);
}

.section-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ==================== 模态框样式 ==================== */
.modal {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 1000;
  align-items: center;
  justify-content: center;
}

.modal.show {
  display: flex;
}

.modal-content {
  position: relative;
  background: var(--bg-primary);
  border-radius: 16px;
  width: 90%;
  max-width: 500px;
  max-height: 90vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
}

.modal-content.modal-sm {
  max-width: 450px;
}

.modal-content.modal-lg {
  max-width: 700px;
}

.modal-header {
  padding: 20px;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.modal-header h3 {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.modal-close {
  width: 32px;
  height: 32px;
  border: none;
  background: transparent;
  font-size: 24px;
  cursor: pointer;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  transition: all 0.2s;
}

.modal-close:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.modal-body {
  padding: 20px;
  overflow-y: auto;
  flex: 1;
}

.modal-footer {
  padding: 16px 20px;
  border-top: 1px solid var(--border-color);
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

/* ==================== 按钮样式 ==================== */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 10px 18px;
  font-size: 14px;
  font-weight: 500;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  text-decoration: none;
}

.btn-primary {
  background: var(--primary-color);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: var(--primary-dark);
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-secondary {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
}

.btn-secondary:hover {
  background: var(--border-color);
}

.btn-block {
  width: 100%;
}

.btn-sm {
  padding: 8px 14px;
  font-size: 13px;
}

/* ==================== 表单样式 ==================== */
.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  margin-bottom: 6px;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.form-group input,
.form-group select,
.form-group textarea,
.form-input,
.form-textarea {
  width: 100%;
  padding: 10px 12px;
  font-size: 14px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  background: var(--bg-primary);
  color: var(--text-primary);
  transition: border-color 0.2s;
}

.form-group input:focus,
.form-group select:focus,
.form-group textarea:focus,
.form-input:focus,
.form-textarea:focus {
  outline: none;
  border-color: var(--primary-color);
}

/* ==================== 通用组件 ==================== */
.placeholder-text {
  color: var(--text-muted);
  text-align: center;
  padding: 20px;
  font-size: 14px;
}

/* ==================== 组件特定样式 ==================== */
.label-optional {
  font-size: 12px;
  color: var(--text-secondary);
  font-weight: normal;
}

.label-required {
  color: var(--error-color, #ef4444);
  font-weight: normal;
}

.note-item {
  padding: 12px;
  border-radius: 8px;
  background-color: var(--bg-secondary);
  margin-bottom: 8px;
}

.note-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.note-type-icon {
  font-size: 14px;
}

.note-date {
  flex: 1;
  font-size: 12px;
  color: var(--text-secondary);
}

.note-actions {
  display: flex;
  gap: 4px;
  margin-left: auto;
}

.note-actions .btn-icon-sm {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.note-actions .btn-icon-sm:hover {
  background: var(--bg-tertiary);
  border-color: var(--primary-color);
}

.note-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 6px;
}

.note-content {
  font-size: 14px;
  line-height: 1.5;
  white-space: pre-wrap;
  color: var(--text-secondary);
}

.note-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}

.note-tag {
  display: inline-block;
  padding: 2px 8px;
  font-size: 12px;
  background: var(--primary-light, rgba(99, 102, 241, 0.1));
  color: var(--primary-color, #6366f1);
  border-radius: 12px;
}

.note-page-link {
  margin-top: 8px;
}

.btn-link {
  background: none;
  border: none;
  color: var(--primary);
  cursor: pointer;
  font-size: 12px;
  padding: 0;
}

.btn-link:hover {
  text-decoration: underline;
}

/* ==================== 笔记面板完整样式 - 从 manga-insight.css 迁移 ==================== */

.notes-list {
    max-height: 300px;
    overflow-y: auto;
    margin-bottom: 12px;
}

.note-item {
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    margin-bottom: 10px;
    border: 1px solid var(--border-color);
    cursor: pointer;
    transition: all 0.2s ease;
}

.note-item:hover {
    border-color: var(--primary-color);
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.1);
}

.note-item.qa-note {
    border-left: 3px solid var(--primary-color);
}

.note-item.text-note {
    border-left: 3px solid var(--success-color);
}

.note-header {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 8px;
}

.note-type-badge {
    font-size: 16px;
    flex-shrink: 0;
}

.note-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.note-preview {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 8px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.note-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 8px;
}

.note-tag {
    font-size: 11px;
    padding: 2px 6px;
    background: var(--primary-color);
    color: white;
    border-radius: 10px;
    opacity: 0.8;
}

.note-meta {
    font-size: 11px;
    color: var(--text-secondary);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.note-meta-left {
    display: flex;
    align-items: center;
    gap: 8px;
}

.note-page-ref {
    color: var(--primary-color);
    cursor: pointer;
}

.note-page-ref:hover {
    text-decoration: underline;
}

.btn-delete-note {
    background: none;
    border: none;
    cursor: pointer;
    color: var(--text-secondary);
    font-size: 14px;
    padding: 2px 6px;
    border-radius: 4px;
    transition: all 0.2s;
}

.btn-delete-note:hover {
    color: var(--error-color);
    background: rgba(239, 68, 68, 0.1);
}

.section-header-with-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}

.section-header-with-actions .section-title {
    margin: 0;
}

.notes-filter-select {
    padding: 4px 8px;
    font-size: 12px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    background: var(--bg-secondary);
    color: var(--text-primary);
    cursor: pointer;
}

.note-detail-content {
    padding: 0;
}

.note-detail-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
}

.note-detail-type-icon {
    font-size: 32px;
}

.note-detail-info {
    flex: 1;
}

.note-detail-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
}

.note-detail-meta {
    font-size: 12px;
    color: var(--text-secondary);
}

.note-detail-body {
    margin-bottom: 16px;
}

.note-detail-section {
    margin-bottom: 20px;
}

.note-detail-section-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.note-detail-text {
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-primary);
    white-space: pre-wrap;
}

.note-detail-qa-section {
    background: var(--bg-tertiary);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
}

.note-detail-qa-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--primary-color);
    margin-bottom: 6px;
}

.note-detail-qa-content {
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-primary);
}

.note-detail-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.note-detail-tag {
    padding: 4px 10px;
    background: var(--primary-color);
    color: white;
    border-radius: 12px;
    font-size: 12px;
}

.note-detail-page-link {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    color: var(--primary-color);
    font-size: 13px;
    cursor: pointer;
    transition: background 0.2s;
}

.note-detail-page-link:hover {
    background: var(--bg-secondary);
}

/* 问答笔记预览样式 */
.qa-preview-text {
    font-size: 13px;
    color: var(--text-secondary);
    font-style: italic;
}

/* 引用页码标签 */
.note-citations {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}

.citation-badge {
    display: inline-block;
    padding: 2px 8px;
    background: var(--primary-color);
    color: white;
    border-radius: 10px;
    font-size: 11px;
    cursor: pointer;
    transition: opacity 0.2s;
}

.citation-badge:hover {
    opacity: 0.8;
}

/* 问答笔记查看模式 */
.qa-note-view {
    background: var(--bg-tertiary);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}

.qa-section {
    margin-bottom: 16px;
}

.qa-section:last-child {
    margin-bottom: 0;
}

.qa-label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.qa-content {
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-primary);
    background: var(--bg-secondary);
    padding: 12px;
    border-radius: 8px;
}

.qa-answer {
    max-height: 200px;
    overflow-y: auto;
}

.qa-citations {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.qa-citation-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    background: var(--primary-color);
    color: white;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.2s;
}

.qa-citation-badge:hover {
    opacity: 0.8;
}
</style>
