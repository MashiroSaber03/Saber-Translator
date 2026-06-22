<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
/**
 * 笔记面板组件
 * 管理漫画分析过程中的笔记
 */

import { ref, computed } from 'vue'
import { useInsightStore, type NoteType, type NoteData } from '@/stores/insightStore'
import BaseModal from '@/components/common/BaseModal.vue'
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

const noteModalStyle = {
  maxWidth: '450px',
  width: '90%',
  borderRadius: '16px',
}

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
            <UiIconButton
              label="编辑"
              size="sm"
              @click.stop="openEditModal(note)"
            >
              ✏️
            </UiIconButton>
            <UiIconButton
              label="删除"
              variant="danger"
              size="sm"
              @click.stop="deleteNote(note.id)"
            >
              🗑️
            </UiIconButton>
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
          <UiButton
            variant="toolbar" 
            class="btn-link" 
            @click.stop="goToPage(note.pageNum)"
          >
            📄 第 {{ note.pageNum }} 页
          </UiButton>
        </div>
      </div>
    </div>
    
    <!-- 添加笔记按钮 -->
    <UiButton
      variant="secondary" 
      class="btn-block" 
      @click="openNoteModal" size="sm"
    >
      + 添加笔记
    </UiButton>
    
    <BaseModal
      v-model="showNoteModal"
      :title="editingNote ? '编辑笔记' : '添加笔记'"
      size="small"
      custom-class="notes-panel-modal"
      :custom-style="noteModalStyle"
      @close="closeNoteModal"
    >
      <template #title>
        <span>{{ editingNote ? '编辑笔记' : '添加笔记' }}</span>
      </template>

      <div class="notes-modal-body">
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
          <div class="notes-panel__field">
            <label>笔记标题 <span class="label-optional">(可选)</span></label>
            <UiInput 
              v-model="newNoteTitle" 
              type="text" 
              class="notes-panel__form-input"
              placeholder="修改标题..."
            />
          </div>
        </template>
        <!-- 文本笔记编辑模式 -->
        <template v-else>
          <div class="notes-panel__field">
            <label>笔记类型</label>
            <CustomSelect
              v-model="newNoteType"
              :options="noteTypeOptions"
            />
          </div>
          <div class="notes-panel__field">
            <label>标题 <span class="label-optional">(可选)</span></label>
            <UiInput 
              v-model="newNoteTitle" 
              type="text" 
              class="notes-panel__form-input"
              placeholder="给笔记起个标题..."
            />
          </div>
          <div class="notes-panel__field">
            <label>内容 <span class="label-required">*</span></label>
            <UiTextarea 
              v-model="newNoteContent"
              class="notes-panel__form-textarea"
              rows="5"
              placeholder="写下你的想法..."
            />
          </div>
          <div class="notes-panel__field">
            <label>关联页码 <span class="label-optional">(可选)</span></label>
            <UiInput 
              v-model.number="newNotePageNum" 
              type="number" 
              class="notes-panel__form-input"
              placeholder="输入页码"
              min="1"
            />
          </div>
          <div class="notes-panel__field">
            <label>标签 <span class="label-optional">(可选)</span></label>
            <UiInput 
              v-model="newNoteTags" 
              type="text" 
              class="notes-panel__form-input"
              placeholder="多个标签用逗号分隔，如: 角色,剧情"
            />
          </div>
        </template>
      </div>

      <template #footer>
        <UiButton variant="secondary" @click="closeNoteModal">取消</UiButton>
        <UiButton
          variant="primary" 
          :disabled="editingNote?.type !== 'qa' && !newNoteContent.trim()"
          @click="saveNote"
        >
          保存
        </UiButton>
      </template>
    </BaseModal>
  </div>
</template>

<style scoped>/* ==================== NotesPanel样式 ==================== */

/* ==================== 工作区通用样式 ==================== */
.notes-section .workspace-section {
  padding: 16px;
  border-bottom: 1px solid var(--color-border-muted);
}

.workspace-section.notes-section {
  --ui-button-padding: 10px 18px;
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--insight-color-primary);
  --ui-button-primary-hover-background: var(--insight-primary-dark);
  --ui-button-secondary-background: var(--insight-bg-tertiary);
  --ui-button-secondary-color: var(--insight-text-primary);
  --ui-button-secondary-border: 1px solid var(--color-border-muted);
  --ui-button-secondary-hover-background: var(--color-border-muted);
  --ui-button-sm-padding: 8px 14px;
  --ui-button-sm-font-size: 13px;
  --ui-button-disabled-opacity: 0.6;

  padding: 20px 18px;
}

.notes-section .section-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--insight-text-secondary);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.notes-section .btn-block {
  width: 100%;
}

/* ==================== 表单样式 ==================== */
.notes-section .notes-panel__field,
.notes-modal-body .notes-panel__field {
  margin-bottom: 16px;
}

.notes-section .notes-panel__field label,
.notes-modal-body .notes-panel__field label {
  display: block;
  margin-bottom: 6px;
  font-size: 14px;
  font-weight: 500;
  color: var(--insight-text-primary);
}

.notes-section .notes-panel__form-input,
.notes-section .notes-panel__form-textarea,
.notes-modal-body .notes-panel__form-input,
.notes-modal-body .notes-panel__form-textarea {
  width: 100%;
  padding: 10px 12px;
  font-size: 14px;
  border: 1px solid var(--color-border-muted);
  border-radius: 6px;
  background: var(--insight-bg-primary);
  color: var(--insight-text-primary);
  transition: border-color 0.2s;
}

.notes-section .notes-panel__form-input,
.notes-section .notes-panel__form-textarea,
.notes-modal-body .notes-panel__form-input,
.notes-modal-body .notes-panel__form-textarea {
  line-height: normal;
}

.notes-section .notes-panel__form-input:focus,
.notes-section .notes-panel__form-textarea:focus,
.notes-modal-body .notes-panel__form-input:focus,
.notes-modal-body .notes-panel__form-textarea:focus {
  outline: none;
  border-color: var(--insight-color-primary);
}

/* ==================== 通用组件 ==================== */
.notes-section .placeholder-text {
  color: var(--insight-text-muted);
  text-align: center;
  padding: 20px;
  font-size: 14px;
}

/* ==================== 组件特定样式 ==================== */
.notes-section .label-optional,
.notes-modal-body .label-optional {
  font-size: 12px;
  color: var(--insight-text-secondary);
  font-weight: normal;
}

.notes-section .label-required,
.notes-modal-body .label-required {
  color: var(--color-status-error, var(--notes-panel-text-primary));
  font-weight: normal;
}

.notes-section .note-item {
  padding: 12px;
  border-radius: 8px;
  background-color: var(--insight-bg-secondary);
  margin-bottom: 8px;
}

.notes-section .note-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.notes-section .note-type-icon {
  font-size: 14px;
}

.notes-section .note-date {
  flex: 1;
  font-size: 12px;
  color: var(--insight-text-secondary);
}

.notes-section .note-actions {
  display: flex;
  gap: 4px;
  margin-left: auto;
}

.notes-section .note-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--insight-text-primary);
  margin-bottom: 6px;
}

.notes-section .note-content {
  font-size: 14px;
  line-height: 1.5;
  white-space: pre-wrap;
  color: var(--insight-text-secondary);
}

.notes-section .note-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}

.notes-section .note-tag {
  display: inline-block;
  padding: 2px 8px;
  font-size: 12px;
  background: var(--color-focus-brand-soft);
  color: var(--color-action-primary, var(--color-text-brand));
  border-radius: 12px;
}

.notes-section .note-page-link {
  margin-top: 8px;
}

.notes-section .btn-link {
  background: none;
  border: none;
  color: var(--insight-primary);
  cursor: pointer;
  font-size: 12px;
  padding: 0;
}

.notes-section .btn-link:hover {
  text-decoration: underline;
}

/* ==================== 笔记面板样式 ==================== */

.notes-section .notes-list {
    max-height: 300px;
    overflow-y: auto;
    margin-bottom: 12px;
}

.notes-section .note-item {
    padding: 12px;
    background: var(--insight-bg-tertiary);
    border-radius: 8px;
    margin-bottom: 10px;
    border: 1px solid var(--color-border-muted);
    cursor: pointer;
    transition: all 0.2s ease;
}

.notes-section .note-item:hover {
    border-color: var(--insight-color-primary);
    box-shadow: 0 2px 8px var(--color-focus-brand-soft);
}

.notes-section .note-item.qa-note {
    border-left: 3px solid var(--insight-color-primary);
}

.notes-section .note-item.text-note {
    border-left: 3px solid var(--insight-success-color);
}

.notes-section .note-header {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 8px;
}

.notes-section .note-type-badge {
    font-size: 16px;
    flex-shrink: 0;
}

.notes-section .note-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--insight-text-primary);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.notes-section .note-preview {
    font-size: 13px;
    color: var(--insight-text-secondary);
    line-height: 1.5;
    margin-bottom: 8px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.notes-section .note-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 8px;
}

.notes-section .note-tag {
    font-size: 11px;
    padding: 2px 6px;
    background: var(--insight-color-primary);
    color: white;
    border-radius: 10px;
    opacity: 0.8;
}

.notes-section .note-meta {
    font-size: 11px;
    color: var(--insight-text-secondary);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.notes-section .note-meta-left {
    display: flex;
    align-items: center;
    gap: 8px;
}

.notes-section .note-page-ref {
    color: var(--insight-color-primary);
    cursor: pointer;
}

.notes-section .note-page-ref:hover {
    text-decoration: underline;
}

.notes-section .btn-delete-note {
    background: none;
    border: none;
    cursor: pointer;
    color: var(--insight-text-secondary);
    font-size: 14px;
    padding: 2px 6px;
    border-radius: 4px;
    transition: all 0.2s;
}

.notes-section .btn-delete-note:hover {
    color: var(--insight-error-color);
    background: var(--notes-panel-surface-base);
}

.notes-section .section-header-with-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}

.notes-section .section-header-with-actions .section-title {
    margin: 0;
}

.notes-section .notes-filter-select {
    padding: 4px 8px;
    font-size: 12px;
    border: 1px solid var(--color-border-muted);
    border-radius: 4px;
    background: var(--insight-bg-secondary);
    color: var(--insight-text-primary);
    cursor: pointer;
}

.notes-section .note-detail-content {
    padding: 0;
}

.notes-section .note-detail-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--color-border-muted);
}

.notes-section .note-detail-type-icon {
    font-size: 32px;
}

.notes-section .note-detail-info {
    flex: 1;
}

.notes-section .note-detail-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--insight-text-primary);
    margin-bottom: 4px;
}

.notes-section .note-detail-meta {
    font-size: 12px;
    color: var(--insight-text-secondary);
}

.notes-section .note-detail-body {
    margin-bottom: 16px;
}

.notes-section .note-detail-section {
    margin-bottom: 20px;
}

.notes-section .note-detail-section-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--insight-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.notes-section .note-detail-text {
    font-size: 14px;
    line-height: 1.7;
    color: var(--insight-text-primary);
    white-space: pre-wrap;
}

.notes-section .note-detail-qa-section {
    background: var(--insight-bg-tertiary);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
}

.notes-section .note-detail-qa-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--insight-color-primary);
    margin-bottom: 6px;
}

.notes-section .note-detail-qa-content {
    font-size: 14px;
    line-height: 1.6;
    color: var(--insight-text-primary);
}

.notes-section .note-detail-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.notes-section .note-detail-tag {
    padding: 4px 10px;
    background: var(--insight-color-primary);
    color: white;
    border-radius: 12px;
    font-size: 12px;
}

.notes-section .note-detail-page-link {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    background: var(--insight-bg-tertiary);
    border-radius: 8px;
    color: var(--insight-color-primary);
    font-size: 13px;
    cursor: pointer;
    transition: background 0.2s;
}

.notes-section .note-detail-page-link:hover {
    background: var(--insight-bg-secondary);
}

/* 问答笔记预览样式 */
.notes-section .qa-preview-text {
    font-size: 13px;
    color: var(--insight-text-secondary);
    font-style: italic;
}

/* 引用页码标签 */
.notes-section .note-citations {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}

.notes-section .citation-badge {
    display: inline-block;
    padding: 2px 8px;
    background: var(--insight-color-primary);
    color: white;
    border-radius: 10px;
    font-size: 11px;
    cursor: pointer;
    transition: opacity 0.2s;
}

.notes-section .citation-badge:hover {
    opacity: 0.8;
}

/* 问答笔记查看模式 */
.notes-section .qa-note-view,
.notes-modal-body .qa-note-view {
    background: var(--insight-bg-tertiary);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}

.notes-section .qa-section,
.notes-modal-body .qa-section {
    margin-bottom: 16px;
}

.notes-section .qa-section:last-child,
.notes-modal-body .qa-section:last-child {
    margin-bottom: 0;
}

.notes-section .qa-label,
.notes-modal-body .qa-label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: var(--insight-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.notes-section .qa-content,
.notes-modal-body .qa-content {
    font-size: 14px;
    line-height: 1.6;
    color: var(--insight-text-primary);
    background: var(--insight-bg-secondary);
    padding: 12px;
    border-radius: 8px;
}

.notes-section .qa-answer,
.notes-modal-body .qa-answer {
    max-height: 200px;
    overflow-y: auto;
}

.notes-section .qa-citations,
.notes-modal-body .qa-citations {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.notes-section .qa-citation-badge,
.notes-modal-body .qa-citation-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    background: var(--insight-color-primary);
    color: white;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.2s;
}

.notes-section .qa-citation-badge:hover,
.notes-modal-body .qa-citation-badge:hover {
    opacity: 0.8;
}
</style>
