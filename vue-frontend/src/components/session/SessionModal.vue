<template>
  <BaseModal
    v-model="isVisible"
    :title="modalTitle"
    size="medium"
    @close="handleClose"
  >
    <!-- 选项卡切换 -->
    <div class="session-tabs">
      <button
        class="tab-btn"
        :class="{ active: activeTab === 'save' }"
        @click="activeTab = 'save'"
      >
        💾 保存会话
      </button>
      <button
        class="tab-btn"
        :class="{ active: activeTab === 'load' }"
        @click="activeTab = 'load'"
      >
        📂 加载会话
      </button>
    </div>

    <!-- 保存会话面板 -->
    <div v-if="activeTab === 'save'" class="session-panel">
      <div class="form-group">
        <label for="sessionName">会话名称</label>
        <input
          id="sessionName"
          v-model="sessionName"
          type="text"
          class="form-input"
          placeholder="请输入会话名称"
          @keyup.enter="handleSave"
        />
      </div>
      <div class="form-hint">
        <span v-if="imageCount > 0">当前有 {{ imageCount }} 张图片将被保存</span>
        <span v-else class="warning">没有图片可保存</span>
      </div>
      
      <!-- 保存进度条 -->
      <div v-if="isSaving && saveProgress.total > 0" class="save-progress">
        <div class="progress-text">保存图片 {{ saveProgress.current }}/{{ saveProgress.total }}...</div>
        <div class="progress-bar">
          <div 
            class="progress-fill" 
            :style="{ width: (saveProgress.current / saveProgress.total * 100) + '%' }"
          ></div>
        </div>
      </div>
      
      <div class="action-buttons">
        <button
          class="btn btn-primary"
          :disabled="!canSave || isSaving"
          @click="handleSave"
        >
          {{ isSaving ? '保存中...' : '保存' }}
        </button>
        <button class="btn btn-secondary" @click="handleClose">取消</button>
      </div>
    </div>

    <!-- 加载会话面板 -->
    <div v-if="activeTab === 'load'" class="session-panel">
      <!-- 加载状态 -->
      <div v-if="isLoadingList" class="loading-state">
        <span class="loading-spinner"></span>
        <span>加载会话列表...</span>
      </div>

      <!-- 会话列表 -->
      <div v-else-if="sessionList.length > 0" class="session-list">
        <div
          v-for="session in sessionList"
          :key="session.name"
          class="session-item"
          :class="{ selected: selectedSession === session.name }"
          @click="selectedSession = session.name"
          @dblclick="handleLoad"
        >
          <div class="session-info">
            <div class="session-name">{{ session.name }}</div>
            <div class="session-meta">
              <span>{{ session.imageCount }} 张图片</span>
              <span>{{ formatDate(session.savedAt) }}</span>
              <span v-if="session.version">v{{ session.version }}</span>
            </div>
          </div>
          <div class="session-actions">
            <button
              class="action-btn"
              title="重命名"
              @click.stop="startRename(session)"
            >
              ✏️
            </button>
            <button
              class="action-btn danger"
              title="删除"
              @click.stop="confirmDelete(session)"
            >
              🗑️
            </button>
          </div>
        </div>
      </div>

      <!-- 空状态 -->
      <div v-else class="empty-state">
        <span>暂无保存的会话</span>
      </div>

      <!-- 操作按钮 -->
      <div class="action-buttons">
        <button
          class="btn btn-primary"
          :disabled="!selectedSession || isLoading"
          @click="handleLoad"
        >
          {{ isLoading ? '加载中...' : '加载' }}
        </button>
        <button class="btn btn-secondary" @click="refreshList">刷新列表</button>
      </div>
    </div>

    <!-- 重命名对话框 -->
    <div v-if="showRenameDialog" class="rename-dialog-overlay" @click.self="cancelRename">
      <div class="rename-dialog">
        <h4>重命名会话</h4>
        <input
          v-model="newSessionName"
          type="text"
          class="form-input"
          placeholder="请输入新名称"
          @keyup.enter="handleRename"
        />
        <div class="dialog-buttons">
          <button class="btn btn-primary" @click="handleRename">确定</button>
          <button class="btn btn-secondary" @click="cancelRename">取消</button>
        </div>
      </div>
    </div>

    <!-- 删除确认对话框 -->
    <div v-if="showDeleteConfirm" class="rename-dialog-overlay" @click.self="cancelDelete">
      <div class="rename-dialog">
        <h4>确认删除</h4>
        <p>确定要删除会话 "{{ sessionToDelete?.name }}" 吗？此操作不可恢复。</p>
        <div class="dialog-buttons">
          <button class="btn btn-danger" @click="handleDelete">删除</button>
          <button class="btn btn-secondary" @click="cancelDelete">取消</button>
        </div>
      </div>
    </div>
  </BaseModal>
</template>

<script setup lang="ts">
/**
 * 会话管理模态框组件
 * 提供会话的保存、加载、删除、重命名功能
 */
import { ref, computed, watch } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import { useSessionStore } from '@/stores/sessionStore'
import { useImageStore } from '@/stores/imageStore'
import {
  getSessionList,
  loadSession,
  deleteSession,
  renameSession
} from '@/api/session'
import { useSettingsStore } from '@/stores/settingsStore'
import type { SessionListItem } from '@/types/api'
import { showToast } from '@/utils/toast'

// Props 定义
interface Props {
  /** 控制模态框显示/隐藏 */
  modelValue: boolean
  /** 默认激活的选项卡 */
  defaultTab?: 'save' | 'load'
}

const props = withDefaults(defineProps<Props>(), {
  defaultTab: 'save'
})

// Emits 定义
const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  /** 会话加载完成 */
  'session-loaded': [sessionData: unknown]
  /** 会话保存完成 */
  'session-saved': [sessionName: string]
}>()

// Store
const sessionStore = useSessionStore()
const imageStore = useImageStore()
const settingsStore = useSettingsStore()

// 状态
const activeTab = ref<'save' | 'load'>(props.defaultTab)
const sessionName = ref('')
const sessionList = ref<SessionListItem[]>([])
const selectedSession = ref<string | null>(null)
const isLoadingList = ref(false)
const isLoading = ref(false)
const isSaving = ref(false)
const saveProgress = ref<{ current: number; total: number }>({ current: 0, total: 0 })

// 重命名相关
const showRenameDialog = ref(false)
const sessionToRename = ref<SessionListItem | null>(null)
const newSessionName = ref('')

// 删除确认相关
const showDeleteConfirm = ref(false)
const sessionToDelete = ref<SessionListItem | null>(null)

// 计算属性
const isVisible = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value)
})

const modalTitle = computed(() => {
  return activeTab.value === 'save' ? '保存会话' : '加载会话'
})

const imageCount = computed(() => imageStore.images.length)

const canSave = computed(() => {
  return sessionName.value.trim() !== '' && imageCount.value > 0
})

// 监听模态框打开
watch(
  () => props.modelValue,
  (visible) => {
    if (visible) {
      activeTab.value = props.defaultTab
      if (props.defaultTab === 'load') {
        refreshList()
      }
    }
  }
)

// 监听选项卡切换
watch(activeTab, (tab) => {
  if (tab === 'load' && sessionList.value.length === 0) {
    refreshList()
  }
})

// 方法

/**
 * 刷新会话列表
 */
async function refreshList() {
  isLoadingList.value = true
  try {
    const response = await getSessionList()
    if (response.success && response.sessions) {
      sessionList.value = response.sessions
      sessionStore.setSessionList(response.sessions)
    } else {
      showToast(response.error || '获取会话列表失败', 'error')
    }
  } catch (error) {
    console.error('获取会话列表失败:', error)
    showToast('获取会话列表失败', 'error')
  } finally {
    isLoadingList.value = false
  }
}

/**
 * 保存会话（逐页保存，显示进度）
 */
async function handleSave() {
  if (!canSave.value) return

  isSaving.value = true
  const totalImages = imageStore.images.length
  saveProgress.value = { current: 0, total: totalImages }
  
  try {
    // 导入公共保存函数
    const { saveAllPagesSequentially, saveSessionMeta } = await import('@/api/pageStorage')
    
    // 构建 UI 设置
    const { textStyle, targetLanguage, sourceLanguage } = settingsStore.settings
    const uiSettings: Record<string, unknown> = {
      targetLanguage,
      sourceLanguage,
      fontSize: textStyle.fontSize,
      autoFontSize: textStyle.autoFontSize,
      fontFamily: textStyle.fontFamily,
      layoutDirection: textStyle.layoutDirection,
      textColor: textStyle.textColor,
      fillColor: textStyle.fillColor,
      useInpaintingMethod: textStyle.inpaintMethod,
      strokeEnabled: textStyle.strokeEnabled,
      strokeColor: textStyle.strokeColor,
      strokeWidth: textStyle.strokeWidth,
      useAutoTextColor: textStyle.useAutoTextColor,
    }

    // 使用公共函数逐页保存
    const savedCount = await saveAllPagesSequentially(
      sessionName.value,
      imageStore.images as unknown as import('@/api/pageStorage').ImageDataForSave[],
      {
        onProgress: (current, total) => {
          saveProgress.value = { current, total }
        }
      }
    )

    // 更新会话元数据
    await saveSessionMeta(sessionName.value, {
      ui_settings: uiSettings,
      total_pages: totalImages,
      currentImageIndex: imageStore.currentImageIndex
    })

    showToast(`会话保存成功 (${savedCount}/${totalImages})`, 'success')
    sessionStore.setSessionName(sessionName.value)
    emit('session-saved', sessionName.value)
    handleClose()
  } catch (error) {
    console.error('保存会话失败:', error)
    showToast('保存会话失败', 'error')
  } finally {
    isSaving.value = false
    saveProgress.value = { current: 0, total: 0 }
  }
}

/**
 * 加载会话
 */
async function handleLoad() {
  if (!selectedSession.value) return

  isLoading.value = true
  try {
    const response = await loadSession(selectedSession.value)
    if (response.success && response.session) {
      showToast('会话加载成功', 'success')
      sessionStore.setSessionName(selectedSession.value)
      emit('session-loaded', response.session)
      handleClose()
    } else {
      showToast(response.error || '加载失败', 'error')
    }
  } catch (error) {
    console.error('加载会话失败:', error)
    showToast('加载会话失败', 'error')
  } finally {
    isLoading.value = false
  }
}

/**
 * 开始重命名
 */
function startRename(session: SessionListItem) {
  sessionToRename.value = session
  newSessionName.value = session.name
  showRenameDialog.value = true
}

/**
 * 取消重命名
 */
function cancelRename() {
  showRenameDialog.value = false
  sessionToRename.value = null
  newSessionName.value = ''
}

/**
 * 执行重命名
 */
async function handleRename() {
  if (!sessionToRename.value || !newSessionName.value.trim()) return

  try {
    const response = await renameSession(sessionToRename.value.name, newSessionName.value)
    if (response.success) {
      showToast('重命名成功', 'success')
      // 更新本地列表
      const oldName = sessionToRename.value?.name
      const index = sessionList.value.findIndex(s => s.name === oldName)
      if (index >= 0 && sessionList.value[index]) {
        sessionList.value[index].name = newSessionName.value
      }
      // 如果当前选中的是被重命名的会话，更新选中状态
      if (selectedSession.value === sessionToRename.value.name) {
        selectedSession.value = newSessionName.value
      }
      cancelRename()
    } else {
      showToast(response.error || '重命名失败', 'error')
    }
  } catch (error) {
    console.error('重命名失败:', error)
    showToast('重命名失败', 'error')
  }
}

/**
 * 确认删除
 */
function confirmDelete(session: SessionListItem) {
  sessionToDelete.value = session
  showDeleteConfirm.value = true
}

/**
 * 取消删除
 */
function cancelDelete() {
  showDeleteConfirm.value = false
  sessionToDelete.value = null
}

/**
 * 执行删除
 */
async function handleDelete() {
  if (!sessionToDelete.value) return

  try {
    const response = await deleteSession(sessionToDelete.value.name)
    if (response.success) {
      showToast('删除成功', 'success')
      // 从列表中移除
      sessionList.value = sessionList.value.filter(s => s.name !== sessionToDelete.value?.name)
      // 如果删除的是当前选中的会话，清除选中状态
      if (selectedSession.value === sessionToDelete.value.name) {
        selectedSession.value = null
      }
      cancelDelete()
    } else {
      showToast(response.error || '删除失败', 'error')
    }
  } catch (error) {
    console.error('删除失败:', error)
    showToast('删除失败', 'error')
  }
}

/**
 * 关闭模态框
 */
function handleClose() {
  isVisible.value = false
  sessionName.value = ''
  selectedSession.value = null
}

/**
 * 格式化日期
 */
function formatDate(dateStr: string): string {
  try {
    const date = new Date(dateStr)
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch {
    return dateStr
  }
}
</script>

<style scoped>
/* 选项卡 */
.session-tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--border-color, #e0e0e0);
  padding-bottom: 12px;
}

.tab-btn {
  padding: 8px 16px;
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 14px;
  color: var(--text-secondary, #666);
  border-radius: 4px;
  transition: all 0.2s ease;
}

.tab-btn:hover {
  background: var(--hover-bg, #f5f5f5);
}

.tab-btn.active {
  background: var(--primary-color, #4a90d9);
  color: white;
}

/* 面板 */
.session-panel {
  min-height: 200px;
}

/* 表单 */
.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  margin-bottom: 6px;
  font-weight: 500;
  color: var(--text-color, #2c3e50);
}

.form-input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  font-size: 14px;
  transition: border-color 0.2s ease;
}

.form-input:focus {
  outline: none;
  border-color: var(--primary-color, #4a90d9);
}

.form-hint {
  margin-bottom: 16px;
  font-size: 13px;
  color: var(--text-secondary, #666);
}

.form-hint .warning {
  color: var(--warning-color, #f0ad4e);
}

/* 保存进度条 */
.save-progress {
  margin-bottom: 16px;
  padding: 12px;
  background: var(--bg-secondary, #f5f7fa);
  border-radius: 6px;
}

.save-progress .progress-text {
  font-size: 13px;
  color: var(--text-secondary, #666);
  margin-bottom: 8px;
  text-align: center;
}

.save-progress .progress-bar {
  height: 8px;
  background: var(--border-color, #e0e0e0);
  border-radius: 4px;
  overflow: hidden;
}

.save-progress .progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #4caf50, #8bc34a);
  border-radius: 4px;
  transition: width 0.3s ease;
}

/* 会话列表 */
.session-list {
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 6px;
  margin-bottom: 16px;
}

.session-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px;
  border-bottom: 1px solid var(--border-color, #e0e0e0);
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.session-item:last-child {
  border-bottom: none;
}

.session-item:hover {
  background: var(--hover-bg, #f5f5f5);
}

.session-item.selected {
  background: var(--selected-bg, #e3f2fd);
}

.session-info {
  flex: 1;
}

.session-name {
  font-weight: 500;
  color: var(--text-color, #2c3e50);
  margin-bottom: 4px;
}

.session-meta {
  display: flex;
  gap: 12px;
  font-size: 12px;
  color: var(--text-secondary, #888);
}

.session-actions {
  display: flex;
  gap: 4px;
}

.action-btn {
  padding: 4px 8px;
  border: none;
  background: transparent;
  cursor: pointer;
  border-radius: 4px;
  transition: background-color 0.2s ease;
}

.action-btn:hover {
  background: var(--hover-bg, #e0e0e0);
}

.action-btn.danger:hover {
  background: var(--danger-bg, #ffebee);
}

/* 操作按钮 */
.action-buttons {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s ease;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: var(--primary-color, #4a90d9);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: var(--primary-hover, #3a7bc8);
}

.btn-secondary {
  background: var(--secondary-bg, #e0e0e0);
  color: var(--text-color, #2c3e50);
}

.btn-secondary:hover:not(:disabled) {
  background: var(--secondary-hover, #d0d0d0);
}

.btn-danger {
  background: var(--danger-color, #dc3545);
  color: white;
}

.btn-danger:hover:not(:disabled) {
  background: var(--danger-hover, #c82333);
}

/* 加载状态 */
.loading-state {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 40px;
  color: var(--text-secondary, #666);
}

.loading-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid var(--border-color, #e0e0e0);
  border-top-color: var(--primary-color, #4a90d9);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

/* @keyframes spin 已迁移到 global.css */

/* 空状态 */
.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
  color: var(--text-secondary, #888);
  border: 1px dashed var(--border-color, #e0e0e0);
  border-radius: 6px;
  margin-bottom: 16px;
}

/* 对话框覆盖层 */
.rename-dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgb(0, 0, 0, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1100;
}

.rename-dialog {
  background: var(--modal-bg, #fff);
  padding: 20px;
  border-radius: 8px;
  min-width: 300px;
  box-shadow: 0 4px 12px rgb(0, 0, 0, 0.15);
}

.rename-dialog h4 {
  margin: 0 0 16px;
  color: var(--text-color, #2c3e50);
}

.rename-dialog p {
  margin: 0 0 16px;
  color: var(--text-secondary, #666);
}

.rename-dialog .form-input {
  margin-bottom: 16px;
}

.dialog-buttons {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
}
</style>
