<script setup lang="ts">
/**
 * 网页导入模态框
 * 核心功能界面：URL输入 → 提取 → 预览 → 下载 → 导入
 */
import { ref, computed, watch } from 'vue'
import { useWebImportStore } from '@/stores/webImportStore'
import { useImageStore } from '@/stores/imageStore'
import { extractImages, downloadImages } from '@/api/webImport'
import type { AgentLog, ExtractResult } from '@/types/webImport'

const webImportStore = useWebImportStore()
const imageStore = useImageStore()

// 本地状态
const urlInput = ref('')
const logsExpanded = ref(true)

// 计算属性
const isVisible = computed(() => webImportStore.modalVisible)
const status = computed(() => webImportStore.status)
const logs = computed(() => webImportStore.logs)
const extractResult = computed(() => webImportStore.extractResult)
const selectedPages = computed(() => webImportStore.selectedPages)
const selectedCount = computed(() => webImportStore.selectedCount)
const downloadProgress = computed(() => webImportStore.downloadProgress)
const downloadProgressPercent = computed(() => webImportStore.downloadProgressPercent)
const error = computed(() => webImportStore.error)
const isProcessing = computed(() => webImportStore.isProcessing)
const showAgentLogs = computed(() => webImportStore.settings.ui.showAgentLogs)

// 是否全选
const isAllSelected = computed(() => {
  if (!extractResult.value?.pages) return false
  return selectedCount.value === extractResult.value.pages.length
})

// 关闭模态框
function handleClose() {
  if (isProcessing.value) {
    if (!confirm('正在处理中，确定要关闭吗？')) return
  }
  webImportStore.closeModal()
  webImportStore.resetState()
  urlInput.value = ''
}

// 开始提取
async function handleExtract() {
  const url = urlInput.value.trim()
  if (!url) {
    alert('请输入网址')
    return
  }

  // 验证 URL
  try {
    new URL(url)
  } catch {
    alert('请输入有效的网址')
    return
  }

  // 重置状态
  webImportStore.resetState()
  webImportStore.setUrl(url)
  webImportStore.setStatus('extracting')

  try {
    await extractImages(
      url,
      webImportStore.settings,
      (log: AgentLog) => {
        webImportStore.addLog(log)
      },
      (result: ExtractResult) => {
        webImportStore.setExtractResult(result)
        if (result.success) {
          webImportStore.setStatus('extracted')
        } else {
          webImportStore.setError(result.error || '提取失败')
        }
      },
      (errorMsg: string) => {
        webImportStore.setError(errorMsg)
      }
    )
  } catch (e) {
    webImportStore.setError(e instanceof Error ? e.message : '提取失败')
  }
}

// 切换页面选择
function togglePage(pageNumber: number) {
  webImportStore.togglePageSelection(pageNumber)
}

// 全选/取消全选
function toggleAll() {
  webImportStore.toggleSelectAll()
}

// 开始下载并导入
async function handleImport() {
  if (!extractResult.value?.pages || selectedCount.value === 0) {
    alert('请选择要导入的图片')
    return
  }

  // 获取选中的页面
  const selectedPagesList = extractResult.value.pages.filter((p) =>
    selectedPages.value.has(p.pageNumber)
  )

  webImportStore.setStatus('downloading')
  webImportStore.updateDownloadProgress(0, selectedPagesList.length)

  try {
    const result = await downloadImages(
      selectedPagesList,
      extractResult.value.sourceUrl,
      webImportStore.settings
    )

    if (result.success && result.images.length > 0) {
      webImportStore.setDownloadedImages(result.images)
      webImportStore.updateDownloadProgress(result.images.length, selectedPagesList.length)

      // 导入到 imageStore (参数顺序: fileName, dataUrl)
      for (const img of result.images) {
        imageStore.addImage(img.filename, img.dataUrl)
      }

      webImportStore.setStatus('completed')

      // 提示成功
      const failedMsg = result.failedCount > 0 ? `，${result.failedCount} 张失败` : ''
      alert(`成功导入 ${result.images.length} 张图片${failedMsg}`)

      // 关闭模态框
      handleClose()
    } else {
      webImportStore.setError(result.error || '下载失败')
    }
  } catch (e) {
    webImportStore.setError(e instanceof Error ? e.message : '下载失败')
  }
}

// 监听模态框打开时聚焦输入框
watch(isVisible, (visible) => {
  if (visible) {
    setTimeout(() => {
      const input = document.querySelector('.url-input') as HTMLInputElement
      input?.focus()
    }, 100)
  }
})
</script>

<template>
  <Teleport to="body">
    <div v-if="isVisible" class="modal-overlay" @click.self="handleClose">
      <div class="modal-container">
        <!-- 头部 -->
        <div class="modal-header">
          <h2 class="modal-title">
            <span class="title-icon">🌐</span>
            从网页导入漫画
          </h2>
          <button class="close-btn" @click="handleClose" title="关闭">×</button>
        </div>

        <!-- 内容 -->
        <div class="modal-body">
          <!-- URL 输入 -->
          <div class="url-section">
            <input
              v-model="urlInput"
              type="url"
              class="url-input"
              placeholder="输入漫画网页 URL，如 https://example.com/chapter-1"
              :disabled="isProcessing"
              @keyup.enter="handleExtract"
            />
            <button
              class="extract-btn"
              :disabled="isProcessing || !urlInput.trim()"
              @click="handleExtract"
            >
              <span v-if="status === 'extracting'" class="loading-spinner"></span>
              <span v-else>🔍</span>
              {{ status === 'extracting' ? '提取中...' : '开始提取' }}
            </button>
          </div>

          <!-- 使用须知 -->
          <div class="notice">
            ⚠️ 请仅爬取您有权访问的内容，并遵守目标网站的使用条款。
          </div>

          <!-- AI 工作日志 -->
          <div v-if="showAgentLogs && logs.length > 0" class="logs-section">
            <div class="logs-header" @click="logsExpanded = !logsExpanded">
              <span class="logs-toggle">{{ logsExpanded ? '▼' : '▶' }}</span>
              <span>AI 工作日志</span>
              <span v-if="status === 'extracting'" class="extracting-hint">(提取中...)</span>
            </div>
            <div v-if="logsExpanded" class="logs-content">
              <div
                v-for="(log, index) in logs"
                :key="index"
                class="log-item"
                :class="`log-${log.type}`"
              >
                <span class="log-time">[{{ log.timestamp }}]</span>
                <span class="log-message">{{ log.message }}</span>
              </div>
            </div>
          </div>

          <!-- 错误提示 -->
          <div v-if="error" class="error-section">
            <span class="error-icon">❌</span>
            <span class="error-message">{{ error }}</span>
          </div>

          <!-- 提取结果 -->
          <div v-if="extractResult?.success" class="result-section">
            <div class="result-header">
              <span class="result-title">
                📖 《{{ extractResult.comicTitle }}》- {{ extractResult.chapterTitle }}
              </span>
              <span class="result-count">共 {{ extractResult.totalPages }} 张</span>
            </div>

            <!-- 选择控制 -->
            <div class="select-control">
              <label class="select-all">
                <input
                  type="checkbox"
                  :checked="isAllSelected"
                  @change="toggleAll"
                />
                全选
              </label>
              <span class="selected-count">已选: {{ selectedCount }} 张</span>
            </div>

            <!-- 图片网格 -->
            <div class="image-grid">
              <div
                v-for="page in extractResult.pages"
                :key="page.pageNumber"
                class="image-item"
                :class="{ selected: selectedPages.has(page.pageNumber) }"
                @click="togglePage(page.pageNumber)"
              >
                <div class="image-checkbox">
                  <input
                    type="checkbox"
                    :checked="selectedPages.has(page.pageNumber)"
                    @click.stop
                    @change="togglePage(page.pageNumber)"
                  />
                </div>
                <div class="image-preview">
                  <img :src="page.imageUrl" :alt="`第${page.pageNumber}页`" loading="lazy" />
                </div>
                <div class="image-label">第 {{ page.pageNumber }} 页</div>
              </div>
            </div>
          </div>

          <!-- 下载进度 -->
          <div v-if="status === 'downloading'" class="progress-section">
            <div class="progress-label">
              下载进度: {{ downloadProgress.current }}/{{ downloadProgress.total }}
            </div>
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: `${downloadProgressPercent}%` }"></div>
            </div>
          </div>
        </div>

        <!-- 底部 -->
        <div class="modal-footer">
          <button class="cancel-btn" @click="handleClose" :disabled="status === 'downloading'">
            取消
          </button>
          <button
            class="import-btn"
            :disabled="!extractResult?.success || selectedCount === 0 || isProcessing"
            @click="handleImport"
          >
            <span v-if="status === 'downloading'" class="loading-spinner"></span>
            <span v-else>📥</span>
            {{ status === 'downloading' ? '下载中...' : '导入' }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-container {
  background: var(--bg-primary, #fff);
  border-radius: 12px;
  width: 90%;
  max-width: 800px;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-color, #eee);
}

.modal-title {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-primary, #333);
}

.title-icon {
  font-size: 22px;
}

.close-btn {
  width: 32px;
  height: 32px;
  border: none;
  background: transparent;
  font-size: 24px;
  cursor: pointer;
  border-radius: 6px;
  color: var(--text-secondary, #666);
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-btn:hover {
  background: var(--bg-secondary, #f5f5f5);
}

.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.url-section {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
}

.url-input {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 8px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.url-input:focus {
  border-color: var(--primary-color, #4a90d9);
}

.extract-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 18px;
  background: var(--btn-primary-bg, #4a90d9);
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
  transition: background 0.2s;
}

.extract-btn:hover:not(:disabled) {
  background: var(--btn-primary-hover-bg, #3a7fc8);
}

.extract-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.notice {
  padding: 10px 14px;
  background: #fff8e6;
  border: 1px solid #ffe0a0;
  border-radius: 6px;
  font-size: 13px;
  color: #856404;
  margin-bottom: 16px;
}

.logs-section {
  margin-bottom: 16px;
  border: 1px solid var(--border-color, #eee);
  border-radius: 8px;
  overflow: hidden;
}

.logs-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: var(--bg-secondary, #f9f9f9);
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  user-select: none;
}

.logs-toggle {
  font-size: 10px;
  color: var(--text-secondary, #888);
}

.extracting-hint {
  color: var(--primary-color, #4a90d9);
  font-weight: normal;
  font-size: 13px;
}

.logs-content {
  max-height: 200px;
  overflow-y: auto;
  padding: 12px;
  background: #1e1e1e;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 12px;
}

.log-item {
  padding: 2px 0;
  color: #ccc;
}

.log-time {
  color: #888;
  margin-right: 8px;
}

.log-info .log-message { color: #9cdcfe; }
.log-tool_call .log-message { color: #dcdcaa; }
.log-tool_result .log-message { color: #6a9955; }
.log-thinking .log-message { color: #ce9178; }
.log-error .log-message { color: #f14c4c; }

.error-section {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 14px;
  background: #fff5f5;
  border: 1px solid #ffc0c0;
  border-radius: 6px;
  margin-bottom: 16px;
  color: #c00;
}

.result-section {
  margin-bottom: 16px;
}

.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.result-title {
  font-size: 15px;
  font-weight: 500;
  color: var(--text-primary, #333);
}

.result-count {
  font-size: 13px;
  color: var(--text-secondary, #666);
}

.select-control {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 12px;
}

.select-all {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  font-size: 14px;
}

.selected-count {
  font-size: 13px;
  color: var(--text-secondary, #666);
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 12px;
  max-height: 300px;
  overflow-y: auto;
  padding: 4px;
}

.image-item {
  position: relative;
  border: 2px solid var(--border-color, #eee);
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.2s;
}

.image-item:hover {
  border-color: var(--primary-color, #4a90d9);
}

.image-item.selected {
  border-color: var(--primary-color, #4a90d9);
  box-shadow: 0 0 0 2px rgba(74, 144, 217, 0.2);
}

.image-checkbox {
  position: absolute;
  top: 6px;
  left: 6px;
  z-index: 1;
}

.image-preview {
  width: 100%;
  aspect-ratio: 3/4;
  background: var(--bg-secondary, #f5f5f5);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.image-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-label {
  padding: 6px;
  text-align: center;
  font-size: 12px;
  color: var(--text-secondary, #666);
  background: var(--bg-primary, #fff);
}

.progress-section {
  margin-bottom: 16px;
}

.progress-label {
  font-size: 13px;
  color: var(--text-secondary, #666);
  margin-bottom: 8px;
}

.progress-bar {
  height: 8px;
  background: var(--bg-secondary, #eee);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--primary-color, #4a90d9);
  transition: width 0.3s ease;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 16px 20px;
  border-top: 1px solid var(--border-color, #eee);
}

.cancel-btn,
.import-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.cancel-btn {
  background: var(--btn-secondary-bg, #f0f0f0);
  border: 1px solid var(--border-color, #ddd);
  color: var(--text-primary, #333);
}

.cancel-btn:hover:not(:disabled) {
  background: var(--btn-secondary-hover-bg, #e5e5e5);
}

.import-btn {
  background: var(--btn-primary-bg, #4a90d9);
  border: none;
  color: #fff;
}

.import-btn:hover:not(:disabled) {
  background: var(--btn-primary-hover-bg, #3a7fc8);
}

.import-btn:disabled,
.cancel-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.loading-spinner {
  width: 14px;
  height: 14px;
  border: 2px solid transparent;
  border-top-color: currentColor;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
