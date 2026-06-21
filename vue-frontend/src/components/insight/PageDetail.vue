<script setup lang="ts">

import UiButton from '@/components/ui/UiButton.vue'
/**
 * 页面详情组件
 * 显示选中页面的详细信息，包括图片、摘要和对话
 * 支持上一页/下一页导航、重新分析、图片预览等功能
 */

import { ref, computed, watch } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'

// ============================================================
// 状态
// ============================================================

const insightStore = useInsightStore()

/** 页面分析数据 */
const pageAnalysis = ref<{
  page_num?: number
  page_summary?: string
  scene?: string
  mood?: string
  analyzed?: boolean
  panels?: Array<{
    dialogues?: Array<{
      speaker_name?: string
      character?: string
      text?: string
      translated_text?: string
    }>
  }>
} | null>(null)

/** 是否正在加载 */
const isLoading = ref(false)

/** 是否正在重新分析 */
const isReanalyzing = ref(false)

/** 是否等待异步重分析任务完成 */
const pendingReanalyzePage = ref<number | null>(null)

/** 是否显示图片预览 */
const showImagePreview = ref(false)

/** 错误消息 */
const errorMessage = ref('')

// ============================================================
// 计算属性
// ============================================================

/** 当前选中的页码 */
const selectedPageNum = computed(() => insightStore.selectedPageNum)

/** 总页数 */
const totalPages = computed(() => insightStore.totalPageCount)

/** 是否有上一页 */
const hasPrevPage = computed(() => {
  return selectedPageNum.value !== null && selectedPageNum.value > 1
})

/** 是否有下一页 */
const hasNextPage = computed(() => {
  return selectedPageNum.value !== null && selectedPageNum.value < totalPages.value
})

/** 页面图片URL */
const pageImageUrl = computed(() => {
  if (!insightStore.currentBookId || !selectedPageNum.value) return ''
  return insightApi.getPageImageUrl(insightStore.currentBookId, selectedPageNum.value)
})

/** 对话列表 */
const dialogues = computed(() => {
  if (!pageAnalysis.value?.panels) return []
  const result: Array<{ speaker: string; text: string; originalText?: string }> = []
  for (const panel of pageAnalysis.value.panels) {
    if (panel.dialogues) {
      for (const d of panel.dialogues) {
        // 优先使用译文，其次使用原文
        const text = d.translated_text || d.text
        if (text) {
          result.push({
            speaker: d.speaker_name || d.character || '未知',
            text: text,
            originalText: d.text !== d.translated_text ? d.text : undefined
          })
        }
      }
    }
  }
  return result
})

/** 页面是否已分析 */
const isPageAnalyzed = computed(() => {
  return pageAnalysis.value?.analyzed === true || !!pageAnalysis.value?.page_summary
})

/** 场景描述 */
const sceneDescription = computed(() => pageAnalysis.value?.scene || '')

/** 氛围/情绪 */
const moodDescription = computed(() => pageAnalysis.value?.mood || '')

/** 当前页是否存在进行中的重分析任务 */
const isReanalyzeTaskRunning = computed(() => {
  return (
    pendingReanalyzePage.value !== null &&
    pendingReanalyzePage.value === selectedPageNum.value &&
    insightStore.analysisStatus === 'running'
  )
})

// ============================================================
// 方法
// ============================================================

/**
 * 加载页面详情
 */
async function loadPageDetail(): Promise<void> {
  if (!insightStore.currentBookId || !selectedPageNum.value) {
    pageAnalysis.value = null
    return
  }

  isLoading.value = true
  errorMessage.value = ''

  try {
    const response = await insightApi.getPageData(
      insightStore.currentBookId, 
      selectedPageNum.value
    )
    
    if (response.success) {
      // 后端API返回的是analysis字段，不是page字段
      if (response.analysis) {
        pageAnalysis.value = response.analysis as any
      } else if (response.page) {
        pageAnalysis.value = response.page as any
      } else {
        pageAnalysis.value = null
      }
    } else {
      pageAnalysis.value = null
      if (response.error) {
        errorMessage.value = response.error
      }
    }
  } catch (error) {
    console.error('加载页面详情失败:', error)
    pageAnalysis.value = null
    errorMessage.value = error instanceof Error ? error.message : '加载失败'
  } finally {
    isLoading.value = false
  }
}

/**
 * 导航到上一页
 */
function navigatePrev(): void {
  if (hasPrevPage.value && selectedPageNum.value) {
    insightStore.selectPage(selectedPageNum.value - 1)
  }
}

/**
 * 导航到下一页
 */
function navigateNext(): void {
  if (hasNextPage.value && selectedPageNum.value) {
    insightStore.selectPage(selectedPageNum.value + 1)
  }
}

/**
 * 重新分析当前页面
 */
async function reanalyzePage(): Promise<void> {
  if (!insightStore.currentBookId || !selectedPageNum.value) return

  isReanalyzing.value = true
  errorMessage.value = ''

  try {
    const response = await insightApi.reanalyzePage(
      insightStore.currentBookId, 
      selectedPageNum.value
    )
    
    if (response.success) {
      if ((response as any).task_id) {
        insightStore.setCurrentTaskId((response as any).task_id)
      }
      pendingReanalyzePage.value = selectedPageNum.value
      insightStore.setAnalysisStatus('running')
    } else {
      errorMessage.value = response.error || '重新分析失败'
    }
  } catch (error) {
    console.error('重新分析失败:', error)
    const message = (error as { message?: string })?.message
    errorMessage.value = message || '重新分析失败'
  } finally {
    isReanalyzing.value = false
  }
}

/**
 * 打开图片预览
 */
function openImagePreview(): void {
  showImagePreview.value = true
}

/**
 * 关闭图片预览
 */
function closeImagePreview(): void {
  showImagePreview.value = false
}

/**
 * 处理键盘事件（图片预览模式）
 */
function handlePreviewKeydown(event: KeyboardEvent): void {
  if (!showImagePreview.value) return
  
  switch (event.key) {
    case 'Escape':
      closeImagePreview()
      break
    case 'ArrowLeft':
      if (hasPrevPage.value) {
        navigatePrev()
      }
      break
    case 'ArrowRight':
      if (hasNextPage.value) {
        navigateNext()
      }
      break
  }
}

/** 是否正在导出 */
const isExporting = ref(false)

/**
 * 导出当前页面分析数据为 Markdown 文件
 */
async function exportPageData(): Promise<void> {
  if (!insightStore.currentBookId || !selectedPageNum.value || !pageAnalysis.value) {
    return
  }

  isExporting.value = true

  try {
    // 构建 Markdown 内容
    let markdown = `# 第 ${selectedPageNum.value} 页分析数据\n\n`
    
    // 页面摘要
    if (pageAnalysis.value.page_summary) {
      markdown += `## 📝 页面摘要\n\n${pageAnalysis.value.page_summary}\n\n`
    }
    
    // 场景和氛围
    if (pageAnalysis.value.scene) {
      markdown += `## 🎬 场景\n\n${pageAnalysis.value.scene}\n\n`
    }
    if (pageAnalysis.value.mood) {
      markdown += `## 🎭 氛围\n\n${pageAnalysis.value.mood}\n\n`
    }
    
    // 对话内容
    if (dialogues.value.length > 0) {
      markdown += `## 💬 对话内容\n\n`
      for (const d of dialogues.value) {
        markdown += `**${d.speaker}**: ${d.text}\n\n`
        if (d.originalText) {
          markdown += `> 原文: ${d.originalText}\n\n`
        }
      }
    }

    // 下载文件
    const blob = new Blob([markdown], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${insightStore.currentBookId}_page_${selectedPageNum.value}.md`
    a.click()
    URL.revokeObjectURL(url)

  } catch (error) {
    console.error('导出页面数据失败:', error)
    errorMessage.value = '导出失败'
  } finally {
    isExporting.value = false
  }
}

// ============================================================
// 监听
// ============================================================

// 监听选中页码变化
watch(selectedPageNum, () => {
  loadPageDetail()
}, { immediate: true })

// 分析完成后自动刷新当前页详情（依赖全局轮询触发 dataRefreshKey）
watch(() => insightStore.dataRefreshKey, async (newKey) => {
  if (newKey <= 0 || !selectedPageNum.value) return

  if (pendingReanalyzePage.value !== null) {
    pendingReanalyzePage.value = null
  }
  await loadPageDetail()
})
</script>

<template>
  <div class="workspace-section page-detail-section">
    <h3 class="section-title">📄 页面详情</h3>
    
    <div class="page-detail">
      <!-- 未选择页面 -->
      <div v-if="!selectedPageNum" class="placeholder-text">
        <div class="empty-icon">📄</div>
        <p>点击左侧导航树中的页面查看详情</p>
      </div>
      
      <!-- 加载中 -->
      <div v-else-if="isLoading" class="loading-state">
        <div class="loading-spinner"></div>
        <p>加载中...</p>
      </div>
      
      <!-- 页面详情内容 -->
      <div v-else class="page-detail-content">
        <!-- 页面标题和导航 -->
        <div class="page-detail-header">
          <h4>📄 第 {{ selectedPageNum }} 页</h4>
          <div class="page-nav-buttons">
            <UiButton
              variant="toolbar" 
              class="btn-page-nav"
              :class="{ disabled: !hasPrevPage }"
              :disabled="!hasPrevPage"
              title="上一页 (←)"
              @click="navigatePrev"
            >
              ◀ 上一张
            </UiButton>
            <span class="page-indicator">{{ selectedPageNum }} / {{ totalPages }}</span>
            <UiButton
              variant="toolbar" 
              class="btn-page-nav"
              :class="{ disabled: !hasNextPage }"
              :disabled="!hasNextPage"
              title="下一页 (→)"
              @click="navigateNext"
            >
              下一张 ▶
            </UiButton>
          </div>
        </div>
        
        <!-- 错误消息 -->
        <div v-if="errorMessage" class="error-message">
          ⚠️ {{ errorMessage }}
        </div>
        
        <!-- 页面图片 -->
        <div class="page-detail-image" @click="openImagePreview">
          <img 
            :src="pageImageUrl" 
            :alt="`第${selectedPageNum}页`"
            @error="($event.target as HTMLImageElement).style.display = 'none'"
          >
          <div class="image-overlay">
            <span class="zoom-hint">🔍 点击放大</span>
          </div>
        </div>
        
        <!-- 分析状态标签 -->
        <div class="analysis-status-tag" :class="{ analyzed: isPageAnalyzed }">
          {{ isPageAnalyzed ? '✓ 已分析' : '○ 未分析' }}
        </div>
        
        <!-- 页面摘要 -->
        <div v-if="pageAnalysis?.page_summary" class="page-summary">
          <h5>📝 页面摘要</h5>
          <p>{{ pageAnalysis.page_summary }}</p>
        </div>
        <div v-else class="page-summary empty">
          <p>此页尚未分析，点击下方按钮开始分析</p>
        </div>
        
        <!-- 场景和氛围 -->
        <div v-if="sceneDescription || moodDescription" class="scene-mood-info">
          <div v-if="sceneDescription" class="info-item">
            <span class="info-label">🎬 场景：</span>
            <span class="info-value">{{ sceneDescription }}</span>
          </div>
          <div v-if="moodDescription" class="info-item">
            <span class="info-label">🎭 氛围：</span>
            <span class="info-value">{{ moodDescription }}</span>
          </div>
        </div>
        
        <!-- 对话列表 -->
        <div v-if="dialogues.length > 0" class="dialogues-section">
          <h5>💬 对话内容 ({{ dialogues.length }})</h5>
          <div 
            v-for="(dialogue, index) in dialogues" 
            :key="index"
            class="dialogue-item"
          >
            <div class="dialogue-speaker">
              <span class="speaker-icon">👤</span>
              {{ dialogue.speaker }}
            </div>
            <div class="dialogue-text">{{ dialogue.text }}</div>
            <div v-if="dialogue.originalText" class="dialogue-original">
              <span class="original-label">原文：</span>{{ dialogue.originalText }}
            </div>
          </div>
        </div>
        <div v-else-if="isPageAnalyzed" class="dialogues-section empty">
          <p>此页没有检测到对话内容</p>
        </div>
        
        <!-- 操作按钮 -->
        <div class="page-detail-actions">
          <UiButton
            variant="secondary" 
            
            :disabled="isReanalyzing || isReanalyzeTaskRunning"
            :class="{ loading: isReanalyzing || isReanalyzeTaskRunning }"
            @click="reanalyzePage" size="sm"
          >
            <span v-if="isReanalyzing || isReanalyzeTaskRunning" class="btn-spinner"></span>
            {{ isReanalyzing ? '启动中...' : (isReanalyzeTaskRunning ? '分析中...' : '🔄 重新分析') }}
          </UiButton>
          <UiButton
            variant="secondary" 
            v-if="isPageAnalyzed"
            
            :disabled="isExporting"
            @click="exportPageData" size="sm"
          >
            {{ isExporting ? '导出中...' : '📄 导出此页' }}
          </UiButton>
        </div>
      </div>
    </div>
    
    <!-- 图片预览模态框 -->
    <div 
      v-if="showImagePreview" 
      class="image-preview-modal"
      tabindex="0"
      @click="closeImagePreview"
      @keydown="handlePreviewKeydown"
    >
      <div class="image-preview-content" @click.stop>
        <UiButton variant="toolbar" class="preview-close" title="关闭 (Esc)" @click="closeImagePreview">&times;</UiButton>
        <img :src="pageImageUrl" :alt="`第${selectedPageNum}页`">
        <!-- 预览模式导航 -->
        <div class="preview-nav">
          <UiButton
            variant="toolbar" 
            class="preview-nav-btn prev"
            :disabled="!hasPrevPage"
            title="上一页 (←)"
            @click.stop="navigatePrev"
          >
            ◀
          </UiButton>
          <span class="preview-page-info">{{ selectedPageNum }} / {{ totalPages }}</span>
          <UiButton
            variant="toolbar" 
            class="preview-nav-btn next"
            :disabled="!hasNextPage"
            title="下一页 (→)"
            @click.stop="navigateNext"
          >
            ▶
          </UiButton>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>/* ==================== PageDetail样式 ==================== */

/* ==================== 按钮样式 ==================== */
.page-detail-section .ui-button {
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

.page-detail-section .ui-button--primary {
  background: var(--insight-color-primary);
  color: white;
}

.page-detail-section .ui-button--primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.page-detail-section .ui-button--primary:hover:not(:disabled) {
  background: var(--insight-primary-dark);
}

.page-detail-section .ui-button--secondary {
  background: var(--insight-bg-tertiary);
  color: var(--insight-text-primary);
  border: 1px solid var(--color-border-muted);
}

.page-detail-section .ui-button--secondary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.page-detail-section .ui-button--secondary:hover:not(:disabled) {
  background: var(--color-border-muted);
}

.page-detail-section .ui-button--sm {
  padding: 8px 14px;
  font-size: 13px;
}

/* ==================== 组件特定样式 ==================== */

.workspace-section.page-detail-section {
  padding: 20px 18px;
}

/* 空状态 */
.page-detail-section .placeholder-text {
  text-align: center;
  padding: 24px;
  color: var(--insight-text-secondary);
}

.page-detail-section .placeholder-text p {
  max-width: 220px;
  margin: 0 auto;
}

.page-detail-section .empty-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

/* 加载状态 */
.page-detail-section .loading-state {
  text-align: center;
  padding: 24px;
  color: var(--insight-text-secondary);
}

.page-detail-section .loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid var(--color-border-muted);
  border-top-color: var(--insight-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto 12px;
}

.page-detail-section .page-detail-header h4 {
  margin: 0;
  font-size: 16px;
}

.page-detail-section .page-nav-buttons {
  display: flex;
  align-items: center;
  gap: 8px;
}

.page-detail-section .btn-page-nav {
  padding: 4px 12px;
  font-size: 12px;
  border: 1px solid var(--color-border-muted);
  border-radius: 4px;
  background: var(--insight-bg-secondary);
  cursor: pointer;
  transition: all 0.2s;
}

.page-detail-section .btn-page-nav:hover:not(.disabled) {
  background: var(--insight-bg-hover);
  border-color: var(--insight-primary);
}

.page-detail-section .btn-page-nav.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-detail-section .page-indicator {
  font-size: 12px;
  color: var(--insight-text-secondary);
  min-width: 60px;
  text-align: center;
}

/* 错误消息 */
.page-detail-section .error-message {
  font-size: 12px;
  color: var(--page-detail-text-primary);
  background: var(--page-detail-surface-base);
  padding: 8px 12px;
  border-radius: 4px;
  margin-bottom: 12px;
}

/* 页面图片 */
.page-detail-section .page-detail-image {
  position: relative;
  margin-bottom: 12px;
  cursor: pointer;
  border-radius: 4px;
  overflow: hidden;
}

.page-detail-section .page-detail-image img {
  max-width: 100%;
  display: block;
  border-radius: 4px;
}

.page-detail-section .image-overlay {
  position: absolute;
  inset: 0;
  background: var(--page-detail-surface-raised);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
}

.page-detail-section .page-detail-image:hover .image-overlay {
  background: var(--page-detail-surface-muted);
}

.page-detail-section .zoom-hint {
  color: white;
  font-size: 14px;
  opacity: 0;
  transition: opacity 0.2s;
}

.page-detail-section .page-detail-image:hover .zoom-hint {
  opacity: 1;
}

/* 分析状态标签 */
.page-detail-section .analysis-status-tag {
  display: inline-block;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  background: var(--insight-bg-secondary);
  color: var(--insight-text-secondary);
  margin-bottom: 12px;
}

.page-detail-section .analysis-status-tag.analyzed {
  background: var(--page-detail-surface-subtle);
  color: var(--page-detail-text-secondary);
}

/* 页面摘要 */
.page-detail-section .page-summary {
  margin-bottom: 16px;
}

.page-detail-section .page-summary h5 {
  font-size: 14px;
  margin: 0 0 8px;
  color: var(--insight-text-primary);
}

.page-detail-section .page-summary p {
  font-size: 14px;
  line-height: 1.6;
  color: var(--insight-text-secondary);
  margin: 0;
}

.page-detail-section .page-summary.empty p {
  font-style: italic;
}

/* 场景和氛围信息 */
.page-detail-section .scene-mood-info {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 16px;
  padding: 10px;
  background: var(--insight-bg-secondary);
  border-radius: 6px;
}

.page-detail-section .info-item {
  font-size: 13px;
}

.page-detail-section .info-label {
  color: var(--insight-text-secondary);
}

.page-detail-section .info-value {
  color: var(--insight-text-primary);
}

/* 对话部分 */
.page-detail-section .dialogues-section {
  margin-bottom: 16px;
}

.page-detail-section .dialogues-section h5 {
  font-size: 14px;
  margin: 0 0 12px;
  color: var(--insight-text-primary);
}

.page-detail-section .dialogues-section.empty p {
  font-size: 13px;
  color: var(--insight-text-secondary);
  font-style: italic;
}

.page-detail-section .dialogue-item {
  padding: 10px 12px;
  margin: 8px 0;
  background: var(--insight-bg-secondary);
  border-radius: 8px;
  border-left: 3px solid var(--insight-primary);
}

.page-detail-section .dialogue-speaker {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
  font-size: 12px;
  color: var(--insight-primary);
  margin-bottom: 6px;
}

.page-detail-section .speaker-icon {
  font-size: 14px;
}

.page-detail-section .dialogue-text {
  font-size: 14px;
  line-height: 1.6;
  color: var(--insight-text-primary);
}

.page-detail-section .dialogue-original {
  font-size: 12px;
  color: var(--insight-text-secondary);
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px dashed var(--color-border-muted);
}

.page-detail-section .original-label {
  font-weight: 500;
}

/* 操作按钮 */
.page-detail-section .page-detail-actions {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border-muted);
}

.page-detail-section .ui-button.loading {
  opacity: 0.7;
  cursor: wait;
}

.page-detail-section .btn-spinner {
  display: inline-block;
  width: 12px;
  height: 12px;
  border: 2px solid currentcolor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-right: 6px;
}

/* 图片预览模态框 */
.page-detail-section .image-preview-modal {
  position: fixed;
  inset: 0;
  background: var(--page-detail-surface-hover);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: var(--z-overlay);
  outline: none;
}

.page-detail-section .image-preview-content {
  position: relative;
  max-width: 90vw;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.page-detail-section .image-preview-content img {
  max-width: 100%;
  max-height: calc(90vh - 60px);
  object-fit: contain;
}

.page-detail-section .preview-close {
  position: absolute;
  top: -45px;
  right: 0;
  background: none;
  border: none;
  color: white;
  font-size: 36px;
  cursor: pointer;
  padding: 5px 10px;
  transition: transform 0.2s;
}

.page-detail-section .preview-close:hover {
  transform: scale(1.1);
}

/* 预览导航 */
.page-detail-section .preview-nav {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 16px;
}

.page-detail-section .preview-nav-btn {
  width: 40px;
  height: 40px;
  border: none;
  border-radius: 50%;
  background: var(--page-detail-surface-active);
  color: white;
  font-size: 18px;
  cursor: pointer;
  transition: all 0.2s;
}

.page-detail-section .preview-nav-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.page-detail-section .preview-nav-btn:hover:not(:disabled) {
  background: var(--page-detail-surface-selected);
}

.page-detail-section .preview-page-info {
  color: white;
  font-size: 14px;
  min-width: 80px;
  text-align: center;
}
</style>
