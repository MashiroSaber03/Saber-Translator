<script setup lang="ts">
import './QAPanel.global.styles.css'

import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
/**
 * 智能问答面板组件
 * 提供基于漫画内容的问答功能，支持流式响应
 */

import { ref, computed, nextTick, onMounted, onUnmounted } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import { marked } from 'marked'
import * as insightApi from '@/api/insight'
import BaseModal from '@/components/common/BaseModal.vue'
import { useQANoteModal } from './useQANoteModal'

// ============================================================
// 状态
// ============================================================

const insightStore = useInsightStore()

/** 问题输入 */
const questionInput = ref('')

/** 问答模式：精确模式或全局模式 */
const qaMode = ref<'precise' | 'global'>('precise')

/** 精确模式选项 */
const useParentChild = ref(true)
const useReasoning = ref(true)
const useReranker = ref(true)
const topK = ref(5)
const threshold = ref(0)
const isRebuildingEmbeddings = ref(false)
const rebuildTaskId = ref<string | null>(null)
const rebuildProgressLabel = ref('')
const rebuildPollingFailures = ref(0)
let rebuildPollingTimer: ReturnType<typeof setInterval> | null = null

/** 消息容器引用 */
const messagesContainer = ref<HTMLElement | null>(null)

// ============================================================
// 计算属性
// ============================================================

/** 问答历史 */
const qaHistory = computed(() => insightStore.qaHistory)

/** 是否正在流式响应 */
const isStreaming = computed(() => insightStore.isStreaming)

/** 是否显示精确模式选项 */
const showPreciseModeOptions = computed(() => qaMode.value === 'precise')

const {
  closeNoteModal,
  noteComment,
  noteTitle,
  openNoteModal,
  pendingQAData,
  saveNote,
  showNoteModal,
} = useQANoteModal(insightStore)

// ============================================================
// 方法
// ============================================================

/**
 * 设置问答模式
 * @param mode - 模式
 */
function setQAMode(mode: 'precise' | 'global'): void {
  qaMode.value = mode
}

/**
 * 发送问题
 */
async function sendQuestion(): Promise<void> {
  const question = questionInput.value.trim()
  if (!question || !insightStore.currentBookId) return
  if (isStreaming.value) return

  // 清空输入
  questionInput.value = ''

  // 清空之前的问答内容（单轮对话模式，与当前行为一致）
  insightStore.clearQAHistory()

  // 添加用户消息
  insightStore.addQAMessage({
    id: Date.now().toString(),
    role: 'user',
    content: question,
    timestamp: new Date().toISOString()
  })

  // 滚动到底部
  await nextTick()
  scrollToBottom()

  // 添加加载消息
  const loadingText = qaMode.value === 'global' ? '正在分析全文...' : '思考中...'
  insightStore.addQAMessage({
    id: (Date.now() + 1).toString(),
    role: 'assistant',
    content: loadingText,
    timestamp: new Date().toISOString(),
    isLoading: true
  })

  insightStore.setStreaming(true)

  try {
    // 使用API封装
    const response = await insightApi.sendChat(insightStore.currentBookId, question, {
      use_parent_child: useParentChild.value,
      use_reasoning: useReasoning.value,
      use_reranker: useReranker.value,
      top_k: topK.value,
      threshold: threshold.value,
      use_global_context: qaMode.value === 'global'
    })

    // 移除加载消息
    insightStore.removeLoadingMessages()

    if (response.success) {
      // 构建回答内容
      const modeLabel = response.mode === 'global' ? '🌐 全局模式' : '🎯 精确模式'
      
      // 添加助手回答
      insightStore.addQAMessage({
        id: (Date.now() + 2).toString(),
        role: 'assistant',
        content: response.answer || '',
        timestamp: new Date().toISOString(),
        mode: modeLabel,
        citations: response.citations || []
      })
    } else {
      insightStore.addQAMessage({
        id: (Date.now() + 2).toString(),
        role: 'assistant',
        content: '抱歉，处理问题时出错: ' + (response.error || '未知错误'),
        timestamp: new Date().toISOString()
      })
    }

  } catch (error) {
    console.error('问答请求失败:', error)
    insightStore.removeLoadingMessages()
    insightStore.addQAMessage({
      id: (Date.now() + 2).toString(),
      role: 'assistant',
      content: '抱歉，网络请求失败，请稍后重试。',
      timestamp: new Date().toISOString()
    })
  } finally {
    insightStore.setStreaming(false)
    await nextTick()
    scrollToBottom()
  }
}

/**
 * 处理键盘事件
 * @param event - 键盘事件
 */
function handleKeydown(event: KeyboardEvent): void {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    sendQuestion()
  }
}

/**
 * 滚动到底部
 */
function scrollToBottom(): void {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

/**
 * 重建向量索引
 */
async function rebuildEmbeddings(): Promise<void> {
  if (!insightStore.currentBookId) return
  if (isRebuildingEmbeddings.value) return
  if (!confirm('确定要重建向量索引吗？\n\n这将删除现有的向量数据并重新构建，可能需要一些时间。')) return

  insightStore.setLoading(true)
  isRebuildingEmbeddings.value = true
  rebuildProgressLabel.value = '准备启动...'

  try {
    const response = await insightApi.rebuildEmbeddings(insightStore.currentBookId)

    if (!response.success || !response.task_id) {
      alert('重建失败: ' + (response.error || '未知错误'))
      isRebuildingEmbeddings.value = false
      rebuildProgressLabel.value = ''
      insightStore.setLoading(false)
      return
    }

    rebuildTaskId.value = response.task_id
    rebuildProgressLabel.value = '任务已启动'
    rebuildPollingFailures.value = 0
    startRebuildStatusPolling(response.task_id)
  } catch (error) {
    console.error('重建向量索引失败:', error)
    const message = error instanceof Error ? error.message : '重建向量索引失败'
    alert(message)
    isRebuildingEmbeddings.value = false
    rebuildProgressLabel.value = ''
    insightStore.setLoading(false)
  }
}

function stopRebuildStatusPolling(): void {
  if (rebuildPollingTimer) {
    clearInterval(rebuildPollingTimer)
    rebuildPollingTimer = null
  }
}

async function pollRebuildStatus(taskId: string): Promise<void> {
  if (!insightStore.currentBookId) return

  const response = await insightApi.getRebuildEmbeddingsStatus(insightStore.currentBookId, taskId)
  const task = response.task
  if (!task) {
    stopRebuildStatusPolling()
    isRebuildingEmbeddings.value = false
    insightStore.setLoading(false)
    rebuildTaskId.value = null
    rebuildProgressLabel.value = ''
    alert('重建失败: 未找到向量重建任务状态')
    return
  }

  rebuildPollingFailures.value = 0

  const progress = task.progress
  if (task.status === 'running' || task.status === 'pending') {
    const phaseText = progress?.current_phase || '重建中'
    const current = progress?.analyzed_pages ?? 0
    const total = progress?.total_pages ?? 0
    rebuildProgressLabel.value = total > 0 ? `${phaseText} (${current}/${total})` : phaseText
  }

  if (task.status === 'completed') {
    stopRebuildStatusPolling()
    isRebuildingEmbeddings.value = false
    insightStore.setLoading(false)
    rebuildTaskId.value = null
    rebuildProgressLabel.value = ''

    let message = '向量索引重建完成'
    if (response.stats) {
      message += `\n页面向量: ${response.stats.pages_count || 0} 条`
      if (response.stats.events_count !== undefined) {
        message += `\n事件向量: ${response.stats.events_count || 0} 条`
      }
    }
    alert(message)
    return
  }

  if (task.status === 'failed' || task.status === 'cancelled') {
    stopRebuildStatusPolling()
    isRebuildingEmbeddings.value = false
    insightStore.setLoading(false)
    rebuildTaskId.value = null
    rebuildProgressLabel.value = ''
    alert('重建失败: ' + (task.error_message || response.error || '未知错误'))
  }
}

function startRebuildStatusPolling(taskId: string): void {
  stopRebuildStatusPolling()
  rebuildPollingTimer = setInterval(() => {
    void pollRebuildStatus(taskId).catch((error) => {
      console.error('轮询向量重建状态失败:', error)
      rebuildPollingFailures.value += 1
      if (rebuildPollingFailures.value >= 3) {
        stopRebuildStatusPolling()
        isRebuildingEmbeddings.value = false
        insightStore.setLoading(false)
        rebuildTaskId.value = null
        rebuildProgressLabel.value = ''
        alert('重建失败: 无法获取任务状态，请稍后查看结果')
      }
    })
  }, 3000)
}

/**
 * 渲染 Markdown 内容
 * @param content - Markdown 文本
 */
function renderMarkdown(content: string): string {
  if (!content) return ''
  return marked.parse(content) as string
}

/**
 * 选择页面（跳转到指定页面）
 * @param pageNum - 页码
 */
function selectPage(pageNum: number): void {
  insightStore.setCurrentPage(pageNum)
}

/**
 * 示例问题列表（全局模式）
 */
const globalModeExamples = [
  '故事的主题是什么？',
  '主角的性格有什么变化？',
  '结局是怎样的？'
]

/**
 * 点击示例问题
 * @param question - 示例问题
 */
function askExampleQuestion(question: string): void {
  questionInput.value = question
  sendQuestion()
}

// ============================================================
// 生命周期
// ============================================================

onMounted(() => {
  // 初始化时滚动到底部
  scrollToBottom()
})

onUnmounted(() => {
  stopRebuildStatusPolling()
})
</script>

<template>
  <div class="qa-container">
    <!-- 消息列表 -->
    <div ref="messagesContainer" class="chat-messages">
      <!-- 欢迎消息 -->
      <div v-if="qaHistory.length === 0" class="welcome-message">
        <div class="welcome-icon">💬</div>
        <h3>智能问答</h3>
        <p>针对已分析的漫画内容提问，获取精准回答</p>
      </div>
      
      <!-- 消息列表 -->
      <div 
        v-for="message in qaHistory" 
        :key="message.id"
        class="chat-message"
        :class="message.role"
      >
        <!-- 头像 -->
        <div class="message-avatar">
          <template v-if="message.role === 'user'">
            <img src="/pic/logo.png" alt="用户" class="avatar-img">
          </template>
          <template v-else>
            🤖
          </template>
        </div>
        <!-- 消息内容 -->
        <div v-if="message.role === 'user'" class="message-content">
          {{ message.content }}
        </div>
        <div v-else class="message-content markdown-content">
          <!-- 加载状态 -->
          <div v-if="message.isLoading" class="loading-dots">
            {{ message.content }}
          </div>
          <template v-else>
            <!-- 模式标识 -->
            <div v-if="message.mode" class="answer-mode-badge">{{ message.mode }}</div>
            <!-- 回答文本（使用v-html渲染Markdown） -->
            <div class="answer-text" v-html="renderMarkdown(message.content)"></div>
            <!-- 引用 -->
            <div v-if="message.citations && message.citations.length > 0" class="message-citations">
              <span>📖 引用: </span>
              <span 
                v-for="citation in message.citations" 
                :key="citation.page"
                class="citation-item"
                @click="selectPage(citation.page)"
              >
                第{{ citation.page }}页
              </span>
            </div>
            <!-- 保存为笔记按钮 -->
            <UiButton
              variant="toolbar" 
              v-if="message.content && !message.isLoading"
              class="message-save-btn"
              :class="{ saved: message.saved }"
              :disabled="message.saved"
              @click="openNoteModal(message)"
            >
              {{ message.saved ? '✅ 已保存' : '📝 保存为笔记' }}
            </UiButton>
          </template>
        </div>
      </div>
    </div>
    
    <!-- 输入区域 -->
    <div class="chat-input-container">
      <!-- 选项栏 -->
      <div class="chat-options">
        <!-- 问答模式切换 -->
        <div class="qa-mode-toggle" title="精确模式：使用RAG检索相关片段；全局模式：使用全文摘要">
          <UiButton
            variant="toolbar" 
            type="button" 
            class="qa-mode-btn"
            :class="{ active: qaMode === 'precise' }"
            @click="setQAMode('precise')"
          >
            🎯 精确模式
          </UiButton>
          <UiButton
            variant="toolbar" 
            type="button" 
            class="qa-mode-btn"
            :class="{ active: qaMode === 'global' }"
            @click="setQAMode('global')"
          >
            🌐 全局模式
          </UiButton>
        </div>
        
        <span class="chat-option-divider">|</span>
        
        <!-- 精确模式选项 -->
        <div v-if="showPreciseModeOptions" class="precise-mode-options">
          <label class="ui-checkbox-label compact" title="启用父子块模式">
            <UiInput v-model="useParentChild" type="checkbox" />
            <span>父子块模式</span>
          </label>
          <label class="ui-checkbox-label compact" title="启用推理检索">
            <UiInput v-model="useReasoning" type="checkbox" />
            <span>推理检索</span>
          </label>
          <label class="ui-checkbox-label compact" title="启用重排序">
            <UiInput v-model="useReranker" type="checkbox" />
            <span>重排序</span>
          </label>
          <span class="chat-option-divider">|</span>
          <label class="input-label compact" title="返回的最大结果数">
            <span>Top K:</span>
            <UiInput v-model.number="topK" type="number" min="1" max="20" class="input-small" />
          </label>
          <label class="input-label compact" title="相关性阈值">
            <span>阈值:</span>
            <UiInput v-model.number="threshold" type="number" min="0" max="1" step="0.1" class="input-small" />
          </label>
          <span class="chat-option-divider">|</span>
          <UiButton
            variant="secondary" 
            type="button" 
            
            title="重建向量索引"
            :disabled="isRebuildingEmbeddings"
            @click="rebuildEmbeddings" size="sm"
          >
            {{ isRebuildingEmbeddings ? `⏳ ${rebuildProgressLabel || '重建中...'}` : '🔄 重建向量' }}
          </UiButton>
        </div>
        
        <!-- 全局模式提示 -->
        <div v-else class="global-mode-hint">
          <span class="hint-text">💡 全局模式使用全文摘要回答，适合总结性问题</span>
          <div class="welcome-examples">
            <span 
              v-for="(example, index) in globalModeExamples" 
              :key="index"
              class="example-tag" 
              @click="askExampleQuestion(example)"
            >
              {{ example }}
            </span>
          </div>
        </div>
      </div>
      
      <!-- 输入框 -->
      <div class="chat-input-wrapper">
        <UiTextarea 
          v-model="questionInput"
          placeholder="输入你的问题..." 
          rows="1"
          :disabled="isStreaming"
          @keydown="handleKeydown"
        />
        <UiButton
          variant="toolbar" 
          class="send-btn" 
          :disabled="isStreaming || !questionInput.trim()"
          @click="sendQuestion"
        >
          <span>发送</span>
        </UiButton>
      </div>
    </div>
    
    <BaseModal
      v-model="showNoteModal"
      title="📝 添加笔记"
      size="medium"
      custom-class="qa-note-modal"
      :custom-style="{ width: '90%', maxWidth: '560px' }"
      @close="closeNoteModal"
    >
      <div class="qa-note-modal-body">
        <!-- 问答预览 -->
        <div v-if="pendingQAData" class="qa-preview">
          <div class="qa-preview-section">
            <label>问题</label>
            <div class="qa-preview-content">{{ pendingQAData.question }}</div>
          </div>
          <div class="qa-preview-section">
            <label>回答</label>
            <div class="qa-preview-content" v-html="renderMarkdown(pendingQAData.answer)"></div>
          </div>
          <div v-if="pendingQAData.citations.length > 0" class="qa-preview-section">
            <label>引用页码</label>
            <div class="qa-preview-citations">
              <span 
                v-for="citation in pendingQAData.citations" 
                :key="citation.page"
                class="qa-citation-badge"
              >
                第{{ citation.page }}页
              </span>
            </div>
          </div>
        </div>
        <!-- 笔记表单 -->
        <div class="note-form">
          <div class="qa-note-modal__field">
            <label for="qaNoteTitle">笔记标题 <span class="optional">(可选)</span></label>
            <UiInput 
              id="qaNoteTitle"
              v-model="noteTitle"
              type="text" 
              class="qa-note-modal__form-input" 
              placeholder="默认使用问题作为标题..."
            />
          </div>
          <div class="qa-note-modal__field">
            <label for="qaNoteComment">补充说明 <span class="optional">(可选)</span></label>
            <UiTextarea 
              id="qaNoteComment"
              v-model="noteComment"
              class="qa-note-modal__form-textarea" 
              rows="3" 
              placeholder="添加你的评论或补充..."
            />
          </div>
        </div>
      </div>

      <template #footer>
        <UiButton variant="secondary" @click="closeNoteModal">取消</UiButton>
        <UiButton variant="primary" @click="saveNote">保存笔记</UiButton>
      </template>
    </BaseModal>
  </div>
</template>

<style scoped>/* ==================== 问答面板样式 ==================== */

/* ==================== 组件样式 ==================== */
.qa-container {
    display: flex;
    flex-direction: column;
    height: 100%;
}

.qa-container .chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.qa-container .welcome-message {
    text-align: center;
    padding: 40px 20px;
}

.qa-container .welcome-icon {
    font-size: 48px;
    margin-bottom: 16px;
}

.qa-container .welcome-message h3 {
    margin-bottom: 8px;
}

.qa-container .welcome-message p {
    color: var(--insight-text-secondary);
    margin-bottom: 20px;
}

.qa-container .suggested-questions {
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: center;
}

.qa-container .suggestion-btn {
    padding: 10px 16px;
    background: var(--insight-bg-secondary);
    border: 1px solid var(--color-border-muted);
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 13px;
}

.qa-container .suggestion-btn:hover {
    background: var(--insight-bg-tertiary);
    border-color: var(--insight-color-primary);
}

.qa-container .chat-message {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    animation: slideIn 0.3s ease;
}

.qa-container .chat-message.user {
    flex-direction: row-reverse;
}

.qa-container .message-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}

.qa-container .chat-message.user .message-avatar {
    background: transparent;
    overflow: hidden;
}

.qa-container .message-avatar .avatar-img {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: cover;
    display: block;
}

.qa-container .chat-message.assistant .message-avatar {
    background: var(--insight-bg-tertiary);
}

.qa-container .message-content {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.6;
}

.qa-container .chat-message.user .message-content {
    background: var(--insight-color-primary);
    color: white;
    border-bottom-right-radius: 4px;
}

.qa-container .chat-message.assistant .message-content {
    background: var(--insight-bg-secondary);
    border: 1px solid var(--color-border-muted);
    border-bottom-left-radius: 4px;
}

.qa-container .chat-message.assistant .message-content.markdown-content {
    max-width: 70%;
}

.qa-container .chat-message.assistant .answer-text {
    line-height: 1.7;
}

.qa-container .chat-message.assistant .answer-text p {
    margin: 0 0 8px;
}

.qa-container .chat-message.assistant .answer-text p:last-child {
    margin-bottom: 0;
}

.qa-container .chat-message.assistant .answer-text ul,
.qa-container .chat-message.assistant .answer-text ol {
    margin: 8px 0;
    padding-left: 20px;
}

.qa-container .chat-message.assistant .answer-text li {
    margin: 4px 0;
}

.qa-container .chat-message.assistant .answer-text strong {
    color: var(--insight-color-primary);
    font-weight: 600;
}

.qa-container .chat-message.assistant .answer-text blockquote {
    margin: 8px 0;
    padding: 6px 12px;
    border-left: 3px solid var(--insight-color-primary);
    background: var(--insight-bg-tertiary);
    border-radius: 0 6px 6px 0;
    font-style: italic;
}

.qa-container .chat-message.assistant .answer-text blockquote p {
    margin: 0;
}

.qa-container .chat-input-container {
    padding: 16px;
    border-top: 1px solid var(--color-border-muted);
    background: var(--insight-bg-secondary);
}

.qa-container .chat-options {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--color-border-muted);
}

.qa-container .chat-options .ui-checkbox-label.compact {
    font-size: 13px;
    color: var(--insight-text-secondary);
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
}

.qa-container .chat-options .ui-checkbox-label.compact:hover {
    color: var(--insight-text-primary);
}

.qa-container .chat-options .ui-checkbox-label.compact input[type="checkbox"] {
    width: 16px;
    height: 16px;
    cursor: pointer;
}

.qa-container .chat-option-divider {
    color: var(--color-border-muted);
    margin: 0 4px;
}

.qa-container .qa-mode-toggle {
    display: flex;
    background: var(--insight-bg-secondary);
    border-radius: 8px;
    padding: 2px;
    gap: 2px;
}

.qa-container .qa-mode-btn {
    padding: 6px 12px;
    border: none;
    background: transparent;
    color: var(--insight-text-secondary);
    font-size: 13px;
    cursor: pointer;
    border-radius: 6px;
    transition: all 0.2s;
    white-space: nowrap;
}

.qa-container .qa-mode-btn:hover {
    color: var(--insight-text-primary);
    background: var(--insight-bg-tertiary);
}

.qa-container .qa-mode-btn.active {
    background: var(--insight-color-primary);
    color: white;
    font-weight: 500;
}

.qa-container .precise-mode-options {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
}

.qa-container .global-mode-hint {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.qa-container .global-mode-hint .hint-text {
    font-size: 13px;
    color: var(--insight-text-secondary);
    font-style: italic;
}

.qa-container .answer-mode-badge {
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--insight-bg-secondary);
    color: var(--insight-text-secondary);
    margin-bottom: 8px;
}

.qa-container .welcome-examples {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
    justify-content: center;
}

.qa-container .example-tag {
    padding: 6px 12px;
    background: var(--insight-bg-secondary);
    border-radius: 16px;
    font-size: 13px;
    color: var(--insight-text-secondary);
    cursor: pointer;
    transition: all 0.2s;
}

.qa-container .example-tag:hover {
    background: var(--insight-color-primary);
    color: white;
}

.qa-container .chat-options .input-label.compact {
    font-size: 13px;
    color: var(--insight-text-secondary);
    display: flex;
    align-items: center;
    gap: 4px;
}

.qa-container .chat-options .input-small {
    width: 50px;
    padding: 2px 6px;
    border: 1px solid var(--color-border-muted);
    border-radius: 4px;
    font-size: 12px;
    background: var(--insight-bg-primary);
    color: var(--insight-text-primary);
}

.qa-container .chat-input-wrapper {
    display: flex;
    gap: 12px;
    align-items: flex-end;
}

.qa-container .chat-input-wrapper textarea {
    flex: 1;
    padding: 12px 16px;
    border: 1px solid var(--color-border-muted);
    border-radius: 12px;
    resize: none;
    font-size: 14px;
    background: var(--insight-bg-primary);
    color: var(--insight-text-primary);
    max-height: 120px;
}

.qa-container .send-btn {
    padding: 12px 24px;
    background: var(--insight-color-primary);
    color: white;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    font-weight: 500;
    transition: background 0.2s;
}

.qa-container .send-btn:hover {
    background: var(--insight-primary-dark);
}

.qa-container .send-btn:disabled {
    background: var(--insight-text-muted);
    cursor: not-allowed;
}

.qa-container .message-citations {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--color-border-muted);
    font-size: 12px;
    color: var(--insight-text-secondary);
}

.qa-container .citation-item {
    display: inline-block;
    padding: 2px 8px;
    background: var(--insight-bg-tertiary);
    border-radius: 4px;
    margin: 2px 4px;
    cursor: pointer;
}

.qa-container .citation-item:hover {
    background: var(--insight-color-primary);
    color: white;
}

.qa-container .message-save-btn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    margin-top: 12px;
    background: var(--insight-bg-tertiary);
    border: 1px solid var(--color-border-muted);
    border-radius: 6px;
    color: var(--insight-text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s;
}

.qa-container .message-save-btn:hover {
    background: var(--insight-color-primary);
    color: white;
    border-color: var(--insight-color-primary);
}

.qa-container .message-save-btn.saved {
    background: var(--insight-success-color);
    color: white;
    border-color: var(--insight-success-color);
    cursor: default;
}

/* 加载动画 */
.qa-container .loading-dots {
  display: inline-block;
  color: var(--insight-text-secondary);
}

.qa-container .loading-dots::after {
  content: '';
  animation: dots 1.5s steps(4, end) infinite;
}

@keyframes dots {
  0%, 20% { content: ''; }
  40% { content: '.'; }
  60% { content: '..'; }
  80%, 100% { content: '...'; }
}

</style>
