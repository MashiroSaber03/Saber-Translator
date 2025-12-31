<script setup lang="ts">
/**
 * 智能问答面板组件
 * 提供基于漫画内容的问答功能，支持流式响应
 */

import { ref, computed, nextTick, onMounted } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import { marked } from 'marked'
import * as insightApi from '@/api/insight'

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

  // 清空之前的问答内容（单轮对话模式，与原版一致）
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
  if (!confirm('确定要重建向量索引吗？\n\n这将删除现有的向量数据并重新构建，可能需要一些时间。')) return

  insightStore.setLoading(true)

  try {
    const response = await insightApi.rebuildEmbeddings(insightStore.currentBookId)
    
    if (response.success) {
      let message = '向量索引重建完成'
      if (response.stats) {
        message += `\n页面向量: ${response.stats.pages_count || 0} 条`
        if (response.stats.dialogues_count) {
          message += `\n对话向量: ${response.stats.dialogues_count} 条`
        }
      }
      alert(message)
    } else {
      alert('重建失败: ' + (response.error || '未知错误'))
    }
  } catch (error) {
    console.error('重建向量索引失败:', error)
    alert('重建向量索引失败')
  } finally {
    insightStore.setLoading(false)
  }
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
// 笔记弹窗相关
// ============================================================

/** 笔记弹窗状态 */
const showNoteModal = ref(false)
/** 当前待保存的问答数据 */
const pendingQAData = ref<{
  messageId: string
  question: string
  answer: string
  citations: Array<{ page: number }>
} | null>(null)
/** 笔记标题 */
const noteTitle = ref('')
/** 笔记补充说明 */
const noteComment = ref('')

/**
 * 打开笔记弹窗
 * @param message - 助手消息
 */
function openNoteModal(message: { id: string; content: string; citations?: Array<{ page: number }> }): void {
  if (!insightStore.currentBookId) return
  
  // 获取最后一个用户问题
  const userMessage = insightStore.qaHistory.find(m => m.role === 'user')
  const question = userMessage?.content || ''
  
  pendingQAData.value = {
    messageId: message.id,
    question,
    answer: message.content,
    citations: message.citations || []
  }
  
  noteTitle.value = ''
  noteComment.value = ''
  showNoteModal.value = true
}

/**
 * 关闭笔记弹窗
 */
function closeNoteModal(): void {
  showNoteModal.value = false
  pendingQAData.value = null
}

/**
 * 保存笔记
 */
async function saveNote(): Promise<void> {
  if (!insightStore.currentBookId || !pendingQAData.value) return
  
  const now = new Date().toISOString()
  const noteData = {
    id: Date.now().toString(),
    type: 'qa' as const,
    title: noteTitle.value || pendingQAData.value.question.substring(0, 30),
    content: pendingQAData.value.answer,
    question: pendingQAData.value.question,
    answer: pendingQAData.value.answer,
    citations: pendingQAData.value.citations,
    comment: noteComment.value || undefined,
    createdAt: now,
    updatedAt: now
  }
  
  try {
    // 添加到本地store
    await insightStore.addNote(noteData)
    
    // 更新消息的保存状态
    const message = insightStore.qaHistory.find(m => m.id === pendingQAData.value?.messageId)
    if (message) {
      message.saved = true
    }
    
    closeNoteModal()
  } catch (error) {
    console.error('保存笔记失败:', error)
    alert('保存笔记失败')
  }
}

// ============================================================
// 生命周期
// ============================================================

onMounted(() => {
  // 初始化时滚动到底部
  scrollToBottom()
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
            <button 
              v-if="message.content && !message.isLoading"
              class="message-save-btn"
              :class="{ saved: message.saved }"
              :disabled="message.saved"
              @click="openNoteModal(message)"
            >
              {{ message.saved ? '✅ 已保存' : '📝 保存为笔记' }}
            </button>
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
          <button 
            type="button" 
            class="qa-mode-btn"
            :class="{ active: qaMode === 'precise' }"
            @click="setQAMode('precise')"
          >
            🎯 精确模式
          </button>
          <button 
            type="button" 
            class="qa-mode-btn"
            :class="{ active: qaMode === 'global' }"
            @click="setQAMode('global')"
          >
            🌐 全局模式
          </button>
        </div>
        
        <span class="chat-option-divider">|</span>
        
        <!-- 精确模式选项 -->
        <div v-if="showPreciseModeOptions" class="precise-mode-options">
          <label class="checkbox-label compact" title="启用父子块模式">
            <input v-model="useParentChild" type="checkbox">
            <span>父子块模式</span>
          </label>
          <label class="checkbox-label compact" title="启用推理检索">
            <input v-model="useReasoning" type="checkbox">
            <span>推理检索</span>
          </label>
          <label class="checkbox-label compact" title="启用重排序">
            <input v-model="useReranker" type="checkbox">
            <span>重排序</span>
          </label>
          <span class="chat-option-divider">|</span>
          <label class="input-label compact" title="返回的最大结果数">
            <span>Top K:</span>
            <input v-model.number="topK" type="number" min="1" max="20" class="input-small">
          </label>
          <label class="input-label compact" title="相关性阈值">
            <span>阈值:</span>
            <input v-model.number="threshold" type="number" min="0" max="1" step="0.1" class="input-small">
          </label>
          <span class="chat-option-divider">|</span>
          <button 
            type="button" 
            class="btn btn-sm btn-secondary" 
            title="重建向量索引"
            @click="rebuildEmbeddings"
          >
            🔄 重建向量
          </button>
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
        <textarea 
          v-model="questionInput"
          placeholder="输入你的问题..." 
          rows="1"
          :disabled="isStreaming"
          @keydown="handleKeydown"
        ></textarea>
        <button 
          class="send-btn" 
          :disabled="isStreaming || !questionInput.trim()"
          @click="sendQuestion"
        >
          <span>发送</span>
        </button>
      </div>
    </div>
    
    <!-- 笔记弹窗 -->
    <div v-if="showNoteModal" class="modal note-modal show">
      <div class="modal-overlay" @click="closeNoteModal"></div>
      <div class="modal-content note-modal-content">
        <div class="modal-header">
          <h2>📝 添加笔记</h2>
          <button class="modal-close" @click="closeNoteModal">&times;</button>
        </div>
        <div class="modal-body">
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
            <div class="form-group">
              <label for="qaNoteTitle">笔记标题 <span class="optional">(可选)</span></label>
              <input 
                v-model="noteTitle"
                type="text" 
                id="qaNoteTitle" 
                class="form-input" 
                placeholder="默认使用问题作为标题..."
              >
            </div>
            <div class="form-group">
              <label for="qaNoteComment">补充说明 <span class="optional">(可选)</span></label>
              <textarea 
                v-model="noteComment"
                id="qaNoteComment" 
                class="form-textarea" 
                rows="3" 
                placeholder="添加你的评论或补充..."
              ></textarea>
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-secondary" @click="closeNoteModal">取消</button>
          <button class="btn btn-primary" @click="saveNote">保存笔记</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ==================== 问答面板完整样式 ==================== */

/* ==================== CSS变量 ==================== */
.qa-container {
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

/* ==================== 组件样式 ==================== */
.qa-container {
    display: flex;
    flex-direction: column;
    height: 100%;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.welcome-message {
    text-align: center;
    padding: 40px 20px;
}

.welcome-icon {
    font-size: 48px;
    margin-bottom: 16px;
}

.welcome-message h3 {
    margin-bottom: 8px;
}

.welcome-message p {
    color: var(--text-secondary);
    margin-bottom: 20px;
}

.suggested-questions {
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: center;
}

.suggestion-btn {
    padding: 10px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 13px;
}

.suggestion-btn:hover {
    background: var(--bg-tertiary);
    border-color: var(--primary-color);
}

.chat-message {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    animation: slideIn 0.3s ease;
}

.chat-message.user {
    flex-direction: row-reverse;
}

.message-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}

.chat-message.user .message-avatar {
    background: transparent;
    overflow: hidden;
}

.message-avatar .avatar-img {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: cover;
    display: block;
}

.chat-message.assistant .message-avatar {
    background: var(--bg-tertiary);
}

.message-content {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.6;
}

.chat-message.user .message-content {
    background: var(--primary-color);
    color: white;
    border-bottom-right-radius: 4px;
}

.chat-message.assistant .message-content {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-bottom-left-radius: 4px;
}

.chat-message.assistant .message-content.markdown-content {
    max-width: 70%;
}

.chat-message.assistant .answer-text {
    line-height: 1.7;
}

.chat-message.assistant .answer-text p {
    margin: 0 0 8px 0;
}

.chat-message.assistant .answer-text p:last-child {
    margin-bottom: 0;
}

.chat-message.assistant .answer-text ul,
.chat-message.assistant .answer-text ol {
    margin: 8px 0;
    padding-left: 20px;
}

.chat-message.assistant .answer-text li {
    margin: 4px 0;
}

.chat-message.assistant .answer-text strong {
    color: var(--primary-color);
    font-weight: 600;
}

.chat-message.assistant .answer-text blockquote {
    margin: 8px 0;
    padding: 6px 12px;
    border-left: 3px solid var(--primary-color);
    background: var(--bg-tertiary);
    border-radius: 0 6px 6px 0;
    font-style: italic;
}

.chat-message.assistant .answer-text blockquote p {
    margin: 0;
}

.chat-input-container {
    padding: 16px;
    border-top: 1px solid var(--border-color);
    background: var(--bg-secondary);
}

.chat-options {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-color);
}

.chat-options .checkbox-label.compact {
    font-size: 13px;
    color: var(--text-secondary);
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
}

.chat-options .checkbox-label.compact:hover {
    color: var(--text-primary);
}

.chat-options .checkbox-label.compact input[type="checkbox"] {
    width: 16px;
    height: 16px;
    cursor: pointer;
}

.chat-option-divider {
    color: var(--border-color);
    margin: 0 4px;
}

.qa-mode-toggle {
    display: flex;
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 2px;
    gap: 2px;
}

.qa-mode-btn {
    padding: 6px 12px;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    border-radius: 6px;
    transition: all 0.2s;
    white-space: nowrap;
}

.qa-mode-btn:hover {
    color: var(--text-primary);
    background: var(--bg-tertiary);
}

.qa-mode-btn.active {
    background: var(--primary-color);
    color: white;
    font-weight: 500;
}

.precise-mode-options {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
}

.global-mode-hint {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.global-mode-hint .hint-text {
    font-size: 13px;
    color: var(--text-secondary);
    font-style: italic;
}

.answer-mode-badge {
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--bg-secondary);
    color: var(--text-secondary);
    margin-bottom: 8px;
}

.welcome-examples {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
    justify-content: center;
}

.example-tag {
    padding: 6px 12px;
    background: var(--bg-secondary);
    border-radius: 16px;
    font-size: 13px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s;
}

.example-tag:hover {
    background: var(--primary-color);
    color: white;
}

.chat-options .input-label.compact {
    font-size: 13px;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 4px;
}

.chat-options .input-small {
    width: 50px;
    padding: 2px 6px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-size: 12px;
    background: var(--bg-primary);
    color: var(--text-primary);
}

.chat-input-wrapper {
    display: flex;
    gap: 12px;
    align-items: flex-end;
}

.chat-input-wrapper textarea {
    flex: 1;
    padding: 12px 16px;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    resize: none;
    font-size: 14px;
    background: var(--bg-primary);
    color: var(--text-primary);
    max-height: 120px;
}

.send-btn {
    padding: 12px 24px;
    background: var(--primary-color);
    color: white;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    font-weight: 500;
    transition: background 0.2s;
}

.send-btn:hover {
    background: var(--primary-dark);
}

.send-btn:disabled {
    background: var(--text-muted);
    cursor: not-allowed;
}

.message-citations {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border-color);
    font-size: 12px;
    color: var(--text-secondary);
}

.citation-item {
    display: inline-block;
    padding: 2px 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    margin: 2px 4px;
    cursor: pointer;
}

.citation-item:hover {
    background: var(--primary-color);
    color: white;
}

.message-save-btn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    margin-top: 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s;
}

.message-save-btn:hover {
    background: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
}

.message-save-btn.saved {
    background: var(--success-color);
    color: white;
    border-color: var(--success-color);
    cursor: default;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 加载动画 */
.loading-dots {
  display: inline-block;
  color: var(--text-secondary);
}

.loading-dots::after {
  content: '';
  animation: dots 1.5s steps(4, end) infinite;
}

@keyframes dots {
  0%, 20% { content: ''; }
  40% { content: '.'; }
  60% { content: '..'; }
  80%, 100% { content: '...'; }
}

/* ==================== 笔记弹窗样式 ==================== */
.modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 1000;
    display: none;
}

.modal.show {
    display: flex;
    align-items: center;
    justify-content: center;
}

.modal-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
}

.note-modal .modal-content {
    max-width: 560px;
}

.note-modal-content {
    position: relative;
    width: 90%;
    max-width: 560px;
    max-height: 90vh;
    background: var(--bg-primary);
    border-radius: 16px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

.modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-color);
}

.modal-header h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
}

.modal-close {
    width: 32px;
    height: 32px;
    border: none;
    background: var(--bg-tertiary);
    border-radius: 8px;
    font-size: 20px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
}

.modal-close:hover {
    background: var(--bg-secondary);
    color: var(--text-primary);
}

.modal-body {
    padding: 24px;
    overflow-y: auto;
    flex: 1;
}

.modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    padding: 16px 24px;
    border-top: 1px solid var(--border-color);
    background: var(--bg-secondary);
}

/* 问答预览 */
.qa-preview {
    background: var(--bg-tertiary);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}

.qa-preview-section {
    margin-bottom: 16px;
}

.qa-preview-section:last-child {
    margin-bottom: 0;
}

.qa-preview-section label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    display: block;
}

.qa-preview-content {
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-primary);
    background: var(--bg-secondary);
    padding: 12px;
    border-radius: 8px;
    max-height: 150px;
    overflow-y: auto;
}

.qa-preview-citations {
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
}

/* 笔记表单 */
.note-form {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.note-form .form-group {
    margin-bottom: 0;
}

.note-form label {
    display: block;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 6px;
}

.note-form .optional {
    font-weight: 400;
    color: var(--text-secondary);
    font-size: 12px;
}

.note-form .form-input,
.note-form .form-textarea {
    width: 100%;
    padding: 10px 12px;
    font-size: 14px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: var(--bg-secondary);
    color: var(--text-primary);
    transition: border-color 0.2s, box-shadow 0.2s;
    box-sizing: border-box;
}

.note-form .form-input:focus,
.note-form .form-textarea:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

.note-form .form-textarea {
    resize: vertical;
    min-height: 80px;
    font-family: inherit;
    line-height: 1.5;
}

/* 按钮样式 */
.btn {
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
}

.btn-primary {
    background: var(--primary-color);
    color: white;
}

.btn-primary:hover {
    background: var(--primary-dark);
}

.btn-secondary {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
}

.btn-secondary:hover {
    background: var(--bg-secondary);
}
</style>
