<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { marked } from 'marked'
import * as insightApi from '@/api/insight'
import { useInsightStore, type QAMessage } from '@/stores/insightStore'
import { sanitizeHtml } from '@/utils/sanitizeHtml'
import { showToast } from '@/utils/toast'
import QAComposer from './qa/QAComposer.vue'
import QAMessageList from './qa/QAMessageList.vue'
import QAOptionsBar from './qa/QAOptionsBar.vue'
import QASaveNoteModal from './qa/QASaveNoteModal.vue'
import { useQANoteModal } from './useQANoteModal'

const insightStore = useInsightStore()
const questionInput = ref('')
const qaMode = ref<'precise' | 'global'>('precise')
const useParentChild = ref(true)
const useReasoning = ref(true)
const useReranker = ref(true)
const topK = ref(5)
const threshold = ref(0)
const isRebuildingEmbeddings = ref(false)
const rebuildTaskId = ref<string | null>(null)
const rebuildProgressLabel = ref('')
const rebuildPollingFailures = ref(0)
const messageList = ref<InstanceType<typeof QAMessageList> | null>(null)
let rebuildPollingTimer: ReturnType<typeof setInterval> | null = null

const qaHistory = computed(() => insightStore.qaHistory)
const isStreaming = computed(() => insightStore.isStreaming)
const globalModeExamples = [
  '故事的主题是什么？',
  '主角的性格有什么变化？',
  '结局是怎样的？',
]

const {
  closeNoteModal,
  noteComment,
  noteTitle,
  openNoteModal,
  pendingQAData,
  saveNote,
  showNoteModal,
} = useQANoteModal(insightStore)

async function sendQuestion(): Promise<void> {
  const question = questionInput.value.trim()
  if (!question || !insightStore.currentBookId || isStreaming.value) return

  questionInput.value = ''
  insightStore.clearQAHistory()
  insightStore.addQAMessage({
    id: Date.now().toString(),
    role: 'user',
    content: question,
    timestamp: new Date().toISOString(),
  })

  await nextTick()
  scrollToBottom()

  const loadingText = qaMode.value === 'global' ? '正在分析全文...' : '思考中...'
  insightStore.addQAMessage({
    id: (Date.now() + 1).toString(),
    role: 'assistant',
    content: loadingText,
    timestamp: new Date().toISOString(),
    isLoading: true,
  })

  insightStore.setStreaming(true)

  try {
    const response = await insightApi.sendChat(insightStore.currentBookId, question, {
      use_parent_child: useParentChild.value,
      use_reasoning: useReasoning.value,
      use_reranker: useReranker.value,
      top_k: topK.value,
      threshold: threshold.value,
      use_global_context: qaMode.value === 'global',
    })

    insightStore.removeLoadingMessages()

    if (response.success) {
      const modeLabel = response.mode === 'global' ? '🌐 全局模式' : '🎯 精确模式'
      insightStore.addQAMessage({
        id: (Date.now() + 2).toString(),
        role: 'assistant',
        content: response.answer || '',
        timestamp: new Date().toISOString(),
        mode: modeLabel,
        citations: response.citations || [],
      })
    } else {
      insightStore.addQAMessage({
        id: (Date.now() + 2).toString(),
        role: 'assistant',
        content: `抱歉，处理问题时出错: ${response.error || '未知错误'}`,
        timestamp: new Date().toISOString(),
      })
    }
  } catch {
    insightStore.removeLoadingMessages()
    insightStore.addQAMessage({
      id: (Date.now() + 2).toString(),
      role: 'assistant',
      content: '抱歉，网络请求失败，请稍后重试。',
      timestamp: new Date().toISOString(),
    })
  } finally {
    insightStore.setStreaming(false)
    await nextTick()
    scrollToBottom()
  }
}

function scrollToBottom(): void {
  messageList.value?.scrollToBottom()
}

async function rebuildEmbeddings(): Promise<void> {
  if (!insightStore.currentBookId || isRebuildingEmbeddings.value) return
  if (!confirm('确定要重建向量索引吗？\n\n这将删除现有的向量数据并重新构建，可能需要一些时间。')) return

  insightStore.setLoading(true)
  isRebuildingEmbeddings.value = true
  rebuildProgressLabel.value = '准备启动...'

  try {
    const response = await insightApi.rebuildEmbeddings(insightStore.currentBookId)

    if (!response.success || !response.task_id) {
      showToast('重建失败: ' + (response.error || '未知错误'), 'error')
      resetRebuildState()
      return
    }

    rebuildTaskId.value = response.task_id
    rebuildProgressLabel.value = '任务已启动'
    rebuildPollingFailures.value = 0
    startRebuildStatusPolling(response.task_id)
  } catch (error) {
    const message = error instanceof Error ? error.message : '重建向量索引失败'
    showToast(message, 'error')
    resetRebuildState()
  }
}

function stopRebuildStatusPolling(): void {
  if (rebuildPollingTimer) {
    clearInterval(rebuildPollingTimer)
    rebuildPollingTimer = null
  }
}

function resetRebuildState(): void {
  stopRebuildStatusPolling()
  isRebuildingEmbeddings.value = false
  insightStore.setLoading(false)
  rebuildTaskId.value = null
  rebuildProgressLabel.value = ''
}

async function pollRebuildStatus(taskId: string): Promise<void> {
  if (!insightStore.currentBookId) return

  const response = await insightApi.getRebuildEmbeddingsStatus(insightStore.currentBookId, taskId)
  const task = response.task
  if (!task) {
    resetRebuildState()
    showToast('重建失败: 未找到向量重建任务状态', 'error')
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
    resetRebuildState()

    let message = '向量索引重建完成'
    if (response.stats) {
      message += `\n页面向量: ${response.stats.pages_count || 0} 条`
      if (response.stats.events_count !== undefined) {
        message += `\n事件向量: ${response.stats.events_count || 0} 条`
      }
    }
    showToast(message, 'success', 6000)
    return
  }

  if (task.status === 'failed' || task.status === 'cancelled') {
    resetRebuildState()
    showToast('重建失败: ' + (task.error_message || response.error || '未知错误'), 'error')
  }
}

function startRebuildStatusPolling(taskId: string): void {
  stopRebuildStatusPolling()
  rebuildPollingTimer = setInterval(() => {
    void pollRebuildStatus(taskId).catch(() => {
      rebuildPollingFailures.value += 1
      if (rebuildPollingFailures.value >= 3) {
        resetRebuildState()
        showToast('重建失败: 无法获取任务状态，请稍后查看结果', 'error')
      }
    })
  }, 3000)
}

function renderMarkdown(content: string): string {
  if (!content) return ''
  return sanitizeHtml(marked.parse(content) as string)
}

function selectPage(pageNum: number): void {
  insightStore.setCurrentPage(pageNum)
}

function askExampleQuestion(question: string): void {
  questionInput.value = question
  sendQuestion()
}

function handleSaveNote(message: QAMessage): void {
  openNoteModal(message)
}

onMounted(() => {
  scrollToBottom()
})

onUnmounted(() => {
  stopRebuildStatusPolling()
})
</script>

<template>
  <div class="qa-container">
    <QAMessageList
      ref="messageList"
      :messages="qaHistory"
      :render-markdown="renderMarkdown"
      @save-note="handleSaveNote"
      @select-page="selectPage"
    />

    <div class="chat-input-container">
      <QAOptionsBar
        v-model:qa-mode="qaMode"
        v-model:use-parent-child="useParentChild"
        v-model:use-reasoning="useReasoning"
        v-model:use-reranker="useReranker"
        v-model:top-k="topK"
        v-model:threshold="threshold"
        :global-mode-examples="globalModeExamples"
        :is-rebuilding-embeddings="isRebuildingEmbeddings"
        :progress-label="rebuildProgressLabel"
        @ask-example="askExampleQuestion"
        @rebuild="rebuildEmbeddings"
      />

      <QAComposer
        v-model:question="questionInput"
        :is-streaming="isStreaming"
        @submit="sendQuestion"
      />
    </div>

    <QASaveNoteModal
      :visible="showNoteModal"
      :pending-q-a-data="pendingQAData"
      :render-markdown="renderMarkdown"
      v-model:note-title="noteTitle"
      v-model:note-comment="noteComment"
      @close="closeNoteModal"
      @save="saveNote"
    />
  </div>
</template>

<style scoped>
.qa-container {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.chat-input-container {
  padding: 16px;
  border-top: 1px solid var(--color-border-muted);
  background: var(--insight-surface-secondary);
}
</style>
