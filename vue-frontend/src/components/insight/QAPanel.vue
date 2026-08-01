<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { marked } from 'marked'
import * as insightApi from '@/api/insight'
import { useInsightStore, type QAMessage } from '@/stores/insightStore'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import { sanitizeHtml } from '@/utils/sanitizeHtml'
import { showToast } from '@/utils/toast'
import { confirmProductAction } from '@/composables/useProductConfirm'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import QAComposer from './qa/QAComposer.vue'
import QAMessageList from './qa/QAMessageList.vue'
import QAOptionsBar from './qa/QAOptionsBar.vue'
import QASaveNoteModal from './qa/QASaveNoteModal.vue'
import { useQANoteModal } from './useQANoteModal'

const insightStore = useInsightStore()
const taskCenterStore = useTaskCenterStore()
const questionInput = ref('')
const qaMode = ref<'precise' | 'global'>('precise')
const useParentChild = ref(true)
const useReasoning = ref(true)
const useReranker = ref(true)
const topK = ref(5)
const threshold = ref(0)
const isRebuildingEmbeddings = ref(false)
const rebuildTaskId = ref<string | null>(null)
const rebuildBookId = ref<string | null>(null)
const rebuildProgressLabel = ref('')
const qaStatus = ref<insightApi.QAStatusResponse | null>(null)
const isQaStatusLoading = ref(false)
const isRepairingQaDependency = ref(false)
const messageScrollRequestId = ref(0)
let chatRequestSequence = 0
let qaStatusRequestSequence = 0
let isQAPanelMounted = true
let handledTerminalRebuildTaskId = ''

const qaHistory = computed(() => insightStore.qaHistory)
const isStreaming = computed(() => insightStore.isStreaming)
const qaComposerDisabled = computed(() => (
  isStreaming.value
  || isQaStatusLoading.value
  || isRepairingQaDependency.value
  || qaStatus.value?.available !== true
))
const qaStatusTitle = computed(() => {
  if (isQaStatusLoading.value) return '正在检查问答依赖'
  return qaMode.value === 'global' ? '全局问答暂不可用' : '精确问答暂不可用'
})
const qaStatusMessage = computed(() => {
  if (isQaStatusLoading.value) return '正在从后端读取当前书籍的问答可用性。'
  const reason = qaStatus.value?.reason
  const messages: Record<string, string> = {
    analysis_missing: '当前书籍还没有可用的分析结果，请先在左侧启动漫画分析。',
    vector_missing: '精确问答需要先构建向量索引，重建完成后即可提问。',
    vector_stale: '漫画分析结果已经变化，现有向量索引已过期，请重建后再提问。',
    global_summary_missing: '全局问答需要故事概要，请先生成故事概要。',
    global_summary_stale: '故事概要已过期，请重新生成后再使用全局问答。',
    compressed_context_missing: '全局问答需要压缩上下文，请先重建压缩上下文。',
    compressed_context_stale: '压缩上下文已过期，请重建后再使用全局问答。',
    status_unavailable: '暂时无法确认问答依赖状态，请稍后重试。',
  }
  return messages[String(reason ?? '')] ?? '当前问答依赖尚未准备完成。'
})
const qaRepairLabel = computed(() => {
  switch (qaStatus.value?.repairAction) {
    case 'vector_rebuild':
      return '重建向量'
    case 'overview_rebuild':
      return '生成故事概要'
    case 'compressed_context_rebuild':
      return '重建压缩上下文'
    default:
      return ''
  }
})
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
  const bookId = insightStore.currentBookId
  if (!question || !bookId || qaComposerDisabled.value) return

  const requestId = ++chatRequestSequence

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
    const response = await insightApi.sendChat(bookId, question, {
      use_parent_child: useParentChild.value,
      use_reasoning: useReasoning.value,
      use_reranker: useReranker.value,
      top_k: topK.value,
      threshold: threshold.value,
      use_global_context: qaMode.value === 'global',
    })

    if (!isCurrentChatRequest(requestId, bookId)) return

    insightStore.removeLoadingMessages()

    if (response.success) {
      const modeLabel = response.mode === 'global' ? '全局模式' : '精确模式'
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
    if (!isCurrentChatRequest(requestId, bookId)) return
    insightStore.removeLoadingMessages()
    insightStore.addQAMessage({
      id: (Date.now() + 2).toString(),
      role: 'assistant',
      content: '抱歉，网络请求失败，请稍后重试。',
      timestamp: new Date().toISOString(),
    })
  } finally {
    if (isCurrentChatRequest(requestId, bookId)) {
      insightStore.setStreaming(false)
      await nextTick()
      scrollToBottom()
    } else if (requestId === chatRequestSequence) {
      insightStore.removeLoadingMessages()
      insightStore.setStreaming(false)
    }
  }
}

function isCurrentChatRequest(requestId: number, bookId: string): boolean {
  return (
    isQAPanelMounted &&
    requestId === chatRequestSequence &&
    insightStore.currentBookId === bookId
  )
}

function scrollToBottom(): void {
  messageScrollRequestId.value += 1
}

async function rebuildEmbeddings(): Promise<void> {
  const bookId = insightStore.currentBookId
  if (!bookId || isRebuildingEmbeddings.value) return

  const confirmed = await confirmProductAction({
    title: '重建向量索引',
    message: '确定要重建向量索引吗？这将删除现有的向量数据并重新构建，可能需要一些时间。',
    confirmText: '重建',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed || bookId !== insightStore.currentBookId) return

  insightStore.setLoading(true)
  isRebuildingEmbeddings.value = true
  rebuildProgressLabel.value = '准备启动...'

  try {
    const response = await insightApi.rebuildEmbeddings(bookId)

    if (!response.success || !response.task_id) {
      showToast('重建失败: ' + (response.error || '未知错误'), 'error')
      resetRebuildState()
      return
    }

    rebuildTaskId.value = response.task_id
    rebuildBookId.value = bookId
    rebuildProgressLabel.value = '任务已启动'
    await taskCenterStore.refresh()
  } catch (error) {
    const message = error instanceof Error ? error.message : '重建向量索引失败'
    showToast(message, 'error')
    resetRebuildState()
  }
}

function resetRebuildState(): void {
  isRebuildingEmbeddings.value = false
  insightStore.setLoading(false)
  rebuildTaskId.value = null
  rebuildBookId.value = null
  rebuildProgressLabel.value = ''
}

async function refreshQAStatus(): Promise<void> {
  const bookId = insightStore.currentBookId
  const mode = qaMode.value
  const requestId = ++qaStatusRequestSequence
  if (!bookId) {
    qaStatus.value = null
    isQaStatusLoading.value = false
    return
  }
  isQaStatusLoading.value = true
  try {
    const status = await insightApi.getQAStatus(bookId, mode)
    if (
      !isQAPanelMounted
      || requestId !== qaStatusRequestSequence
      || insightStore.currentBookId !== bookId
      || qaMode.value !== mode
    ) return
    qaStatus.value = status
  } catch {
    if (
      requestId !== qaStatusRequestSequence
      || insightStore.currentBookId !== bookId
      || qaMode.value !== mode
    ) return
    qaStatus.value = {
      available: false,
      reason: 'status_unavailable',
    }
  } finally {
    if (
      requestId === qaStatusRequestSequence
      && insightStore.currentBookId === bookId
      && qaMode.value === mode
    ) {
      isQaStatusLoading.value = false
    }
  }
}

async function repairQAStatus(): Promise<void> {
  const bookId = insightStore.currentBookId
  const repairAction = qaStatus.value?.repairAction
  if (!bookId || !repairAction || isRepairingQaDependency.value) return
  if (repairAction === 'vector_rebuild') {
    await rebuildEmbeddings()
    return
  }
  if (repairAction === 'analyze') return

  isRepairingQaDependency.value = true
  try {
    const response = repairAction === 'overview_rebuild'
      ? await insightApi.regenerateOverview(bookId, 'story_summary', true)
      : await insightApi.rebuildCompressedContext(bookId)
    if (!response.success) {
      showToast(response.error || '修复任务创建失败', 'error')
      return
    }
    await taskCenterStore.refresh()
    showToast(response.message || '修复任务已进入任务中心', 'success')
  } catch (error) {
    showToast(error instanceof Error ? error.message : '修复任务创建失败', 'error')
  } finally {
    isRepairingQaDependency.value = false
  }
}

function projectRebuildJob(): void {
  const bookId = insightStore.currentBookId
  if (!bookId) {
    resetRebuildState()
    return
  }
  if (rebuildBookId.value && rebuildBookId.value !== bookId) {
    resetRebuildState()
  }
  if (!rebuildTaskId.value) {
    const active = taskCenterStore.queue.find(job => (
      job.bookId === bookId
      && job.kind === 'vector_rebuild'
      && job.status !== 'interrupted'
    ))
    if (active) {
      rebuildTaskId.value = active.jobId
      rebuildBookId.value = bookId
      isRebuildingEmbeddings.value = true
      insightStore.setLoading(true)
    }
  }
  const taskId = rebuildTaskId.value
  if (!taskId) return
  const job = [...taskCenterStore.queue, ...taskCenterStore.history]
    .find(item => item.jobId === taskId)
  if (!job) return
  if (['queued', 'running', 'pausing', 'paused', 'cancelling'].includes(job.status)) {
    const progress = job.progress as Record<string, unknown>
    const currentStep = progress.currentStep && typeof progress.currentStep === 'object'
      ? progress.currentStep as Record<string, unknown>
      : undefined
    const phase = String(currentStep?.kind ?? '重建中')
    const current = Number(progress.completedItems ?? 0)
    const total = Number(progress.totalItems ?? 0)
    rebuildProgressLabel.value = total > 0 ? `${phase} (${current}/${total})` : phase
    return
  }
  if (handledTerminalRebuildTaskId === taskId) return
  handledTerminalRebuildTaskId = taskId
  const succeeded = job.status === 'completed' || job.status === 'completed_with_errors'
  resetRebuildState()
  showToast(
    succeeded ? '向量索引重建完成' : '向量索引重建未完成，请在任务中心查看详情',
    succeeded ? 'success' : 'error',
    succeeded ? 6000 : undefined,
  )
  if (succeeded) void refreshQAStatus()
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
  projectRebuildJob()
})

onUnmounted(() => {
  isQAPanelMounted = false
  chatRequestSequence += 1
  qaStatusRequestSequence += 1
  insightStore.removeLoadingMessages()
  insightStore.setStreaming(false)
})

watch(
  [
    () => taskCenterStore.queue,
    () => taskCenterStore.history,
    () => insightStore.currentBookId,
  ],
  projectRebuildJob,
  { immediate: true },
)

watch(
  [
    () => insightStore.currentBookId,
    () => qaMode.value,
    () => insightStore.dataRefreshKey,
  ],
  refreshQAStatus,
  { immediate: true },
)
</script>

<template>
  <div class="qa-panel">
    <QAMessageList
      :messages="qaHistory"
      :render-markdown="renderMarkdown"
      :scroll-request-id="messageScrollRequestId"
      @save-note="handleSaveNote"
      @select-page="selectPage"
    />

    <div class="qa-panel__input-shell">
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

      <ProductStatusBanner
        v-if="isQaStatusLoading || qaStatus?.available !== true"
        class="qa-panel__status"
        :title="qaStatusTitle"
        :tone="isQaStatusLoading ? 'neutral' : 'warning'"
        role="status"
        aria-live="polite"
      >
        {{ qaStatusMessage }}
        <template v-if="qaRepairLabel" #actions>
          <UiButton
            variant="secondary"
            size="sm"
            :disabled="isRepairingQaDependency || isRebuildingEmbeddings"
            @click="repairQAStatus"
          >
            {{ qaRepairLabel }}
          </UiButton>
        </template>
      </ProductStatusBanner>

      <QAComposer
        v-model:question="questionInput"
        :is-streaming="isStreaming"
        :disabled="qaComposerDisabled"
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
.qa-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.qa-panel__input-shell {
  padding: 16px;
  border-top: 1px solid var(--color-border-muted);
  background: var(--insight-surface-secondary);
}

.qa-panel__status {
  margin-bottom: 12px;
}
</style>
