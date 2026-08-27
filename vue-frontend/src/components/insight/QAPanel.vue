<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { marked } from 'marked'
import * as insightApi from '@/api/insight'
import {
  NONTERMINAL_JOB_STATUSES,
  type V2Job,
} from '@/api/v2/jobs'
import { useInsightStore } from '@/stores/insightStore'
import type { QAMessage, QAMode } from '@/types/insight'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import { sanitizeHtml } from '@/utils/sanitizeHtml'
import { showToast } from '@/utils/toast'
import { confirmProductAction } from '@/composables/useProductConfirm'
import { stepKindLabel } from '@/utils/taskDisplay'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import QAComposer from './qa/QAComposer.vue'
import QAMessageList from './qa/QAMessageList.vue'
import QAOptionsBar from './qa/QAOptionsBar.vue'
import QASaveNoteModal from './qa/QASaveNoteModal.vue'
import { useQANoteModal } from './useQANoteModal'

type QARepairAction = NonNullable<insightApi.QAStatusResponse['repairAction']>

const insightStore = useInsightStore()
const taskCenterStore = useTaskCenterStore()
const questionInput = ref('')
const qaMode = ref<QAMode>('precise')
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
const isSubmittingQaRepair = ref(false)
const qaRepairTaskId = ref<string | null>(null)
const qaRepairBookId = ref<string | null>(null)
const qaRepairAction = ref<QARepairAction | null>(null)
const qaRepairProgressLabel = ref('')
const messageScrollRequestId = ref(0)
let chatRequestSequence = 0
let qaStatusRequestSequence = 0
let qaRepairRequestSequence = 0
let isQAPanelMounted = true
let handledTerminalRebuildTaskId = ''
let handledTerminalQaRepairTaskId = ''
let chatAbortController: AbortController | null = null
let streamRenderFrame: number | null = null
let pendingStreamRender: {
  requestId: number
  bookId: string
  messageId: string
  content: string
} | null = null

const qaHistory = computed(() => insightStore.qaHistory)
const isStreaming = computed(() => insightStore.isStreaming)
const qaComposerDisabled = computed(
  () => isStreaming.value || isQaStatusLoading.value || qaStatus.value?.available !== true
)
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
const activeQaDependencyTaskId = computed(() =>
  qaStatus.value?.repairAction === 'vector_rebuild'
    ? rebuildTaskId.value
    : qaRepairAction.value === qaStatus.value?.repairAction
      ? qaRepairTaskId.value
      : null
)
const qaStatusActionLabel = computed(() => {
  if (qaStatus.value?.repairAction === 'vector_rebuild' && rebuildTaskId.value) {
    return rebuildProgressLabel.value || '查看向量重建任务'
  }
  if (qaRepairTaskId.value && qaRepairAction.value === qaStatus.value?.repairAction) {
    return qaRepairProgressLabel.value || '查看修复任务'
  }
  return qaRepairLabel.value || (qaStatus.value?.reason === 'status_unavailable' ? '重新检查' : '')
})
const globalModeExamples = ['故事的主题是什么？', '主角的性格有什么变化？', '结局是怎样的？']

const {
  closeNoteModal,
  isSavingNote,
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
  const abortController = new AbortController()
  chatAbortController?.abort()
  chatAbortController = abortController

  questionInput.value = ''
  insightStore.clearQAHistory()
  insightStore.addQAMessage({
    id: Date.now().toString(),
    role: 'user',
    content: question,
  })

  await nextTick()
  scrollToBottom()

  const loadingText = qaMode.value === 'global' ? '正在分析全文...' : '思考中...'
  const loadingMessageId = (Date.now() + 1).toString()
  insightStore.addQAMessage({
    id: loadingMessageId,
    role: 'assistant',
    content: loadingText,
    isLoading: true,
  })

  insightStore.setStreaming(true)
  await nextTick()
  scrollToBottom()

  try {
    const streamOptions = {
      signal: abortController.signal,
      onChunk: (content: string) =>
        queueStreamRender({
          requestId,
          bookId,
          messageId: loadingMessageId,
          content,
        }),
    }
    const chatOptions: insightApi.SendChatOptions =
      qaMode.value === 'global'
        ? { ...streamOptions, mode: 'global' }
        : {
            ...streamOptions,
            mode: 'precise',
            threshold: threshold.value,
            topK: topK.value,
            useParentChild: useParentChild.value,
            useReasoning: useReasoning.value,
            useReranker: useReranker.value,
          }
    const response = await insightApi.sendChat(bookId, question, chatOptions)

    if (!isCurrentChatRequest(requestId, bookId)) return

    clearPendingStreamRender()
    insightStore.removeLoadingMessages()

    insightStore.addQAMessage({
      id: (Date.now() + 2).toString(),
      role: 'assistant',
      content: response.answer,
      mode: response.mode,
      citations: response.citations,
    })
  } catch (error) {
    if (!isCurrentChatRequest(requestId, bookId)) return
    clearPendingStreamRender()
    insightStore.removeLoadingMessages()
    insightStore.addQAMessage({
      id: (Date.now() + 2).toString(),
      role: 'assistant',
      content: `抱歉，处理问题时出错: ${error instanceof Error ? error.message : '未知错误'}`,
    })
  } finally {
    if (chatAbortController === abortController) chatAbortController = null
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

function queueStreamRender(update: NonNullable<typeof pendingStreamRender>): void {
  pendingStreamRender = update
  if (streamRenderFrame !== null) return
  // Coalesce token-sized SSE chunks so Markdown rendering and scrolling update
  // at most once per frame instead of once per provider token.
  streamRenderFrame = requestAnimationFrame(() => {
    streamRenderFrame = null
    const pending = pendingStreamRender
    pendingStreamRender = null
    if (!pending || !isCurrentChatRequest(pending.requestId, pending.bookId)) return
    insightStore.updateQAMessage(pending.messageId, { content: pending.content })
    scrollToBottom()
  })
}

function clearPendingStreamRender(): void {
  if (streamRenderFrame !== null) cancelAnimationFrame(streamRenderFrame)
  streamRenderFrame = null
  pendingStreamRender = null
}

function isCurrentChatRequest(requestId: number, bookId: string): boolean {
  return (
    isQAPanelMounted && requestId === chatRequestSequence && insightStore.currentBookId === bookId
  )
}

function scrollToBottom(): void {
  messageScrollRequestId.value += 1
}

async function rebuildEmbeddings(): Promise<void> {
  const bookId = insightStore.currentBookId
  if (!bookId || isRebuildingEmbeddings.value) return

  isRebuildingEmbeddings.value = true
  rebuildBookId.value = bookId
  rebuildProgressLabel.value = '等待确认...'
  const confirmed = await confirmProductAction({
    title: '重建向量索引',
    message: '确定要生成新的向量索引吗？当前索引会保留到新版本构建并发布完成。',
    confirmText: '重建',
    cancelText: '取消',
    tone: 'primary',
  })
  if (!confirmed) {
    if (isQAPanelMounted && bookId === insightStore.currentBookId) resetRebuildState()
    return
  }
  if (!isQAPanelMounted || bookId !== insightStore.currentBookId || rebuildTaskId.value) return

  rebuildProgressLabel.value = '准备启动...'

  try {
    const taskId = await insightApi.rebuildEmbeddings(bookId)
    if (!isQAPanelMounted || insightStore.currentBookId !== bookId) return
    rebuildTaskId.value = taskId
    rebuildBookId.value = bookId
    rebuildProgressLabel.value = '任务已启动'
  } catch (error) {
    if (!isQAPanelMounted || insightStore.currentBookId !== bookId) return
    const message = error instanceof Error ? error.message : '重建向量索引失败'
    showToast(message, 'error')
    resetRebuildState()
  }
}

function resetRebuildState(): void {
  isRebuildingEmbeddings.value = false
  rebuildTaskId.value = null
  rebuildBookId.value = null
  rebuildProgressLabel.value = ''
}

function resetQaRepairState(): void {
  qaRepairTaskId.value = null
  qaRepairBookId.value = null
  qaRepairAction.value = null
  qaRepairProgressLabel.value = ''
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
  qaStatus.value = null
  isQaStatusLoading.value = true
  try {
    const status = await insightApi.getQAStatus(bookId, mode)
    if (
      !isQAPanelMounted ||
      requestId !== qaStatusRequestSequence ||
      insightStore.currentBookId !== bookId ||
      qaMode.value !== mode
    )
      return
    qaStatus.value = status
  } catch {
    if (
      !isQAPanelMounted ||
      requestId !== qaStatusRequestSequence ||
      insightStore.currentBookId !== bookId ||
      qaMode.value !== mode
    )
      return
    qaStatus.value = {
      available: false,
      reason: 'status_unavailable',
    }
  } finally {
    if (
      isQAPanelMounted &&
      requestId === qaStatusRequestSequence &&
      insightStore.currentBookId === bookId &&
      qaMode.value === mode
    ) {
      isQaStatusLoading.value = false
    }
  }
}

async function repairQAStatus(): Promise<void> {
  const bookId = insightStore.currentBookId
  const mode = qaMode.value
  const repairAction = qaStatus.value?.repairAction
  if (!bookId || !repairAction || isSubmittingQaRepair.value) return
  if (repairAction === 'vector_rebuild') {
    await rebuildEmbeddings()
    return
  }
  if (repairAction === 'analyze') return

  const requestId = ++qaRepairRequestSequence
  isSubmittingQaRepair.value = true
  try {
    let taskId: string
    if (repairAction === 'overview_rebuild') {
      const result = await insightApi.regenerateOverview(bookId, 'story_summary', true)
      if (result.kind !== 'queued') throw new Error('故事概要重建未创建任务')
      taskId = result.jobId
    } else {
      taskId = await insightApi.rebuildCompressedContext(bookId)
    }
    if (
      !isQAPanelMounted ||
      requestId !== qaRepairRequestSequence ||
      insightStore.currentBookId !== bookId ||
      qaMode.value !== mode
    )
      return
    qaRepairTaskId.value = taskId
    qaRepairBookId.value = bookId
    qaRepairAction.value = repairAction
    qaRepairProgressLabel.value = '任务已启动'
    showToast('修复任务已进入任务中心', 'success')
  } catch (error) {
    if (
      !isQAPanelMounted ||
      requestId !== qaRepairRequestSequence ||
      insightStore.currentBookId !== bookId ||
      qaMode.value !== mode
    )
      return
    showToast(error instanceof Error ? error.message : '修复任务创建失败', 'error')
  } finally {
    if (requestId === qaRepairRequestSequence) {
      isSubmittingQaRepair.value = false
    }
  }
}

function handleQAStatusAction(): void {
  if (activeQaDependencyTaskId.value) {
    taskCenterStore.open({ jobId: activeQaDependencyTaskId.value })
    return
  }
  if (qaStatus.value?.reason === 'status_unavailable') {
    void refreshQAStatus()
  } else {
    void repairQAStatus()
  }
}

function dependencyTaskProgressLabel(job: V2Job, runningLabel: string): string {
  if (job.status === 'queued') return '等待执行'
  if (job.status === 'paused') return '已暂停'
  if (job.status === 'interrupted') return '已中断，请在任务中心继续或取消'
  const phase = job.progress.currentStep?.kind
    ? stepKindLabel(job.progress.currentStep.kind)
    : runningLabel
  const current = job.progress.completedItems
  const total = job.progress.totalItems
  return total > 0 ? `${phase} (${current}/${total})` : phase
}

function isActiveDependencyJob(job: V2Job): boolean {
  return NONTERMINAL_JOB_STATUSES.has(job.status)
}

function matchesDerivedRepairJob(
  job: V2Job,
  bookId: string,
  repairAction: QARepairAction
): boolean {
  if (job.bookId !== bookId || job.kind !== 'derived_rebuild' || !isActiveDependencyJob(job))
    return false
  if (repairAction === 'overview_rebuild') {
    return job.target.kind === 'overview' && job.target.template === 'story_summary'
  }
  if (repairAction === 'compressed_context_rebuild') {
    return job.target.kind === 'compressed_context' && job.target.template === 'default'
  }
  return false
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
    const active = [...taskCenterStore.queue, ...taskCenterStore.history].find(
      job => job.bookId === bookId && job.kind === 'vector_rebuild' && isActiveDependencyJob(job)
    )
    if (active) {
      rebuildTaskId.value = active.jobId
      rebuildBookId.value = bookId
      isRebuildingEmbeddings.value = true
    }
  }
  const taskId = rebuildTaskId.value
  if (!taskId) return
  const job = [...taskCenterStore.queue, ...taskCenterStore.history].find(
    item => item.jobId === taskId
  )
  if (!job) return
  if (isActiveDependencyJob(job)) {
    rebuildProgressLabel.value = dependencyTaskProgressLabel(job, '重建中')
    return
  }
  if (handledTerminalRebuildTaskId === taskId) return
  handledTerminalRebuildTaskId = taskId
  const succeeded = job.status === 'completed'
  resetRebuildState()
  showToast(
    succeeded ? '向量索引重建完成' : '向量索引重建未完成，请在任务中心查看详情',
    succeeded ? 'success' : 'error',
    succeeded ? 6000 : undefined
  )
  if (succeeded) void refreshQAStatus()
}

function projectQaRepairJob(): void {
  const bookId = insightStore.currentBookId
  if (!bookId) {
    resetQaRepairState()
    return
  }
  if (qaRepairBookId.value && qaRepairBookId.value !== bookId) {
    resetQaRepairState()
  }
  const currentRepairAction = qaStatus.value?.repairAction
  if (
    !qaRepairTaskId.value &&
    currentRepairAction &&
    currentRepairAction !== 'analyze' &&
    currentRepairAction !== 'vector_rebuild'
  ) {
    const active = [...taskCenterStore.queue, ...taskCenterStore.history].find(job =>
      matchesDerivedRepairJob(job, bookId, currentRepairAction)
    )
    if (active) {
      qaRepairTaskId.value = active.jobId
      qaRepairBookId.value = bookId
      qaRepairAction.value = currentRepairAction
    }
  }
  const taskId = qaRepairTaskId.value
  if (!taskId) return
  const job = [...taskCenterStore.queue, ...taskCenterStore.history].find(
    item => item.jobId === taskId
  )
  if (!job) return
  if (isActiveDependencyJob(job)) {
    qaRepairProgressLabel.value = dependencyTaskProgressLabel(job, '修复中')
    return
  }
  if (handledTerminalQaRepairTaskId === taskId) return
  handledTerminalQaRepairTaskId = taskId
  const succeeded = job.status === 'completed'
  resetQaRepairState()
  showToast(
    succeeded ? '问答依赖修复完成' : '问答依赖修复未完成，请在任务中心查看详情',
    succeeded ? 'success' : 'error',
    succeeded ? 6000 : undefined
  )
  void refreshQAStatus()
}

function projectDependencyJobs(): void {
  projectRebuildJob()
  projectQaRepairJob()
}

function renderMarkdown(content: string): string {
  if (!content) return ''
  return sanitizeHtml(marked.parse(content) as string)
}

function selectPage(pageNum: number): void {
  insightStore.selectPage(pageNum)
}

function askExampleQuestion(question: string): void {
  questionInput.value = question
  void sendQuestion()
}

function handleSaveNote(message: QAMessage): void {
  openNoteModal(message)
}

function resetChatSession(): void {
  chatRequestSequence += 1
  chatAbortController?.abort()
  chatAbortController = null
  clearPendingStreamRender()
  questionInput.value = ''
  insightStore.clearQAHistory()
  insightStore.setStreaming(false)
  closeNoteModal()
}

onMounted(() => {
  scrollToBottom()
})

onUnmounted(() => {
  isQAPanelMounted = false
  chatRequestSequence += 1
  qaStatusRequestSequence += 1
  qaRepairRequestSequence += 1
  chatAbortController?.abort()
  chatAbortController = null
  clearPendingStreamRender()
  insightStore.removeLoadingMessages()
  insightStore.setStreaming(false)
})

watch(
  () => insightStore.currentBookId,
  () => {
    qaRepairRequestSequence += 1
    qaStatus.value = null
    isSubmittingQaRepair.value = false
    resetRebuildState()
    resetQaRepairState()
    resetChatSession()
  },
  { immediate: true }
)

watch(qaMode, () => {
  qaRepairRequestSequence += 1
  isSubmittingQaRepair.value = false
  resetChatSession()
})

watch(
  [
    () => taskCenterStore.queue,
    () => taskCenterStore.history,
    () => insightStore.currentBookId,
    () => qaStatus.value?.repairAction,
  ],
  projectDependencyJobs,
  { immediate: true }
)

watch(
  [() => insightStore.currentBookId, () => qaMode.value, () => insightStore.dataRefreshKey],
  refreshQAStatus,
  { immediate: true }
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
        <template v-if="qaStatusActionLabel" #actions>
          <UiButton
            variant="secondary"
            size="sm"
            :disabled="
              isQaStatusLoading ||
                isSubmittingQaRepair ||
                (isRebuildingEmbeddings && !rebuildTaskId)
            "
            @click="handleQAStatusAction"
          >
            {{ qaStatusActionLabel }}
          </UiButton>
        </template>
      </ProductStatusBanner>

      <QAComposer
        v-model:question="questionInput"
        :disabled="qaComposerDisabled"
        @submit="sendQuestion"
      />
    </div>

    <QASaveNoteModal
      :visible="showNoteModal"
      :pending-q-a-data="pendingQAData"
      :render-markdown="renderMarkdown"
      :is-saving="isSavingNote"
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
  position: relative;
  padding: 16px;
  border-top: 1px solid var(--color-border-muted);
  background: var(--insight-surface-secondary);
}

.qa-panel__status {
  --product-status-banner-align-items: center;
  --product-status-banner-gap: 8px;
  --product-status-banner-padding: 7px 10px;
  --product-status-banner-content-display: flex;
  --product-status-banner-content-align-items: center;
  --product-status-banner-content-gap: 6px;
  --product-status-banner-title-margin-bottom: 0;
  --product-status-banner-title-font-size: 12px;
  --product-status-banner-body-font-size: 12px;

  position: absolute;
  right: 16px;
  bottom: calc(100% + 8px);
  left: 16px;
  margin: 0;
}
</style>
