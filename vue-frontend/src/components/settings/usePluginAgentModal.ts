import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { marked } from 'marked'
import { fetchModels as fetchV2Models, testAiTranslateConnection } from '@/api/v2/diagnostics'
import {
  cancelPluginAgentExecution,
  createPluginAgentSession,
  deletePluginAgentSession,
  getPluginAgentSession,
  getPluginAgentSettings,
  listPluginAgentJobEvents,
  lockPluginAgentTarget,
  pluginAgentEventFromJobEvent,
  sendPluginAgentMessage,
  startPluginAgentExecution,
  type PluginAgentAssistantDeltaPayload,
  type PluginAgentAssistantPayload,
  type PluginAgentDonePayload,
  type PluginAgentErrorPayload,
  type PluginAgentEvent,
  type PluginAgentOverviewSection,
  type PluginAgentSession,
  type PluginAgentStatePayload,
  type PluginAgentToolResultPayload,
  type PluginAgentValidationPayload,
} from '@/api/pluginAgent'
import { useSettingsStore } from '@/stores/settings'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import type { V2JobEvent } from '@/api/v2/jobs'
import type { PluginAgentProvider } from '@/types/settings'
import { providerRequiresApiKeyForBaseUrl, providerRequiresBaseUrl } from '@/config/aiProviders'
import { sanitizeHtml } from '@/utils/sanitizeHtml'
import { useToast } from '@/utils/toast'
import {
  useAiModelDiscovery,
  type AiModelDiscoveryMessageTone,
} from '@/composables/useAiModelDiscovery'
import { buildTimelineItems, type PluginAgentTimelineItem } from './pluginAgentTimeline'
import { usePluginAgentDisplayAnimation } from './usePluginAgentDisplayAnimation'

export interface PluginAgentModalProps {
  modelValue: boolean
}

export type PluginAgentModalEmit = {
  (e: 'update:modelValue', value: boolean): void
  (e: 'pluginsChanged'): void
  (e: 'settingsSaved'): void
}

export function usePluginAgentModal(props: PluginAgentModalProps, emit: PluginAgentModalEmit) {
  const settingsStore = useSettingsStore()
  const taskCenterStore = useTaskCenterStore()
  const toast = useToast()

  type HistoryScrollTarget = HTMLElement | { scrollToBottom: () => void }

  interface PluginAgentConversationMessage {
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: string
    isLoading?: boolean
    isOptimistic?: boolean
  }

  const stateLabelMap: Record<string, string> = {
    drafting: '等待需求描述',
    awaiting_target_lock: '等待锁定目标插件',
    ready: '已就绪',
    running: '执行中',
    pausing: '正在暂停',
    paused: '已暂停',
    cancelling: '正在取消',
    completed: '已完成',
    failed: '执行失败',
    cancelled: '已取消',
  }
  const shouldAnimateAssistantStream =
    typeof navigator !== 'undefined' ? !navigator.userAgent.toLowerCase().includes('jsdom') : true

  const isOpen = ref(props.modelValue)
  const mode = ref<'create' | 'modify'>('create')
  const selectedPluginId = ref('')
  const overview = ref<string[]>([])
  const overviewSections = ref<PluginAgentOverviewSection[]>([])
  const promptExamples = ref<string[]>([])
  const providerOptions = ref<Array<{ value: string; label: string }>>([])
  const pluginOptions = ref<Array<{ value: string; label: string }>>([])
  const session = ref<PluginAgentSession | null>(null)
  const messageInput = ref('')
  const eventFeed = ref<PluginAgentEvent[]>([])
  const messagesContainer = ref<HistoryScrollTarget | null>(null)
  const isDebugExpanded = ref(false)
  const isTestingConnection = ref(false)
  const isSavingAgentSettings = ref(false)
  const isAwaitingPlanningReply = ref(false)
  const isSessionCommandPending = ref(false)
  const optimisticMessages = ref<PluginAgentConversationMessage[]>([])
  let activeStreamJobId = ''
  let bufferedJobEvents: V2JobEvent[] = []
  let jobEventCursor = 0
  let sawPluginTerminalEvent = false
  let stopTaskEvents: (() => void) | null = null
  let streamGeneration = 0
  let streamIsCatchingUp = false
  let initializeRequestId = 0

  const agentSettings = computed(() => settingsStore.settings.pluginAgent)
  const hasStoredAgentCredential = computed(() =>
    settingsStore.hasCredential('plugin_agent', agentSettings.value.provider)
  )
  function notifyModelDiscovery(message: string, tone: AiModelDiscoveryMessageTone): void {
    toast[tone](message)
  }
  const modelDiscovery = useAiModelDiscovery({
    source: () => ({
      provider: agentSettings.value.provider,
      apiKey: agentSettings.value.apiKey,
      baseUrl: agentSettings.value.customBaseUrl,
      hasStoredCredential: hasStoredAgentCredential.value,
    }),
    fetcher: (provider, apiKey, baseUrl) =>
      fetchV2Models(provider, apiKey, baseUrl, 'plugin_agent'),
    notify: notifyModelDiscovery,
    requiresApiKey: providerRequiresApiKeyForBaseUrl,
    emptyBaseUrl: '',
    errorMessage: error => (error instanceof Error ? error.message : '获取模型失败'),
  })
  const { isFetchingModels } = modelDiscovery
  const fetchedModels = computed(() =>
    modelDiscovery.models.value.map(model => ({
      id: model.id,
      name: model.name || model.id,
    }))
  )
  const displayAnimation = usePluginAgentDisplayAnimation({
    animate: shouldAnimateAssistantStream,
    onTick: syncHistoryScrollToBottom,
  })
  const { assistantMessageDisplayContent, assistantDisplayContent, assistantDisplayTargets } =
    displayAnimation
  const clearAssistantDisplayAnimation = displayAnimation.clear
  const setAssistantDisplayTarget = displayAnimation.setStreamTarget
  function setAssistantMessageDisplayTarget(
    messageId: string,
    targetContent: string,
    options: { animate: boolean }
  ): void {
    displayAnimation.setMessageTarget(messageId, targetContent, options.animate)
  }

  const messages = computed<PluginAgentConversationMessage[]>(() => {
    const sessionMessages = (session.value?.messages || []).map(message => ({
      ...message,
      isLoading: false,
      isOptimistic: false,
    }))
    return [...sessionMessages, ...optimisticMessages.value]
  })
  const modelListOptions = computed(() => {
    const options = [{ label: '-- 选择模型 --', value: '' }]
    for (const model of fetchedModels.value) {
      options.push({
        label: model.name || model.id,
        value: model.id,
      })
    }
    return options
  })
  const timelineItems = computed<PluginAgentTimelineItem[]>(() =>
    buildTimelineItems(
      eventFeed.value,
      assistantDisplayContent.value,
      assistantDisplayTargets.value
    )
  )
  const isRunning = computed(() =>
    ['running', 'pausing', 'paused', 'cancelling'].includes(
      session.value?.run_state || ''
    )
  )
  const isConversationPending = computed(
    () => isRunning.value || isAwaitingPlanningReply.value || isSessionCommandPending.value,
  )
  const canBeginConversation = computed(() => {
    if (isConversationPending.value) return false
    if (mode.value === 'modify') {
      return Boolean(selectedPluginId.value && messageInput.value.trim())
    }
    return Boolean(messageInput.value.trim())
  })
  const canLockTarget = computed(
    () => !isSessionCommandPending.value
      && mode.value === 'create'
      && Boolean(session.value?.pending_target)
  )
  const canStartExecution = computed(
    () =>
      !isSessionCommandPending.value &&
      Boolean(session.value?.locked_target && session.value?.run_state === 'ready') &&
      messages.value.some(message => message.role === 'user')
  )
  const currentRunStateLabel = computed(() => {
    return stateLabelMap[session.value?.run_state || 'drafting'] || '等待需求描述'
  })
  const lockedTargetLabel = computed(() => {
    if (session.value?.locked_target) {
      return `${session.value.locked_target.display_name} (${session.value.locked_target.plugin_id})`
    }
    return '未锁定'
  })

  function applySession(nextSession: PluginAgentSession | null): void {
    const previousSession = session.value
    session.value = nextSession
    if (!nextSession) {
      eventFeed.value = []
      isDebugExpanded.value = false
      optimisticMessages.value = []
      isAwaitingPlanningReply.value = false
      clearAssistantDisplayAnimation()
      return
    }
    mode.value = nextSession.mode
    selectedPluginId.value =
      nextSession.selected_plugin_id ||
      nextSession.locked_target?.plugin_id ||
      selectedPluginId.value
    const sessionEvents = nextSession.events.map(event => ({
      ...event,
      eventKey: `session:${event.id}`,
    }))
    if (previousSession?.session_id === nextSession.session_id && eventFeed.value.length > 0) {
      const merged = [...eventFeed.value]
      for (const event of sessionEvents) {
        if (
          !merged.some(
            existing => (existing.eventKey ?? `session:${existing.id}`) === event.eventKey
          )
        ) {
          merged.push(event)
        }
      }
      eventFeed.value = merged
    } else {
      eventFeed.value = sessionEvents
    }

    const previousMessageIds = new Set((previousSession?.messages ?? []).map(message => message.id))
    const shouldAnimatePlanningMessages = previousSession?.session_id === nextSession.session_id
    for (const message of nextSession.messages) {
      if (message.role !== 'assistant') {
        continue
      }
      if (shouldAnimatePlanningMessages && !previousMessageIds.has(message.id)) {
        setAssistantMessageDisplayTarget(message.id, message.content, { animate: true })
      } else {
        setAssistantMessageDisplayTarget(message.id, message.content, { animate: false })
      }
    }
  }

  function scrollHistoryToBottom(): void {
    const element = messagesContainer.value
    if (!element) return
    if ('scrollToBottom' in element) {
      element.scrollToBottom()
      return
    }
    element.scrollTop = element.scrollHeight
  }

  async function syncHistoryScrollToBottom(): Promise<void> {
    await nextTick()
    scrollHistoryToBottom()
  }

  function buildOptimisticPlanningMessages(userContent: string): PluginAgentConversationMessage[] {
    const now = new Date().toISOString()
    return [
      {
        id: `optimistic-user-${Date.now()}`,
        role: 'user',
        content: userContent,
        timestamp: now,
        isOptimistic: true,
      },
      {
        id: `optimistic-assistant-${Date.now() + 1}`,
        role: 'assistant',
        content: 'Agent 正在分析需求',
        timestamp: now,
        isLoading: true,
        isOptimistic: true,
      },
    ]
  }

  function clearPlanningOptimisticState(): void {
    optimisticMessages.value = []
    isAwaitingPlanningReply.value = false
  }

  async function syncSessionFromServer(sessionId: string): Promise<void> {
    const result = await getPluginAgentSession(sessionId)
    applySession(result)
  }

  watch(
    () => props.modelValue,
    async value => {
      isOpen.value = value
      if (value) {
        await initializeModal()
      } else {
        stopStreaming()
      }
    },
    { immediate: true }
  )

  watch(isOpen, value => {
    if (!value) {
      initializeRequestId += 1
      emit('update:modelValue', false)
    }
  })

  onBeforeUnmount(() => {
    stopStreaming()
    clearAssistantDisplayAnimation()
    modelDiscovery.invalidate()
  })

  async function initializeModal(options: { preserveSessionWhenMissing?: boolean } = {}): Promise<void> {
    if (!isOpen.value) return
    const requestId = ++initializeRequestId
    try {
      modelDiscovery.invalidate()
      const result = await getPluginAgentSettings()
      if (requestId !== initializeRequestId || !isOpen.value) return
      overview.value = result.overview
      overviewSections.value = result.overview_sections
      promptExamples.value = result.prompt_examples
      providerOptions.value = result.providers
      pluginOptions.value = [
        { value: '', label: '-- 选择插件 --' },
        ...result.plugins.map(plugin => ({
          value: plugin.pluginId,
          label: plugin.displayName,
        })),
      ]
      if (result.session || !options.preserveSessionWhenMissing) {
        applySession(result.session)
      }
      if (session.value) {
        selectedPluginId.value =
          session.value.selected_plugin_id ||
          session.value.locked_target?.plugin_id ||
          selectedPluginId.value
        if (isRunning.value) {
          void startStreaming()
        }
      }
    } catch (error) {
      if (requestId !== initializeRequestId || !isOpen.value) return
      toast.error(error instanceof Error ? error.message : '加载插件 Agent 设置失败')
    }
  }

  async function handleModeChange(nextMode: 'create' | 'modify'): Promise<void> {
    if (isConversationPending.value) return
    if (nextMode === mode.value) return
    const sessionId = session.value?.session_id
    isSessionCommandPending.value = true
    try {
      if (sessionId) {
        await deletePluginAgentSession(sessionId)
      }
      mode.value = nextMode
      applySession(null)
      selectedPluginId.value = ''
      messageInput.value = ''
      stopStreaming()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '切换任务模式失败')
    } finally {
      isSessionCommandPending.value = false
    }
  }

  function handleProviderChange(value: string | number): void {
    modelDiscovery.invalidate()
    const provider = String(value || '') as PluginAgentProvider
    if (!providerOptions.value.some(option => option.value === provider)) return
    settingsStore.setPluginAgentProvider(provider)
  }

  async function handleSelectedPluginChange(value: string | number): Promise<void> {
    if (isConversationPending.value) return
    const nextPluginId = String(value || '')
    if (selectedPluginId.value === nextPluginId) return
    const sessionId = mode.value === 'modify' ? session.value?.session_id : undefined
    if (!sessionId) {
      selectedPluginId.value = nextPluginId
      return
    }
    isSessionCommandPending.value = true
    try {
      await deletePluginAgentSession(sessionId)
      applySession(null)
      selectedPluginId.value = nextPluginId
      messageInput.value = ''
      stopStreaming()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '切换目标插件失败')
    } finally {
      isSessionCommandPending.value = false
    }
  }

  function handleModelSelected(value: string | number): void {
    settingsStore.updatePluginAgent({ modelName: String(value || '') })
  }

  function updateAgentString(
    field: 'apiKey' | 'customBaseUrl' | 'modelName',
    value: string,
  ): void {
    settingsStore.updatePluginAgent({ [field]: value })
  }

  function updateAgentNumber(
    field: 'rpmLimit' | 'transportRetries' | 'businessRetries',
    value: number | null,
  ): void {
    if (value !== null) settingsStore.updatePluginAgent({ [field]: value })
  }

  function updateAgentBoolean(
    field: 'forceJsonOutput' | 'useStream',
    value: boolean,
  ): void {
    settingsStore.updatePluginAgent({ [field]: value })
  }

  function updateAgentExtraBody(value: Record<string, unknown> | undefined): void {
    settingsStore.updatePluginAgent({ extraBody: value })
  }

  function applyExamplePrompt(example: string): void {
    messageInput.value = example
  }

  async function beginConversation(): Promise<void> {
    let activeSessionId = session.value?.session_id || ''
    let hadExistingSession = Boolean(session.value?.session_id)
    try {
      if (!canBeginConversation.value) return
      const userContent = messageInput.value.trim()
      if (!userContent) {
        return
      }

      optimisticMessages.value = buildOptimisticPlanningMessages(userContent)
      isAwaitingPlanningReply.value = true
      messageInput.value = ''
      await syncHistoryScrollToBottom()

      if (!session.value) {
        const createResult = await createPluginAgentSession({
          mode: mode.value,
          ...(mode.value === 'modify' ? { plugin_id: selectedPluginId.value } : {}),
        })
        applySession(createResult)
        activeSessionId = createResult.session_id
        hadExistingSession = false
      }

      const activeSession = session.value
      if (!activeSession) {
        clearPlanningOptimisticState()
        toast.error('会话初始化失败')
        return
      }

      activeSessionId = activeSession.session_id

      const result = await sendPluginAgentMessage(activeSession.session_id, {
        content: userContent,
      })

      clearPlanningOptimisticState()
      applySession(result)
      await syncHistoryScrollToBottom()
    } catch (error) {
      if (activeSessionId) {
        try {
          await syncSessionFromServer(activeSessionId)
        } catch {
          if (!hadExistingSession) {
            applySession(null)
          }
        }
      } else if (!hadExistingSession) {
        applySession(null)
      }
      clearPlanningOptimisticState()
      toast.error(error instanceof Error ? error.message : '发送消息失败')
    }
  }

  async function lockTarget(): Promise<void> {
    if (isSessionCommandPending.value || !session.value?.pending_target) return
    isSessionCommandPending.value = true
    try {
      const result = await lockPluginAgentTarget(
        session.value.session_id,
        session.value.pending_target
      )
      applySession(result)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '锁定目标失败')
    } finally {
      isSessionCommandPending.value = false
    }
  }

  async function startExecution(): Promise<void> {
    if (isSessionCommandPending.value || !session.value || !canStartExecution.value) return
    isSessionCommandPending.value = true
    try {
      const result = await startPluginAgentExecution(session.value.session_id)
      applySession(result)
      await syncHistoryScrollToBottom()
      await startStreaming()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '启动执行失败')
    } finally {
      isSessionCommandPending.value = false
    }
  }

  function appendEvent(event: PluginAgentEvent): void {
    const key = event.eventKey ?? `session:${event.id}`
    if (eventFeed.value.some(existing => (existing.eventKey ?? `session:${existing.id}`) === key))
      return
    eventFeed.value = [...eventFeed.value, event]
    applyEventToSession(event)
  }

  function applyEventToSession(event: PluginAgentEvent): void {
    if (!session.value) return

    if (event.type === 'assistant_delta') {
      const payload = event.payload as PluginAgentAssistantDeltaPayload
      setAssistantDisplayTarget(payload.stream_id, payload.content || payload.delta || '')
      return
    }

    if (event.type === 'assistant') {
      const payload = event.payload as PluginAgentAssistantPayload
      if (payload.stream_id) {
        setAssistantDisplayTarget(payload.stream_id, payload.message)
      }
    }

    if (event.type === 'state') {
      const payload = event.payload as PluginAgentStatePayload
      if (payload.run_state) {
        session.value.run_state = payload.run_state
      }
      if (Object.prototype.hasOwnProperty.call(payload, 'locked_target')) {
        session.value.locked_target = payload.locked_target ?? null
      }
      if (Object.prototype.hasOwnProperty.call(payload, 'pending_target')) {
        session.value.pending_target = payload.pending_target ?? null
      }
      return
    }

    if (event.type === 'tool_result') {
      const payload = event.payload as PluginAgentToolResultPayload
      for (const filePath of payload.changed_files || []) {
        if (!session.value.touched_files.includes(filePath)) {
          session.value.touched_files.push(filePath)
        }
      }
      const previews = payload.file_previews || {}
      for (const [filePath, preview] of Object.entries(previews)) {
        session.value.file_previews[filePath] = preview
      }
      return
    }

    if (event.type === 'validation') {
      const payload = event.payload as PluginAgentValidationPayload
      session.value.last_validation = payload.details
      return
    }

    if (event.type === 'done') {
      const payload = event.payload as PluginAgentDonePayload
      session.value.run_state = payload.run_state
      session.value.last_validation = payload.validation || session.value.last_validation || null
      session.value.last_error = null
      return
    }

    if (event.type === 'error') {
      const payload = event.payload as PluginAgentErrorPayload
      session.value.run_state = payload.run_state
      session.value.last_error = payload.message
    }
  }

  async function applyJobEvent(event: V2JobEvent, generation: number): Promise<void> {
    if (
      generation !== streamGeneration ||
      event.jobId !== activeStreamJobId ||
      event.eventId <= jobEventCursor
    )
      return
    jobEventCursor = event.eventId
    const mapped = pluginAgentEventFromJobEvent(event)
    if (!mapped) return
    if (mapped.type === 'error' && sawPluginTerminalEvent) return
    if (mapped.type === 'done' || mapped.type === 'error') {
      sawPluginTerminalEvent = true
    }
    appendEvent(mapped)
    await syncHistoryScrollToBottom()
    if (mapped.type === 'done' || mapped.type === 'error') {
      stopStreaming()
    }
    if (mapped.type === 'done') {
      emit('pluginsChanged')
      await initializeModal({ preserveSessionWhenMissing: true })
    }
  }

  async function startStreaming(): Promise<void> {
    const activeSession = session.value
    if (!activeSession || !isRunning.value || !activeSession.job_id) return
    stopStreaming()

    const generation = streamGeneration
    activeStreamJobId = activeSession.job_id
    bufferedJobEvents = []
    jobEventCursor = 0
    sawPluginTerminalEvent = false
    streamIsCatchingUp = true
    stopTaskEvents = taskCenterStore.subscribeEvents(event => {
      if (event.jobId !== activeStreamJobId) return
      if (streamIsCatchingUp) {
        bufferedJobEvents.push(event)
        return
      }
      void applyJobEvent(event, generation)
    })

    try {
      const backlog = await listPluginAgentJobEvents(activeStreamJobId)
      if (generation !== streamGeneration) return
      jobEventCursor = backlog.cursor
      let backlogCompleted = false
      for (const event of backlog.events) {
        if (event.type === 'error' && sawPluginTerminalEvent) continue
        if (event.type === 'done' || event.type === 'error') sawPluginTerminalEvent = true
        if (event.type === 'done') backlogCompleted = true
        appendEvent(event)
      }
      streamIsCatchingUp = false
      const buffered = bufferedJobEvents.sort((left, right) => left.eventId - right.eventId)
      bufferedJobEvents = []
      for (const event of buffered) await applyJobEvent(event, generation)
      await syncHistoryScrollToBottom()
      if (backlogCompleted && generation === streamGeneration) {
        stopStreaming()
        emit('pluginsChanged')
        await initializeModal({ preserveSessionWhenMissing: true })
      } else if (sawPluginTerminalEvent && generation === streamGeneration) {
        stopStreaming()
      }
    } catch (error) {
      if (generation === streamGeneration) {
        stopStreaming()
        toast.error(error instanceof Error ? error.message : '任务事件加载失败')
      }
    }
  }

  async function cancelExecution(): Promise<void> {
    if (isSessionCommandPending.value || !session.value?.job_id) return
    isSessionCommandPending.value = true
    try {
      await cancelPluginAgentExecution(session.value.job_id)
      toast.info('已请求取消执行')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '取消执行失败')
    } finally {
      isSessionCommandPending.value = false
    }
  }

  async function clearSession(): Promise<void> {
    if (isSessionCommandPending.value || !session.value?.session_id) return
    isSessionCommandPending.value = true
    try {
      await deletePluginAgentSession(session.value.session_id)
      applySession(null)
      messageInput.value = ''
      stopStreaming()
      await initializeModal()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '清理会话失败')
    } finally {
      isSessionCommandPending.value = false
    }
  }

  function stopStreaming(): void {
    streamGeneration += 1
    stopTaskEvents?.()
    stopTaskEvents = null
    activeStreamJobId = ''
    bufferedJobEvents = []
    streamIsCatchingUp = false
  }

  const fetchModels = modelDiscovery.fetchModels

  async function testConnection(): Promise<void> {
    if (isTestingConnection.value) return
    const provider = agentSettings.value.provider
    const apiKey = agentSettings.value.apiKey.trim()
    const modelName = agentSettings.value.modelName.trim()
    const baseUrl = agentSettings.value.customBaseUrl.trim()
    if (
      providerRequiresApiKeyForBaseUrl(provider, baseUrl)
      && !apiKey
      && !hasStoredAgentCredential.value
    ) {
      toast.warning('请先填写 API Key')
      return
    }
    if (!modelName) {
      toast.warning('请填写模型名称')
      return
    }
    if (providerRequiresBaseUrl(provider) && !baseUrl) {
      toast.warning('自定义服务需要填写 Base URL')
      return
    }
    isTestingConnection.value = true
    try {
      const result = await testAiTranslateConnection({
        provider,
        apiKey,
        modelName,
        baseUrl,
        domain: 'plugin_agent',
      })
      if (result.success) {
        toast.success(result.message || '连接成功')
      } else {
        toast.error(result.message || '连接失败')
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '连接测试失败')
    } finally {
      isTestingConnection.value = false
    }
  }

  async function saveAgentSettings(): Promise<void> {
    if (isSavingAgentSettings.value) return
    isSavingAgentSettings.value = true
    try {
      const success = await settingsStore.savePluginAgentSettings()
      if (success) {
        emit('settingsSaved')
        toast.success('Agent 设置已保存')
      } else {
        toast.error('保存 Agent 设置失败')
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '保存 Agent 设置失败')
    } finally {
      isSavingAgentSettings.value = false
    }
  }

  function handleClose(): void {
    isOpen.value = false
  }

  function renderMarkdown(content: string): string {
    return sanitizeHtml(marked.parse(content) as string)
  }

  function getAssistantMessageContent(messageId: string, fallback: string): string {
    return assistantMessageDisplayContent.value[messageId] ?? fallback
  }

  function escapeHtml(content: string): string {
    return content
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
  }

  function formatEventPayload(payload: unknown): string {
    return JSON.stringify(payload, null, 2)
  }

  return {
    isOpen,
    mode,
    selectedPluginId,
    overview,
    overviewSections,
    promptExamples,
    providerOptions,
    pluginOptions,
    fetchedModels,
    session,
    messageInput,
    eventFeed,
    messagesContainer,
    isDebugExpanded,
    isFetchingModels,
    isTestingConnection,
    isSavingAgentSettings,
    isAwaitingPlanningReply,
    isSessionCommandPending,
    agentSettings,
    hasStoredAgentCredential,
    messages,
    modelListOptions,
    timelineItems,
    canBeginConversation,
    canLockTarget,
    canStartExecution,
    isRunning,
    isConversationPending,
    currentRunStateLabel,
    lockedTargetLabel,
    handleModeChange,
    handleSelectedPluginChange,
    handleProviderChange,
    handleModelSelected,
    updateAgentString,
    updateAgentNumber,
    updateAgentBoolean,
    updateAgentExtraBody,
    fetchModels,
    testConnection,
    saveAgentSettings,
    applyExamplePrompt,
    beginConversation,
    lockTarget,
    startExecution,
    cancelExecution,
    clearSession,
    handleClose,
    renderMarkdown,
    getAssistantMessageContent,
    escapeHtml,
    formatEventPayload,
  }
}
