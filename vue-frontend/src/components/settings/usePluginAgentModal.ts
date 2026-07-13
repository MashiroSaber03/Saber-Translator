import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { marked } from 'marked'
import { configApi } from '@/api/config'
import {
  cancelPluginAgentExecution,
  createPluginAgentSession,
  deletePluginAgentSession,
  getPluginAgentSession,
  getPluginAgentSettings,
  lockPluginAgentTarget,
  sendPluginAgentMessage,
  startPluginAgentExecution,
  subscribePluginAgentEvents,
  type PluginAgentAssistantDeltaPayload,
  type PluginAgentAssistantPayload,
  type PluginAgentDonePayload,
  type PluginAgentErrorPayload,
  type PluginAgentEvent,
  type PluginAgentAgentConfig,
  type PluginAgentOverviewSection,
  type PluginAgentSession,
  type PluginAgentStatePayload,
  type PluginAgentToolResultPayload,
  type PluginAgentValidationPayload,
} from '@/api/pluginAgent'
import { useSettingsStore } from '@/stores/settings'
import type { PluginAgentProvider } from '@/types/settings'
import { sanitizeHtml } from '@/utils/sanitizeHtml'
import { useToast } from '@/utils/toast'
import { useAiModelDiscovery, type AiModelDiscoveryMessageTone } from '@/composables/useAiModelDiscovery'
import { buildTimelineItems, type PluginAgentTimelineItem } from './pluginAgentTimeline'
import { usePluginAgentDisplayAnimation } from './usePluginAgentDisplayAnimation'

export interface PluginAgentModalProps {
  modelValue: boolean
}

export type PluginAgentModalEmit = {
  (e: 'update:modelValue', value: boolean): void
  (e: 'pluginsChanged'): void
}

export function usePluginAgentModal(props: PluginAgentModalProps, emit: PluginAgentModalEmit) {
  const settingsStore = useSettingsStore()
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
    completed: '已完成',
    failed: '执行失败',
    cancelled: '已取消',
  }
  const shouldAnimateAssistantStream = typeof navigator !== 'undefined'
    ? !navigator.userAgent.toLowerCase().includes('jsdom')
    : true

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
  const optimisticMessages = ref<PluginAgentConversationMessage[]>([])
  let streamAbortController: AbortController | null = null

  const localAgentSettings = ref({
    provider: settingsStore.settings.pluginAgent.provider,
    apiKey: settingsStore.settings.pluginAgent.apiKey,
    modelName: settingsStore.settings.pluginAgent.modelName,
    customBaseUrl: settingsStore.settings.pluginAgent.customBaseUrl,
    rpmLimit: settingsStore.settings.pluginAgent.openaiOptions.execution.rpmLimit,
    transportRetries: settingsStore.settings.pluginAgent.openaiOptions.execution.transportRetries,
    businessRetries: settingsStore.settings.pluginAgent.openaiOptions.execution.businessRetries,
    forceJsonOutput: settingsStore.settings.pluginAgent.openaiOptions.request.forceJsonOutput,
    useStream: settingsStore.settings.pluginAgent.openaiOptions.execution.useStream,
    extraBody: settingsStore.settings.pluginAgent.openaiOptions.request.extraBody,
  })
  function notifyModelDiscovery(message: string, tone: AiModelDiscoveryMessageTone): void {
    toast[tone](message)
  }
  const modelDiscovery = useAiModelDiscovery({
    source: () => ({
      provider: localAgentSettings.value.provider,
      apiKey: localAgentSettings.value.apiKey,
      baseUrl: localAgentSettings.value.customBaseUrl,
    }),
    notify: notifyModelDiscovery,
    requiresApiKey: () => false,
    emptyBaseUrl: '',
    errorMessage: error => error instanceof Error ? error.message : '获取模型失败',
  })
  const { isFetchingModels } = modelDiscovery
  const fetchedModels = computed(() => modelDiscovery.models.value.map(model => ({
    id: model.id,
    name: model.name || model.id,
  })))
  const displayAnimation = usePluginAgentDisplayAnimation({
    animate: shouldAnimateAssistantStream,
    onTick: syncHistoryScrollToBottom,
  })
  const {
    assistantMessageDisplayContent,
    assistantDisplayContent,
    assistantDisplayTargets,
  } = displayAnimation
  const clearAssistantDisplayAnimation = displayAnimation.clear
  const setAssistantDisplayTarget = displayAnimation.setStreamTarget
  function setAssistantMessageDisplayTarget(
    messageId: string,
    targetContent: string,
    options: { animate: boolean },
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
  const timelineItems = computed<PluginAgentTimelineItem[]>(() => (
    buildTimelineItems(eventFeed.value, assistantDisplayContent.value, assistantDisplayTargets.value)
  ))
  const canBeginConversation = computed(() => {
    if (mode.value === 'modify') {
      return Boolean(selectedPluginId.value && messageInput.value.trim())
    }
    return Boolean(messageInput.value.trim())
  })
  const canLockTarget = computed(() => mode.value === 'create' && Boolean(session.value?.pending_target))
  const canStartExecution = computed(() => (
    Boolean(session.value?.locked_target && session.value?.run_state === 'ready')
    && messages.value.some(message => message.role === 'user')
  ))
  const isRunning = computed(() => session.value?.run_state === 'running')
  const isConversationPending = computed(() => isRunning.value || isAwaitingPlanningReply.value)
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
    if ((!nextSession.messages || nextSession.messages.length === 0) && previousSession?.messages?.length) {
      session.value = {
        ...nextSession,
        messages: [...previousSession.messages],
      }
    }
    selectedPluginId.value = nextSession.selected_plugin_id || nextSession.locked_target?.plugin_id || selectedPluginId.value
    if (previousSession?.session_id === nextSession.session_id && eventFeed.value.length > 0) {
      const merged = [...eventFeed.value]
      for (const event of nextSession.events || []) {
        if (!merged.some(existing => existing.id === event.id)) {
          merged.push(event)
        }
      }
      eventFeed.value = merged.sort((left, right) => left.id - right.id)
    } else {
      eventFeed.value = [...(nextSession.events || [])]
    }

    const previousMessageIds = new Set((previousSession?.messages || []).map(message => message.id))
    const shouldAnimatePlanningMessages = previousSession?.session_id === nextSession.session_id
    for (const message of nextSession.messages || []) {
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

  function getLastEventId(): number {
    const events = eventFeed.value
    if (events.length === 0) return 0
    return events[events.length - 1]?.id || 0
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
    if (result.success) {
      applySession(result.session)
    }
  }

  watch(
    () => props.modelValue,
    async (value) => {
      isOpen.value = value
      if (value) {
        await initializeModal()
      } else {
        stopStreaming()
      }
    },
    { immediate: true },
  )

  watch(isOpen, (value) => {
    if (!value) {
      emit('update:modelValue', false)
    }
  })

  watch(() => localAgentSettings.value.apiKey, (value) => {
    settingsStore.updatePluginAgent({ apiKey: value })
  })
  watch(() => localAgentSettings.value.modelName, (value) => {
    settingsStore.updatePluginAgent({ modelName: value })
  })
  watch(() => localAgentSettings.value.customBaseUrl, (value) => {
    settingsStore.updatePluginAgent({ customBaseUrl: value })
  })
  watch(() => localAgentSettings.value.rpmLimit, (value) => {
    settingsStore.updatePluginAgent({ rpmLimit: value })
  })
  watch(() => localAgentSettings.value.transportRetries, (value) => {
    settingsStore.updatePluginAgent({ transportRetries: value })
  })
  watch(() => localAgentSettings.value.businessRetries, (value) => {
    settingsStore.updatePluginAgent({ businessRetries: value })
  })
  watch(() => localAgentSettings.value.forceJsonOutput, (value) => {
    settingsStore.updatePluginAgent({ forceJsonOutput: value })
  })
  watch(() => localAgentSettings.value.useStream, (value) => {
    settingsStore.updatePluginAgent({ useStream: value })
  })
  watch(() => localAgentSettings.value.extraBody, (value) => {
    settingsStore.updatePluginAgent({ extraBody: value })
  })

  watch(selectedPluginId, async (value, previousValue) => {
    if (
      mode.value === 'modify'
      && session.value?.session_id
      && value !== previousValue
    ) {
      try {
        await deletePluginAgentSession(session.value.session_id)
      } catch {
        // 忽略切换目标插件时的清理失败
      }
      applySession(null)
      messageInput.value = ''
      stopStreaming()
    }
  })

  onBeforeUnmount(() => {
    stopStreaming()
    clearAssistantDisplayAnimation()
    modelDiscovery.invalidate()
  })

  function buildAgentConfig(): PluginAgentAgentConfig {
    return {
      provider: localAgentSettings.value.provider,
      apiKey: localAgentSettings.value.apiKey,
      modelName: localAgentSettings.value.modelName,
      customBaseUrl: localAgentSettings.value.customBaseUrl,
      openaiOptions: settingsStore.settings.pluginAgent.openaiOptions,
    }
  }

  function syncLocalAgentSettingsFromStore(): void {
    localAgentSettings.value.provider = settingsStore.settings.pluginAgent.provider
    localAgentSettings.value.apiKey = settingsStore.settings.pluginAgent.apiKey
    localAgentSettings.value.modelName = settingsStore.settings.pluginAgent.modelName
    localAgentSettings.value.customBaseUrl = settingsStore.settings.pluginAgent.customBaseUrl
    localAgentSettings.value.rpmLimit = settingsStore.settings.pluginAgent.openaiOptions.execution.rpmLimit
    localAgentSettings.value.transportRetries = settingsStore.settings.pluginAgent.openaiOptions.execution.transportRetries
    localAgentSettings.value.businessRetries = settingsStore.settings.pluginAgent.openaiOptions.execution.businessRetries
    localAgentSettings.value.forceJsonOutput = settingsStore.settings.pluginAgent.openaiOptions.request.forceJsonOutput
    localAgentSettings.value.useStream = settingsStore.settings.pluginAgent.openaiOptions.execution.useStream
    localAgentSettings.value.extraBody = settingsStore.settings.pluginAgent.openaiOptions.request.extraBody
  }

  async function initializeModal(): Promise<void> {
    try {
      syncLocalAgentSettingsFromStore()
      modelDiscovery.invalidate()
      const result = await getPluginAgentSettings()
      if (!result.success) {
        toast.error(result.error || '加载插件 Agent 设置失败')
        return
      }

      overview.value = result.overview || []
      overviewSections.value = result.overview_sections || []
      promptExamples.value = result.prompt_examples || []
      providerOptions.value = result.providers || []
      pluginOptions.value = [{ value: '', label: '-- 选择插件 --' }, ...(result.plugins || []).map(plugin => ({
        value: plugin.id,
        label: plugin.display_name,
      }))]
      if (result.session) {
        applySession(result.session)
      }
      if (session.value) {
        selectedPluginId.value = session.value.selected_plugin_id || session.value.locked_target?.plugin_id || selectedPluginId.value
        if (session.value.run_state === 'running') {
          void startStreaming()
        }
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '加载插件 Agent 设置失败')
    }
  }

  async function handleModeChange(nextMode: 'create' | 'modify'): Promise<void> {
    if (nextMode === mode.value && !session.value) {
      return
    }
    if (session.value?.session_id) {
      try {
        await deletePluginAgentSession(session.value.session_id)
      } catch {
        // 忽略切换模式时的清理失败，避免阻断 UI
      }
    }
    mode.value = nextMode
    applySession(null)
    selectedPluginId.value = ''
    messageInput.value = ''
    stopStreaming()
  }

  function handleProviderChange(value: string | number): void {
    modelDiscovery.invalidate()
    const provider = String(value || '') as PluginAgentProvider
    localAgentSettings.value.provider = provider
    settingsStore.setPluginAgentProvider(provider)
    localAgentSettings.value.apiKey = settingsStore.settings.pluginAgent.apiKey
    localAgentSettings.value.modelName = settingsStore.settings.pluginAgent.modelName
    localAgentSettings.value.customBaseUrl = settingsStore.settings.pluginAgent.customBaseUrl
    localAgentSettings.value.rpmLimit = settingsStore.settings.pluginAgent.openaiOptions.execution.rpmLimit
    localAgentSettings.value.transportRetries = settingsStore.settings.pluginAgent.openaiOptions.execution.transportRetries
    localAgentSettings.value.businessRetries = settingsStore.settings.pluginAgent.openaiOptions.execution.businessRetries
    localAgentSettings.value.forceJsonOutput = settingsStore.settings.pluginAgent.openaiOptions.request.forceJsonOutput
    localAgentSettings.value.useStream = settingsStore.settings.pluginAgent.openaiOptions.execution.useStream
    localAgentSettings.value.extraBody = settingsStore.settings.pluginAgent.openaiOptions.request.extraBody
  }

  function handleSelectedPluginChange(value: string | number): void {
    selectedPluginId.value = String(value || '')
  }

  function handleModelSelected(value: string | number): void {
    localAgentSettings.value.modelName = String(value || '')
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
        if (!createResult.success) {
          clearPlanningOptimisticState()
          toast.error(createResult.error || '创建会话失败')
          return
        }
        applySession(createResult.session)
        activeSessionId = createResult.session.session_id
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
        agentConfig: buildAgentConfig(),
      })

      if (!result.success) {
        clearPlanningOptimisticState()
        toast.error(result.error || '发送消息失败')
        return
      }

      clearPlanningOptimisticState()
      applySession(result.session)
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
    try {
      if (!session.value?.pending_target) return

      const result = await lockPluginAgentTarget(session.value.session_id, session.value.pending_target)
      if (!result.success) {
        toast.error(result.error || '锁定目标失败')
        return
      }
      applySession(result.session)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '锁定目标失败')
    }
  }

  async function startExecution(): Promise<void> {
    try {
      if (!session.value || !canStartExecution.value) return

      const result = await startPluginAgentExecution(session.value.session_id, buildAgentConfig())
      if (!result.success) {
        toast.error(result.error || '启动执行失败')
        return
      }

      applySession(result.session)
      await syncHistoryScrollToBottom()
      await startStreaming()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '启动执行失败')
    }
  }

  function appendEvent(event: PluginAgentEvent): void {
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

  async function startStreaming(): Promise<void> {
    if (!session.value) return
    stopStreaming()

    streamAbortController = new AbortController()
    while (session.value?.run_state === 'running' && !streamAbortController.signal.aborted) {
      const activeSession = session.value
      if (!activeSession) break
      await subscribePluginAgentEvents(activeSession.session_id, {
        afterId: getLastEventId(),
        signal: streamAbortController.signal,
        onEvent: async (event) => {
          appendEvent(event)
          await syncHistoryScrollToBottom()
          if (event.type === 'done') {
            emit('pluginsChanged')
            await initializeModal()
          }
        },
        onError: (error) => {
          if (!streamAbortController?.signal.aborted) {
            toast.error(error)
          }
        },
      })
    }
  }

  async function cancelExecution(): Promise<void> {
    try {
      if (!session.value?.session_id) return
      await cancelPluginAgentExecution(session.value.session_id)
      toast.info('已请求取消执行')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '取消执行失败')
    }
  }

  async function clearSession(): Promise<void> {
    try {
      if (!session.value?.session_id) return
      await deletePluginAgentSession(session.value.session_id)
      applySession(null)
      messageInput.value = ''
      stopStreaming()
      await initializeModal()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '清理会话失败')
    }
  }

  function stopStreaming(): void {
    streamAbortController?.abort()
    streamAbortController = null
  }

  const fetchModels = modelDiscovery.fetchModels

  async function testConnection(): Promise<void> {
    isTestingConnection.value = true
    try {
      const result = await configApi.testAiTranslateConnection({
        provider: localAgentSettings.value.provider,
        apiKey: localAgentSettings.value.apiKey,
        modelName: localAgentSettings.value.modelName,
        baseUrl: localAgentSettings.value.customBaseUrl,
      })
      if (result.success) {
        toast.success(result.message || '连接成功')
      } else {
        toast.error(result.message || result.error || '连接失败')
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '连接测试失败')
    } finally {
      isTestingConnection.value = false
    }
  }

  async function saveAgentSettings(): Promise<void> {
    isSavingAgentSettings.value = true
    try {
      const success = await settingsStore.savePluginAgentSettings()
      if (success) {
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
    localAgentSettings,
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
