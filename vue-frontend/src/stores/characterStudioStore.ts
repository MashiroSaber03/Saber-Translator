import { defineStore } from 'pinia'
import { computed, nextTick, ref } from 'vue'
import {
  createCharacterStudioChatSession,
  createCharacterStudioDocument,
  deleteCharacterStudioChatMessage,
  deleteCharacterStudioDocument,
  downloadCharacterStudioExport,
  exportCharacterStudioChatSession,
  downloadCharacterStudioWorldbook,
  editCharacterStudioChatMessage,
  generateCharacterStudioSection,
  getCharacterStudioChatPromptPreview,
  getCharacterStudioChatState,
  getCharacterStudioDocument,
  getCharacterStudioIndex,
  importCharacterStudioFile,
  importCharacterStudioChatSession,
  importWorldbookIntoCharacterStudioDocument,
  regenerateCharacterStudioChatMessage,
  runCharacterStudioAgent,
  saveCharacterStudioDocument,
  streamCharacterStudioChatMessage,
  summarizeCharacterStudioChatSession,
  switchCharacterStudioChatSession,
  validateCharacterStudioDocument,
} from '@/api/characterStudio'
import { applyCharacterStudioAgentPatch } from '@/stores/characterStudioPatch'
import type {
  CharacterStudioAgentPatchV2,
  CharacterStudioChatAttachment,
  CharacterStudioChatMessage,
  CharacterStudioChatSession,
  CharacterStudioChatSessionSummary,
  CharacterStudioCandidate,
  CharacterStudioDocument,
  CharacterStudioEditorPendingState,
  CharacterStudioGreetingOption,
  CharacterStudioSummary,
  ExportDiagnostic,
} from '@/types/characterStudio'

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

export const useCharacterStudioStore = defineStore('character-studio', () => {
  const bookId = ref('')
  const documents = ref<CharacterStudioSummary[]>([])
  const candidates = ref<CharacterStudioCandidate[]>([])
  const hasTimeline = ref(true)
  const currentDocument = ref<CharacterStudioDocument | null>(null)
  const activeChatSession = ref<CharacterStudioChatSession | null>(null)
  const archivedChatSessions = ref<CharacterStudioChatSessionSummary[]>([])
  const availableGreetings = ref<CharacterStudioGreetingOption[]>([])
  const chatPromptPreview = ref('')
  const chatPromptPreviewError = ref('')
  const diagnostics = ref<ExportDiagnostic | null>(null)
  const agentMessages = ref<Array<{ role: 'user' | 'assistant'; content: string }>>([])
  const agentHtmlPreview = ref('')
  const pendingAgentPatch = ref<CharacterStudioAgentPatchV2 | null>(null)
  const activeEditorTab = ref<'overview' | 'character' | 'greetings' | 'lorebook' | 'scripts' | 'export'>('overview')
  const activeScriptTab = ref<'regex' | 'tasks'>('regex')
  const resourcePanelOpen = ref(false)
  const activeWorkspaceTab = ref<'chat' | 'assistant' | 'runtime'>('chat')
  const isWorkspaceLoading = ref(false)
  const isDocumentLoading = ref(false)
  const isSaving = ref(false)
  const isChatLoading = ref(false)
  const isChatStreaming = ref(false)
  const isChatMutating = ref(false)
  const isChatSummarizing = ref(false)
  const isChatImporting = ref(false)
  const isChatExporting = ref(false)
  const isChatPromptLoading = ref(false)
  const isAgentBusy = ref(false)
  const isCreatingManual = ref(false)
  const isImportingFile = ref(false)
  const isImportingWorldbook = ref(false)
  const isDeleting = ref(false)
  const isValidating = ref(false)
  const openingDocumentId = ref('')
  const creatingCandidateName = ref('')
  const generatingSection = ref<string | null>(null)
  const downloadingFormat = ref<string | null>(null)
  const errorMessage = ref('')
  const selectedLibrarySearch = ref('')
  const _suspendAutosave = ref(false)
  const lastSyncedFingerprint = ref('')
  let autosaveTimer: ReturnType<typeof setTimeout> | null = null
  const patchSnapshot = ref<CharacterStudioDocument | null>(null)
  let chatAbortController: AbortController | null = null
  let chatStreamRunId = 0

  const canUndoPatch = computed(() => patchSnapshot.value !== null)
  const editorPendingState = computed<CharacterStudioEditorPendingState>(() => ({
    generatingSection: generatingSection.value,
    validating: isValidating.value,
    importingWorldbook: isImportingWorldbook.value,
    deleting: isDeleting.value,
    saving: isSaving.value,
    downloadingFormat: downloadingFormat.value,
  }))

  const hasBusyAction = computed(() => [
    isWorkspaceLoading.value,
    isDocumentLoading.value,
    isSaving.value,
    isChatLoading.value,
    isChatStreaming.value,
    isChatMutating.value,
    isChatSummarizing.value,
    isChatImporting.value,
    isChatExporting.value,
    isChatPromptLoading.value,
    isAgentBusy.value,
    isCreatingManual.value,
    isImportingFile.value,
    isImportingWorldbook.value,
    isDeleting.value,
    isValidating.value,
    !!openingDocumentId.value,
    !!creatingCandidateName.value,
    !!generatingSection.value,
    !!downloadingFormat.value,
  ].some(Boolean))

  const activeActionLabel = computed(() => {
    if (isDocumentLoading.value) return '正在打开角色文档'
    if (openingDocumentId.value) return '正在切换角色文档'
    if (isWorkspaceLoading.value) return '正在加载角色工坊'
    if (isCreatingManual.value) return '正在新建角色文档'
    if (creatingCandidateName.value) return `正在从候选创建「${creatingCandidateName.value}」`
    if (isImportingFile.value) return '正在导入角色卡'
    if (isImportingWorldbook.value) return '正在导入世界书'
    if (isChatLoading.value) return '正在加载聊天会话'
    if (isChatStreaming.value) return '正在生成聊天回复'
    if (isChatMutating.value) return '正在处理聊天记录'
    if (isChatSummarizing.value) return '正在总结聊天'
    if (isChatImporting.value) return '正在导入聊天记录'
    if (isChatExporting.value) return '正在导出聊天记录'
    if (isChatPromptLoading.value) return '正在加载提示词预览'
    if (generatingSection.value) {
      return {
        full: '正在补全整张角色卡',
        identity: '正在补全角色设定',
        review: '正在审查当前角色',
        translate: '正在翻译整卡',
        greetings: '正在生成问候语',
        lorebook: '正在生成世界书',
        regex: '正在生成正则脚本',
        'state-tasks': '正在生成状态任务',
      }[generatingSection.value] || '正在生成内容'
    }
    if (isValidating.value) return '正在执行角色诊断'
    if (isSaving.value) return '正在保存角色文档'
    if (downloadingFormat.value) {
      return {
        v3: '正在导出 V3 JSON',
        v2: '正在导出 V2 JSON',
        png: '正在导出 PNG',
        worldbook: '正在导出世界书',
      }[downloadingFormat.value] || '正在导出文件'
    }
    if (isDeleting.value) return '正在删除角色文档'
    if (isAgentBusy.value) return '正在请求卡片助手'
    return ''
  })

  const filteredDocuments = computed(() => {
    const keyword = selectedLibrarySearch.value.trim().toLowerCase()
    if (!keyword) return documents.value
    return documents.value.filter(item => {
      return (
        item.title.toLowerCase().includes(keyword) ||
        item.tags.some(tag => tag.toLowerCase().includes(keyword)) ||
        (item.source_character || '').toLowerCase().includes(keyword)
      )
    })
  })

  const filteredCandidates = computed(() => {
    const keyword = selectedLibrarySearch.value.trim().toLowerCase()
    if (!keyword) return candidates.value
    return candidates.value.filter(item => {
      return (
        item.name.toLowerCase().includes(keyword) ||
        item.aliases.some(alias => alias.toLowerCase().includes(keyword))
      )
    })
  })

  function resetChatState() {
    activeChatSession.value = null
    archivedChatSessions.value = []
    availableGreetings.value = []
    chatPromptPreview.value = ''
    chatPromptPreviewError.value = ''
  }

  function abortActiveChatStream() {
    if (chatAbortController) {
      chatAbortController.abort()
      chatAbortController = null
    }
  }

  function resetWorkspaceState() {
    currentDocument.value = null
    markDocumentSynced(null)
    resetChatState()
    diagnostics.value = null
    agentMessages.value = []
    agentHtmlPreview.value = ''
    pendingAgentPatch.value = null
    patchSnapshot.value = null
    activeEditorTab.value = 'overview'
    activeScriptTab.value = 'regex'
    activeWorkspaceTab.value = 'chat'
    abortActiveChatStream()
    if (autosaveTimer) {
      clearTimeout(autosaveTimer)
      autosaveTimer = null
    }
  }

  async function loadWorkspace(nextBookId: string) {
    if (!nextBookId) return
    const isBookChanged = !!bookId.value && bookId.value !== nextBookId
    isWorkspaceLoading.value = true
    errorMessage.value = ''
    if (isBookChanged) {
      resetWorkspaceState()
    }
    bookId.value = nextBookId
    try {
      const response = await getCharacterStudioIndex(nextBookId)
      if (!response.success) {
        throw new Error(response.error || '加载角色工坊失败')
      }
      documents.value = response.documents || []
      candidates.value = response.candidates || []
      hasTimeline.value = response.has_timeline !== false
      if (
        currentDocument.value &&
        !documents.value.some(item => item.id === currentDocument.value?.id)
      ) {
        resetWorkspaceState()
      }
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : '加载角色工坊失败'
    } finally {
      isWorkspaceLoading.value = false
    }
  }

  async function openDocument(docId: string) {
    if (!bookId.value || !docId) return
    const requestedBookId = bookId.value
    isDocumentLoading.value = true
    openingDocumentId.value = docId
    clearErrorMessage()
    try {
      const response = await getCharacterStudioDocument(requestedBookId, docId)
      if (!response.success || !response.document) {
        throw new Error(response.error || '加载角色文档失败')
      }
      const document = response.document
      if (bookId.value !== requestedBookId) return
      await runWithoutAutosave(async () => {
        abortActiveChatStream()
        currentDocument.value = document
        markDocumentSynced(document)
        resetChatState()
        diagnostics.value = null
        agentMessages.value = []
        pendingAgentPatch.value = null
        agentHtmlPreview.value = ''
        patchSnapshot.value = null
        activeEditorTab.value = 'overview'
        activeScriptTab.value = 'regex'
      })
      try {
        await loadChatState(docId)
      } catch (chatError) {
        errorMessage.value = chatError instanceof Error ? chatError.message : '加载聊天状态失败'
      }
    } catch (error) {
      throw createActionError(error, '加载角色文档失败')
    } finally {
      isDocumentLoading.value = false
      openingDocumentId.value = ''
    }
  }

  async function createManualDocument(title: string = '新角色') {
    if (!bookId.value) return
    isCreatingManual.value = true
    clearErrorMessage()
    try {
      const response = await createCharacterStudioDocument(bookId.value, { title })
      if (!response.success || !response.document) {
        throw new Error(response.error || '创建角色失败')
      }
      const document = response.document
      await loadWorkspace(bookId.value)
      await openDocument(document.id)
    } catch (error) {
      throw createActionError(error, '创建角色失败')
    } finally {
      isCreatingManual.value = false
    }
  }

  async function createDocumentFromCandidate(candidateName: string) {
    if (!bookId.value) return
    creatingCandidateName.value = candidateName
    clearErrorMessage()
    try {
      const response = await createCharacterStudioDocument(bookId.value, { candidate_name: candidateName })
      if (!response.success || !response.document) {
        throw new Error(response.error || '创建角色失败')
      }
      const document = response.document
      await loadWorkspace(bookId.value)
      await openDocument(document.id)
    } catch (error) {
      throw createActionError(error, '创建角色失败')
    } finally {
      creatingCandidateName.value = ''
    }
  }

  async function persistCurrentDocument() {
    if (!bookId.value || !currentDocument.value) return
    if (autosaveTimer) {
      clearTimeout(autosaveTimer)
      autosaveTimer = null
    }
    if (isSaving.value) return
    isSaving.value = true
    clearErrorMessage()
    try {
      const response = await saveCharacterStudioDocument(bookId.value, currentDocument.value.id, currentDocument.value as unknown as Record<string, unknown>)
      if (!response.success || !response.document) {
        throw new Error(response.error || '保存失败')
      }
      const document = response.document
      await runWithoutAutosave(async () => {
        currentDocument.value = document
        markDocumentSynced(document)
      })
      await loadWorkspace(bookId.value)
    } catch (error) {
      throw createActionError(error, '保存失败')
    } finally {
      isSaving.value = false
    }
  }

  function scheduleAutosave() {
    if (_suspendAutosave.value || !currentDocument.value) return
    if (autosaveTimer) clearTimeout(autosaveTimer)
    autosaveTimer = setTimeout(() => {
      void persistCurrentDocument().catch(error => {
        errorMessage.value = error instanceof Error ? error.message : '自动保存失败'
      })
    }, 800)
  }

  function buildAutosaveFingerprint(document: CharacterStudioDocument | null) {
    if (!document) return ''
    const snapshot = JSON.parse(JSON.stringify(document)) as CharacterStudioDocument
    snapshot.meta.updated_at = ''
    snapshot.status.last_validated_at = null
    return JSON.stringify(snapshot)
  }

  function markDocumentSynced(document: CharacterStudioDocument | null) {
    lastSyncedFingerprint.value = buildAutosaveFingerprint(document)
  }

  function clearErrorMessage() {
    errorMessage.value = ''
  }

  function createActionError(error: unknown, fallback: string): Error {
    const normalized = error instanceof Error ? error : new Error(fallback)
    errorMessage.value = normalized.message || fallback
    return normalized
  }

  function updateCurrentDocument(nextDocument: CharacterStudioDocument | null) {
    currentDocument.value = nextDocument
    if (!nextDocument) return
    if (buildAutosaveFingerprint(nextDocument) === lastSyncedFingerprint.value) return
    scheduleAutosave()
  }

  async function runWithoutAutosave(callback: () => void | Promise<void>) {
    _suspendAutosave.value = true
    if (autosaveTimer) {
      clearTimeout(autosaveTimer)
      autosaveTimer = null
    }
    try {
      await callback()
    } finally {
      await nextTick()
      _suspendAutosave.value = false
    }
  }

  async function deleteCurrentDocument() {
    if (!bookId.value || !currentDocument.value) return
    const docId = currentDocument.value.id
    isDeleting.value = true
    clearErrorMessage()
    try {
      const response = await deleteCharacterStudioDocument(bookId.value, docId)
      if (!response.success) {
        throw new Error(response.error || '删除失败')
      }
      currentDocument.value = null
      markDocumentSynced(null)
      diagnostics.value = null
      agentMessages.value = []
      pendingAgentPatch.value = null
      agentHtmlPreview.value = ''
      patchSnapshot.value = null
      await loadWorkspace(bookId.value)
    } catch (error) {
      throw createActionError(error, '删除失败')
    } finally {
      isDeleting.value = false
    }
  }

  async function generateSection(section: string) {
    if (!bookId.value || !currentDocument.value) return
    generatingSection.value = section
    clearErrorMessage()
    try {
      const response = await generateCharacterStudioSection(bookId.value, currentDocument.value.id, section)
      if (!response.success || !response.document) {
        throw new Error(response.error || '生成失败')
      }
      const document = response.document
      await runWithoutAutosave(async () => {
        currentDocument.value = document
        markDocumentSynced(document)
      })
      await loadWorkspace(bookId.value)
    } catch (error) {
      throw createActionError(error, '生成失败')
    } finally {
      generatingSection.value = null
    }
  }

  async function validateCurrentDocument() {
    if (!bookId.value || !currentDocument.value) return
    isValidating.value = true
    clearErrorMessage()
    try {
      const response = await validateCharacterStudioDocument(bookId.value, currentDocument.value.id)
      if (!response.success) {
        throw new Error(response.error || '诊断失败')
      }
      diagnostics.value = {
        valid: response.valid,
        errors: response.errors || [],
        warnings: response.warnings || [],
        checks: response.checks || {},
      }
    } catch (error) {
      throw createActionError(error, '诊断失败')
    } finally {
      isValidating.value = false
    }
  }

  function extractPatch(content: string): CharacterStudioAgentPatchV2 | null {
    const match = content.match(/```json:patch\s*([\s\S]*?)```/i)
    if (!match) return null
    try {
      return JSON.parse(match[1]!.trim()) as CharacterStudioAgentPatchV2
    } catch {
      return null
    }
  }

  function extractHtmlPreview(content: string): string {
    const match = content.match(/```html\s*([\s\S]*?)```/i)
    return match?.[1]?.trim() || ''
  }

  function cloneDocument(document: CharacterStudioDocument): CharacterStudioDocument {
    return JSON.parse(JSON.stringify(document)) as CharacterStudioDocument
  }

  function applyPendingPatch() {
    if (!currentDocument.value || !pendingAgentPatch.value) return
    clearErrorMessage()
    try {
      const nextDocument = applyCharacterStudioAgentPatch(currentDocument.value, pendingAgentPatch.value)
      if (buildAutosaveFingerprint(nextDocument) === buildAutosaveFingerprint(currentDocument.value)) {
        pendingAgentPatch.value = null
        return
      }
      patchSnapshot.value = cloneDocument(currentDocument.value)
      currentDocument.value = nextDocument
      pendingAgentPatch.value = null
      scheduleAutosave()
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : '应用 patch 失败'
    }
  }

  function undoLastPatch() {
    if (!patchSnapshot.value) return
    currentDocument.value = patchSnapshot.value
    patchSnapshot.value = null
    pendingAgentPatch.value = null
    scheduleAutosave()
  }

  async function sendAgentMessage(message: string) {
    if (!bookId.value || !currentDocument.value || !message.trim()) return
    isAgentBusy.value = true
    clearErrorMessage()
    agentMessages.value.push({ role: 'user', content: message })
    try {
      const response = await runCharacterStudioAgent(bookId.value, currentDocument.value.id, message)
      if (!response.success) {
        throw new Error(response.error || 'Agent 调用失败')
      }
      const content = response.content || ''
      agentMessages.value.push({ role: 'assistant', content })
      pendingAgentPatch.value = extractPatch(content)
      agentHtmlPreview.value = extractHtmlPreview(content)
    } catch (error) {
      throw createActionError(error, 'Agent 调用失败')
    } finally {
      isAgentBusy.value = false
    }
  }

  function applyChatStatePayload(payload: {
    active_session?: CharacterStudioChatSession
    archived_sessions?: CharacterStudioChatSessionSummary[]
    available_greetings?: CharacterStudioGreetingOption[]
    session?: CharacterStudioChatSession
    prompt_preview?: string
  }) {
    const nextSession = payload.active_session || payload.session || null
    if (nextSession) {
      if (activeChatSession.value?.session_id !== nextSession.session_id) {
        chatPromptPreview.value = ''
      }
      activeChatSession.value = nextSession
    }
    if (payload.archived_sessions) {
      archivedChatSessions.value = payload.archived_sessions
    }
    if (payload.available_greetings) {
      availableGreetings.value = payload.available_greetings
    }
    if (typeof payload.prompt_preview === 'string') {
      chatPromptPreview.value = payload.prompt_preview
    }
  }

  async function loadChatState(docId: string) {
    if (!bookId.value || !docId) return
    isChatLoading.value = true
    clearErrorMessage()
    try {
      const response = await getCharacterStudioChatState(bookId.value, docId)
      if (!response.success) {
        throw new Error(response.error || '加载聊天状态失败')
      }
      applyChatStatePayload(response)
    } catch (error) {
      throw createActionError(error, '加载聊天状态失败')
    } finally {
      isChatLoading.value = false
    }
  }

  async function createChatSession(greetingId?: string) {
    if (!bookId.value || !currentDocument.value) return
    abortActiveChatStream()
    isChatMutating.value = true
    clearErrorMessage()
    try {
      const response = await createCharacterStudioChatSession(bookId.value, currentDocument.value.id, greetingId)
      if (!response.success) {
        throw new Error(response.error || '创建聊天会话失败')
      }
      applyChatStatePayload(response)
    } catch (error) {
      throw createActionError(error, '创建聊天会话失败')
    } finally {
      isChatMutating.value = false
    }
  }

  async function switchChatSession(sessionId: string) {
    if (!bookId.value || !currentDocument.value || !sessionId) return
    abortActiveChatStream()
    isChatMutating.value = true
    clearErrorMessage()
    try {
      const response = await switchCharacterStudioChatSession(bookId.value, currentDocument.value.id, sessionId)
      if (!response.success) {
        throw new Error(response.error || '切换聊天会话失败')
      }
      applyChatStatePayload(response)
    } catch (error) {
      throw createActionError(error, '切换聊天会话失败')
    } finally {
      isChatMutating.value = false
    }
  }

  function createOptimisticAttachment(file: File): CharacterStudioChatAttachment {
    return {
      attachment_id: `temp-att-${Date.now()}-${Math.random().toString(16).slice(2, 6)}`,
      filename: file.name,
      mime_type: file.type || 'application/octet-stream',
      asset_path: URL.createObjectURL(file),
      created_at: new Date().toISOString(),
    }
  }

  function revokeAttachmentUrls(attachments: CharacterStudioChatAttachment[]) {
    attachments.forEach(item => {
      if (item.asset_path?.startsWith('blob:')) {
        URL.revokeObjectURL(item.asset_path)
      }
    })
  }

  function revokeOptimisticSessionAssets(session: CharacterStudioChatSession | null) {
    if (!session) return
    session.messages.forEach(message => revokeAttachmentUrls(message.attachments || []))
  }

  function createOptimisticMessage(
    role: 'user' | 'assistant',
    content: string,
    attachments: CharacterStudioChatAttachment[] = [],
  ): CharacterStudioChatMessage {
    const now = new Date().toISOString()
    return {
      message_id: `temp-msg-${Date.now()}-${Math.random().toString(16).slice(2, 6)}`,
      role,
      content,
      attachments,
      runtime_log: [],
      variables_snapshot: activeChatSession.value?.variables || {},
      generation_meta: {},
      created_at: now,
      updated_at: now,
    }
  }

  async function sendChatMessage(content: string, attachments: File[] = []) {
    if (!bookId.value || !currentDocument.value || !activeChatSession.value) return
    if (!content.trim() && attachments.length === 0) return
    if (chatAbortController) {
      chatAbortController.abort()
      chatAbortController = null
    }
    const controller = new AbortController()
    chatAbortController = controller
    const streamRunId = ++chatStreamRunId
    isChatStreaming.value = true
    clearErrorMessage()
    activeWorkspaceTab.value = 'chat'
    const previousSession = JSON.parse(JSON.stringify(activeChatSession.value)) as CharacterStudioChatSession
    const optimisticSession = JSON.parse(JSON.stringify(activeChatSession.value)) as CharacterStudioChatSession
    optimisticSession.messages.push(
      createOptimisticMessage('user', content, attachments.map(createOptimisticAttachment)),
      createOptimisticMessage('assistant', ''),
    )
    activeChatSession.value = optimisticSession
    try {
      await streamCharacterStudioChatMessage(bookId.value, currentDocument.value.id, {
        sessionId: previousSession.session_id,
        content,
        attachments,
        signal: controller.signal,
        onEvent: event => {
          if (event.type === 'assistant_delta' && activeChatSession.value) {
            const lastMessage = activeChatSession.value.messages[activeChatSession.value.messages.length - 1]
            if (lastMessage && lastMessage.role === 'assistant') {
              lastMessage.content = event.content
            }
          } else if (event.type === 'runtime' && activeChatSession.value) {
            const lastMessage = activeChatSession.value.messages[activeChatSession.value.messages.length - 1]
            if (lastMessage && lastMessage.role === 'assistant') {
              lastMessage.runtime_log = event.runtime_log
              lastMessage.variables_snapshot = event.variables
            }
          } else if (event.type === 'state') {
            revokeOptimisticSessionAssets(activeChatSession.value)
            applyChatStatePayload({ session: event.session as CharacterStudioChatSession })
          } else if (event.type === 'error') {
            errorMessage.value = event.message
          }
        },
      })
    } catch (error) {
      if (controller.signal.aborted) return
      revokeOptimisticSessionAssets(activeChatSession.value)
      activeChatSession.value = previousSession
      throw createActionError(error, '发送聊天消息失败')
    } finally {
      if (chatAbortController === controller) {
        chatAbortController = null
      }
      if (streamRunId === chatStreamRunId) {
        isChatStreaming.value = false
      }
    }
  }

  async function editChatMessage(messageId: string, content: string) {
    if (!bookId.value || !currentDocument.value || !activeChatSession.value) return
    isChatMutating.value = true
    clearErrorMessage()
    try {
      const response = await editCharacterStudioChatMessage(
        bookId.value,
        currentDocument.value.id,
        activeChatSession.value.session_id,
        messageId,
        content,
      )
      if (!response.success) {
        throw new Error(response.error || '编辑消息失败')
      }
      applyChatStatePayload({ session: response.session as CharacterStudioChatSession })
    } catch (error) {
      throw createActionError(error, '编辑消息失败')
    } finally {
      isChatMutating.value = false
    }
  }

  async function deleteChatMessage(messageId: string) {
    if (!bookId.value || !currentDocument.value || !activeChatSession.value) return
    isChatMutating.value = true
    clearErrorMessage()
    try {
      const response = await deleteCharacterStudioChatMessage(
        bookId.value,
        currentDocument.value.id,
        activeChatSession.value.session_id,
        messageId,
      )
      if (!response.success) {
        throw new Error(response.error || '删除消息失败')
      }
      applyChatStatePayload({ session: response.session as CharacterStudioChatSession })
    } catch (error) {
      throw createActionError(error, '删除消息失败')
    } finally {
      isChatMutating.value = false
    }
  }

  async function regenerateChatMessage(messageId: string) {
    if (!bookId.value || !currentDocument.value || !activeChatSession.value) return
    if (chatAbortController) {
      chatAbortController.abort()
      chatAbortController = null
    }
    const controller = new AbortController()
    chatAbortController = controller
    const streamRunId = ++chatStreamRunId
    isChatStreaming.value = true
    clearErrorMessage()
    const previousSession = JSON.parse(JSON.stringify(activeChatSession.value)) as CharacterStudioChatSession
    const anchorIndex = previousSession.messages.findIndex(item => item.message_id === messageId)
    if (anchorIndex >= 0) {
      let userIndex = anchorIndex
      if (previousSession.messages[anchorIndex]?.role === 'assistant') {
        userIndex = previousSession.messages
          .slice(0, anchorIndex)
          .map((item, index) => ({ item, index }))
          .reverse()
          .find(({ item }) => item.role === 'user')?.index ?? anchorIndex
      }
      const optimisticSession = JSON.parse(JSON.stringify(previousSession)) as CharacterStudioChatSession
      optimisticSession.messages = optimisticSession.messages.slice(0, userIndex + 1)
      optimisticSession.messages.push(createOptimisticMessage('assistant', ''))
      activeChatSession.value = optimisticSession
    }
    try {
      await regenerateCharacterStudioChatMessage(
        bookId.value,
        currentDocument.value.id,
        activeChatSession.value.session_id,
        messageId,
        event => {
          if (event.type === 'assistant_delta') {
            if (!activeChatSession.value) return
            const lastMessage = activeChatSession.value.messages[activeChatSession.value.messages.length - 1]
            if (lastMessage && lastMessage.role === 'assistant') {
              lastMessage.content = event.content
            }
          } else if (event.type === 'runtime' && activeChatSession.value) {
            const lastMessage = activeChatSession.value.messages[activeChatSession.value.messages.length - 1]
            if (lastMessage && lastMessage.role === 'assistant') {
              lastMessage.runtime_log = event.runtime_log
              lastMessage.variables_snapshot = event.variables
            }
          } else if (event.type === 'state') {
            revokeOptimisticSessionAssets(activeChatSession.value)
            applyChatStatePayload({ session: event.session as CharacterStudioChatSession })
          } else if (event.type === 'error') {
            errorMessage.value = event.message
          }
        },
        controller.signal,
      )
    } catch (error) {
      if (controller.signal.aborted) return
      revokeOptimisticSessionAssets(activeChatSession.value)
      activeChatSession.value = previousSession
      throw createActionError(error, '消息重生失败')
    } finally {
      if (chatAbortController === controller) {
        chatAbortController = null
      }
      if (streamRunId === chatStreamRunId) {
        isChatStreaming.value = false
      }
    }
  }

  async function summarizeChatSession(cutoffMessageId?: string) {
    if (!bookId.value || !currentDocument.value || !activeChatSession.value) return
    isChatSummarizing.value = true
    clearErrorMessage()
    try {
      const response = await summarizeCharacterStudioChatSession(
        bookId.value,
        currentDocument.value.id,
        activeChatSession.value.session_id,
        cutoffMessageId,
      )
      if (!response.success) {
        throw new Error(response.error || '总结聊天失败')
      }
      applyChatStatePayload({ session: response.session as CharacterStudioChatSession })
    } catch (error) {
      throw createActionError(error, '总结聊天失败')
    } finally {
      isChatSummarizing.value = false
    }
  }

  async function exportChatSession() {
    if (!bookId.value || !currentDocument.value || !activeChatSession.value) return
    isChatExporting.value = true
    clearErrorMessage()
    try {
      const { blob, filename } = await exportCharacterStudioChatSession(
        bookId.value,
        currentDocument.value.id,
        activeChatSession.value.session_id,
      )
      downloadBlob(blob, filename)
    } catch (error) {
      throw createActionError(error, '导出聊天记录失败')
    } finally {
      isChatExporting.value = false
    }
  }

  async function importChatSession(file: File) {
    if (!bookId.value || !currentDocument.value) return
    isChatImporting.value = true
    clearErrorMessage()
    try {
      const response = await importCharacterStudioChatSession(bookId.value, currentDocument.value.id, file)
      if (!response.success) {
        throw new Error(response.error || '导入聊天记录失败')
      }
      applyChatStatePayload(response)
    } catch (error) {
      throw createActionError(error, '导入聊天记录失败')
    } finally {
      isChatImporting.value = false
    }
  }

  async function loadChatPromptPreview() {
    if (!bookId.value || !currentDocument.value || !activeChatSession.value) return
    isChatPromptLoading.value = true
    clearErrorMessage()
    chatPromptPreview.value = ''
    chatPromptPreviewError.value = ''
    try {
      const response = await getCharacterStudioChatPromptPreview(
        bookId.value,
        currentDocument.value.id,
        activeChatSession.value.session_id,
      )
      if (!response.success) {
        throw new Error(response.error || '加载提示词预览失败')
      }
      applyChatStatePayload(response)
    } catch (error) {
      chatPromptPreviewError.value = error instanceof Error ? error.message : '加载提示词预览失败'
      throw createActionError(error, '加载提示词预览失败')
    } finally {
      isChatPromptLoading.value = false
    }
  }

  async function importFile(file: File) {
    if (!bookId.value) return
    isImportingFile.value = true
    clearErrorMessage()
    try {
      const response = await importCharacterStudioFile(bookId.value, file)
      if (!response.success || !response.document) {
        throw new Error(response.error || '导入失败')
      }
      const document = response.document
      await loadWorkspace(bookId.value)
      await openDocument(document.id)
    } catch (error) {
      throw createActionError(error, '导入失败')
    } finally {
      isImportingFile.value = false
    }
  }

  async function importWorldbook(file: File) {
    if (!bookId.value || !currentDocument.value) return
    isImportingWorldbook.value = true
    clearErrorMessage()
    try {
      const response = await importWorldbookIntoCharacterStudioDocument(bookId.value, currentDocument.value.id, file)
      if (!response.success || !response.document) {
        throw new Error(response.error || '世界书导入失败')
      }
      const document = response.document
      await runWithoutAutosave(async () => {
        currentDocument.value = document
        markDocumentSynced(document)
      })
      await loadWorkspace(bookId.value)
    } catch (error) {
      throw createActionError(error, '世界书导入失败')
    } finally {
      isImportingWorldbook.value = false
    }
  }

  async function downloadCurrent(format: string) {
    if (!bookId.value || !currentDocument.value) return
    downloadingFormat.value = format
    clearErrorMessage()
    try {
      const { blob, filename } = format === 'worldbook'
        ? await downloadCharacterStudioWorldbook(bookId.value, currentDocument.value.id)
        : await downloadCharacterStudioExport(bookId.value, currentDocument.value.id, format)
      downloadBlob(blob, filename)
    } catch (error) {
      throw createActionError(error, '导出失败')
    } finally {
      downloadingFormat.value = null
    }
  }

  return {
    bookId,
    documents,
    candidates,
    hasTimeline,
    currentDocument,
    activeChatSession,
    archivedChatSessions,
    availableGreetings,
    chatPromptPreview,
    chatPromptPreviewError,
    diagnostics,
    agentMessages,
    agentHtmlPreview,
    pendingAgentPatch,
    canUndoPatch,
    editorPendingState,
    hasBusyAction,
    activeActionLabel,
    activeEditorTab,
    activeScriptTab,
    resourcePanelOpen,
    activeWorkspaceTab,
    isWorkspaceLoading,
    isDocumentLoading,
    isSaving,
    isChatLoading,
    isChatStreaming,
    isChatMutating,
    isChatSummarizing,
    isChatImporting,
    isChatExporting,
    isChatPromptLoading,
    isAgentBusy,
    isCreatingManual,
    isImportingFile,
    isImportingWorldbook,
    isDeleting,
    isValidating,
    openingDocumentId,
    creatingCandidateName,
    generatingSection,
    downloadingFormat,
    errorMessage,
    clearErrorMessage,
    selectedLibrarySearch,
    filteredDocuments,
    filteredCandidates,
    updateCurrentDocument,
    loadWorkspace,
    openDocument,
    loadChatState,
    createChatSession,
    switchChatSession,
    sendChatMessage,
    editChatMessage,
    deleteChatMessage,
    regenerateChatMessage,
    summarizeChatSession,
    exportChatSession,
    importChatSession,
    loadChatPromptPreview,
    createManualDocument,
    createDocumentFromCandidate,
    persistCurrentDocument,
    deleteCurrentDocument,
    generateSection,
    validateCurrentDocument,
    sendAgentMessage,
    applyPendingPatch,
    undoLastPatch,
    importFile,
    importWorldbook,
    downloadCurrent,
  }
})
