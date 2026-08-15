import { defineStore } from 'pinia'
import { computed, nextTick, ref } from 'vue'
import {
  createCharacterStudioChatSession,
  createCharacterStudioDocument,
  deleteCharacterStudioChatMessage,
  deleteCharacterStudioChatSession,
  deleteCharacterStudioDocument,
  editCharacterStudioChatMessage,
  generateCharacterStudioSection,
  getCharacterStudioChatPromptPreview,
  getCharacterStudioChatState,
  getCharacterStudioDocument,
  getCharacterStudioIndex,
  importCharacterStudioFile,
  importCharacterStudioChatSession,
  importWorldbookIntoCharacterStudioDocument,
  runCharacterStudioAgent,
  saveCharacterStudioDocument,
  summarizeCharacterStudioChatSession,
  switchCharacterStudioChatSession,
  validateCharacterStudioDocument,
} from '@/api/characterStudio'
import {
  downloadStudioChatTranscript,
  downloadStudioDocumentExport,
} from '@/stores/characterStudioExports'
import {
  getCharacterStudioActionLabel,
  hasCharacterStudioBusyAction,
} from '@/stores/characterStudioActivity'
import { parseCharacterStudioAgentOutput } from '@/stores/characterStudioAgentOutput'
import { applyCharacterStudioAgentPatch } from '@/stores/characterStudioPatch'
import { useCharacterStudioChat } from './characterStudio/useCharacterStudioChat'
import type {
  CharacterStudioAgentPatchV2,
  CharacterStudioChatSession,
  CharacterStudioChatSessionSummary,
  CharacterStudioCandidate,
  CharacterStudioDocument,
  CharacterStudioEditorPendingState,
  CharacterStudioGenerationSection,
  CharacterStudioGreetingOption,
  CharacterStudioSummary,
  ExportDiagnostic,
} from '@/types/characterStudio'
import { deepClone } from '@/utils/deepClone'
import { characterStudioDocumentContent } from '@/utils/characterStudioDocumentContent'

export const useCharacterStudioStore = defineStore('character-studio', () => {
  const bookId = ref('')
  const documents = ref<CharacterStudioSummary[]>([])
  const candidates = ref<CharacterStudioCandidate[]>([])
  const hasTimeline = ref(false)
  const currentDocument = ref<CharacterStudioDocument | null>(null)
  const chatIndexRevision = ref<number | null>(null)
  const activeChatSession = ref<CharacterStudioChatSession | null>(null)
  const archivedChatSessions = ref<CharacterStudioChatSessionSummary[]>([])
  const availableChatGreetings = ref<CharacterStudioGreetingOption[]>([])
  const chatPromptPreview = ref('')
  const chatPromptPreviewError = ref('')
  const diagnostics = ref<ExportDiagnostic | null>(null)
  const agentMessages = ref<Array<{ role: 'user' | 'assistant'; content: string }>>([])
  const agentHtmlPreview = ref('')
  const pendingAgentPatch = ref<CharacterStudioAgentPatchV2 | null>(null)
  const activeEditorTab = ref<
    'overview' | 'character' | 'greetings' | 'lorebook' | 'scripts' | 'export'
  >('overview')
  const activeScriptTab = ref<'regex' | 'tasks'>('regex')
  const resourcePanelOpen = ref(false)
  const activeWorkspaceTab = ref<'chat' | 'assistant' | 'runtime'>('chat')
  const isWorkspaceLoading = ref(false)
  const isDocumentLoading = ref(false)
  const isSaving = ref(false)
  const isChatLoading = ref(false)
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
  const creatingCandidateId = ref('')
  const creatingCandidateName = ref('')
  const generatingSection = ref<string | null>(null)
  const downloadingFormat = ref<string | null>(null)
  const errorMessage = ref('')
  const selectedLibrarySearch = ref('')
  const _suspendAutosave = ref(false)
  const lastSyncedFingerprint = ref('')
  const lastSyncedChatFingerprint = ref('')
  const pendingChatRehydrate = ref(false)
  let autosaveTimer: ReturnType<typeof setTimeout> | null = null
  let activeDocumentSave: Promise<void> | null = null
  const patchSnapshot = ref<CharacterStudioDocument | null>(null)
  let workspaceLoadRequestId = 0
  let documentLoadRequestId = 0
  let chatStateLoadRequestId = 0
  let chatPromptPreviewRequestId = 0
  const chat = useCharacterStudioChat({
    bookId,
    currentDocument,
    activeChatSession,
    activeWorkspaceTab,
    errorMessage,
    applySession: session => applyChatStatePayload({ active_session: session }),
    flushPendingRehydrate: flushPendingChatRehydrate,
    reloadChatState: loadChatState,
  })
  const {
    activeChatOperationId,
    acceptedChatSubmissionCount,
    isChatStreaming,
    abortActiveChatOperation,
    abortActiveChatStream,
    sendChatMessage: sendChatMessageWithoutDocumentFlush,
    regenerateChatMessage: regenerateChatMessageWithoutDocumentFlush,
  } = chat

  const canUndoPatch = computed(() => patchSnapshot.value !== null)
  const editorPendingState = computed<CharacterStudioEditorPendingState>(() => ({
    generatingSection: generatingSection.value,
    validating: isValidating.value,
    importingWorldbook: isImportingWorldbook.value,
    deleting: isDeleting.value,
    saving: isSaving.value,
    downloadingFormat: downloadingFormat.value,
  }))

  const activityState = computed(() => ({
    isWorkspaceLoading: isWorkspaceLoading.value,
    isDocumentLoading: isDocumentLoading.value,
    isSaving: isSaving.value,
    isChatLoading: isChatLoading.value,
    isChatStreaming: isChatStreaming.value,
    isChatMutating: isChatMutating.value,
    isChatSummarizing: isChatSummarizing.value,
    isChatImporting: isChatImporting.value,
    isChatExporting: isChatExporting.value,
    isChatPromptLoading: isChatPromptLoading.value,
    isAgentBusy: isAgentBusy.value,
    isCreatingManual: isCreatingManual.value,
    isImportingFile: isImportingFile.value,
    isImportingWorldbook: isImportingWorldbook.value,
    isDeleting: isDeleting.value,
    isValidating: isValidating.value,
    openingDocumentId: openingDocumentId.value,
    creatingCandidateName: creatingCandidateName.value,
    generatingSection: generatingSection.value,
    downloadingFormat: downloadingFormat.value,
  }))
  const hasBusyAction = computed(() => hasCharacterStudioBusyAction(activityState.value))
  const activeActionLabel = computed(() => getCharacterStudioActionLabel(activityState.value))

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
    chatStateLoadRequestId += 1
    chatPromptPreviewRequestId += 1
    activeChatSession.value = null
    chatIndexRevision.value = null
    archivedChatSessions.value = []
    availableChatGreetings.value = []
    chatPromptPreview.value = ''
    chatPromptPreviewError.value = ''
    pendingChatRehydrate.value = false
  }

  function isActiveWorkspaceRequest(requestId: number, requestedBookId: string) {
    return requestId === workspaceLoadRequestId && bookId.value === requestedBookId
  }

  function isActiveDocumentRequest(requestId: number, requestedBookId: string) {
    return requestId === documentLoadRequestId && bookId.value === requestedBookId
  }

  function isActiveChatStateRequest(
    requestId: number,
    requestedBookId: string,
    requestedDocId: string
  ) {
    return (
      requestId === chatStateLoadRequestId &&
      bookId.value === requestedBookId &&
      currentDocument.value?.id === requestedDocId
    )
  }

  function isActiveChatPromptPreviewRequest(
    requestId: number,
    requestedBookId: string,
    requestedDocId: string,
    requestedSessionId: string
  ) {
    return (
      requestId === chatPromptPreviewRequestId &&
      bookId.value === requestedBookId &&
      currentDocument.value?.id === requestedDocId &&
      activeChatSession.value?.session_id === requestedSessionId
    )
  }

  function resetWorkspaceState() {
    currentDocument.value = null
    markDocumentSynced(null)
    abortActiveChatStream()
    resetChatState()
    diagnostics.value = null
    agentMessages.value = []
    agentHtmlPreview.value = ''
    pendingAgentPatch.value = null
    patchSnapshot.value = null
    activeEditorTab.value = 'overview'
    activeScriptTab.value = 'regex'
    activeWorkspaceTab.value = 'chat'
    if (autosaveTimer) {
      clearTimeout(autosaveTimer)
      autosaveTimer = null
    }
  }

  function invalidateDocumentDerivedCaches() {
    diagnostics.value = null
    if (currentDocument.value) {
      currentDocument.value.status.last_diagnostics = null
      currentDocument.value.status.last_validated_at = null
    }
    chatPromptPreview.value = ''
    chatPromptPreviewError.value = ''
  }

  function shouldDelayChatRehydrate() {
    return (
      isChatStreaming.value ||
      isChatMutating.value ||
      isChatSummarizing.value ||
      isChatImporting.value ||
      isChatExporting.value
    )
  }

  async function rehydrateChatAfterDocumentMutation(docId: string) {
    if (!bookId.value || !docId) return
    if (shouldDelayChatRehydrate()) {
      pendingChatRehydrate.value = true
      return
    }
    pendingChatRehydrate.value = false
    try {
      await loadChatState(docId)
    } catch {
      // 聊天态补刷失败不应回滚已成功的文档变更，错误由 store 状态承载。
    }
  }

  async function flushPendingChatRehydrate() {
    if (!pendingChatRehydrate.value) return
    const docId = currentDocument.value?.id || ''
    if (!docId || shouldDelayChatRehydrate()) return
    await rehydrateChatAfterDocumentMutation(docId)
  }

  async function loadWorkspace(nextBookId: string) {
    if (!nextBookId) return
    const requestId = ++workspaceLoadRequestId
    const isBookChanged = !!bookId.value && bookId.value !== nextBookId
    isWorkspaceLoading.value = true
    errorMessage.value = ''
    if (isBookChanged) {
      resetWorkspaceState()
      documents.value = []
      candidates.value = []
      hasTimeline.value = false
      selectedLibrarySearch.value = ''
    }
    bookId.value = nextBookId
    try {
      const index = await getCharacterStudioIndex(nextBookId)
      if (!isActiveWorkspaceRequest(requestId, nextBookId)) return
      documents.value = index.documents
      candidates.value = index.candidates
      hasTimeline.value = index.has_timeline
      if (
        currentDocument.value &&
        !documents.value.some(item => item.id === currentDocument.value?.id)
      ) {
        resetWorkspaceState()
      }
    } catch (error) {
      if (!isActiveWorkspaceRequest(requestId, nextBookId)) return
      errorMessage.value = error instanceof Error ? error.message : '加载角色工坊失败'
    } finally {
      if (requestId === workspaceLoadRequestId) {
        isWorkspaceLoading.value = false
      }
    }
  }

  async function openDocument(docId: string) {
    if (!bookId.value || !docId) return
    const requestId = ++documentLoadRequestId
    const requestedBookId = bookId.value
    isDocumentLoading.value = true
    openingDocumentId.value = docId
    clearErrorMessage()
    try {
      const document = await getCharacterStudioDocument(docId)
      if (!isActiveDocumentRequest(requestId, requestedBookId)) return
      if (document.bookId !== requestedBookId) {
        throw new Error('角色文档不属于当前书籍')
      }
      await runWithoutAutosave(async () => {
        abortActiveChatStream()
        currentDocument.value = document
        markDocumentSynced(document)
        resetChatState()
        diagnostics.value = document.status.last_diagnostics
          ? deepClone(document.status.last_diagnostics)
          : null
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
      if (!isActiveDocumentRequest(requestId, requestedBookId)) return
      throw createActionError(error, '加载角色文档失败')
    } finally {
      if (requestId === documentLoadRequestId) {
        isDocumentLoading.value = false
        openingDocumentId.value = ''
      }
    }
  }

  async function createManualDocument(title: string = '新角色') {
    if (!bookId.value) return
    isCreatingManual.value = true
    clearErrorMessage()
    try {
      const document = await createCharacterStudioDocument(bookId.value, { title })
      await loadWorkspace(bookId.value)
      await openDocument(document.id)
    } catch (error) {
      throw createActionError(error, '创建角色失败')
    } finally {
      isCreatingManual.value = false
    }
  }

  async function createDocumentFromCandidate(candidateId: string) {
    if (!bookId.value) return
    const candidate = candidates.value.find(item => item.id === candidateId)
    if (!candidate) {
      throw createActionError(new Error('候选角色不存在'), '创建角色失败')
    }
    creatingCandidateId.value = candidateId
    creatingCandidateName.value = candidate.name
    clearErrorMessage()
    try {
      const document = await createCharacterStudioDocument(bookId.value, {
        candidate_id: candidateId,
      })
      await loadWorkspace(bookId.value)
      await openDocument(document.id)
    } catch (error) {
      throw createActionError(error, '创建角色失败')
    } finally {
      creatingCandidateId.value = ''
      creatingCandidateName.value = ''
    }
  }

  function persistCurrentDocument(): Promise<void> {
    if (!bookId.value || !currentDocument.value) return Promise.resolve()
    if (autosaveTimer) {
      clearTimeout(autosaveTimer)
      autosaveTimer = null
    }
    if (activeDocumentSave) return activeDocumentSave
    if (buildAutosaveFingerprint(currentDocument.value) === lastSyncedFingerprint.value)
      return Promise.resolve()

    activeDocumentSave = savePendingDocumentEdits().finally(() => {
      isSaving.value = false
      activeDocumentSave = null
    })
    return activeDocumentSave
  }

  async function savePendingDocumentEdits() {
    isSaving.value = true
    clearErrorMessage()
    let chatStateChanged = false
    try {
      while (bookId.value && currentDocument.value) {
        const requestedBookId = bookId.value
        const snapshot = deepClone(currentDocument.value)
        const snapshotFingerprint = buildAutosaveFingerprint(snapshot)
        if (autosaveTimer) {
          clearTimeout(autosaveTimer)
          autosaveTimer = null
        }
        const document = await saveCharacterStudioDocument(snapshot.id, snapshot)

        if (bookId.value !== requestedBookId || currentDocument.value?.id !== snapshot.id) return

        chatStateChanged ||=
          buildChatFingerprint(snapshot) !== lastSyncedChatFingerprint.value ||
          buildChatFingerprint(document) !== lastSyncedChatFingerprint.value
        const editedWhileSaving =
          buildAutosaveFingerprint(currentDocument.value) !== snapshotFingerprint
        await runWithoutAutosave(async () => {
          if (editedWhileSaving && currentDocument.value) {
            currentDocument.value = rebaseUnsavedDocument(currentDocument.value, document)
          } else {
            currentDocument.value = document
          }
          markDocumentSynced(document)
          updateDocumentSummary(currentDocument.value)
        })

        if (editedWhileSaving) continue
        if (chatStateChanged) {
          await rehydrateChatAfterDocumentMutation(document.id)
          chatStateChanged = false
        }
        if (
          currentDocument.value &&
          buildAutosaveFingerprint(currentDocument.value) !== lastSyncedFingerprint.value
        )
          continue
        return
      }
    } catch (error) {
      throw createActionError(error, '保存失败')
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
    return JSON.stringify(characterStudioDocumentContent(document))
  }

  function buildChatFingerprint(document: CharacterStudioDocument | null) {
    if (!document) return ''
    return JSON.stringify({
      firstMessage: document.coreMessages.first_message,
      alternateGreetings: document.coreMessages.alternate_greetings,
    })
  }

  function markDocumentSynced(document: CharacterStudioDocument | null) {
    lastSyncedFingerprint.value = buildAutosaveFingerprint(document)
    lastSyncedChatFingerprint.value = buildChatFingerprint(document)
  }

  function updateDocumentSummary(document: CharacterStudioDocument | null) {
    if (!document) return
    const index = documents.value.findIndex(item => item.id === document.id)
    const summary: CharacterStudioSummary = {
      id: document.id,
      title: document.meta.title,
      origin: document.origin.type,
      source_character: document.origin.source_character ?? null,
      updated_at: document.updatedAt,
      tags: [...document.meta.tags],
      is_favorite: document.status.is_favorite,
      has_avatar: Boolean(document.avatarUrl),
    }
    if (index < 0) {
      documents.value = [summary, ...documents.value]
      return
    }
    const nextDocuments = [...documents.value]
    nextDocuments[index] = summary
    documents.value = nextDocuments
  }

  function rebaseUnsavedDocument(
    localDocument: CharacterStudioDocument,
    savedDocument: CharacterStudioDocument
  ): CharacterStudioDocument {
    const rebased = deepClone(localDocument)
    rebased.revision = savedDocument.revision
    rebased.avatarUrl = savedDocument.avatarUrl
    rebased.createdAt = savedDocument.createdAt
    rebased.updatedAt = savedDocument.updatedAt
    return rebased
  }

  function clearErrorMessage() {
    errorMessage.value = ''
  }

  function requireChatIndexRevision(): number {
    if (chatIndexRevision.value === null) {
      throw new Error('聊天状态版本缺失，请重新加载')
    }
    return chatIndexRevision.value
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
    invalidateDocumentDerivedCaches()
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
      await deleteCharacterStudioDocument(docId)
      resetWorkspaceState()
      await loadWorkspace(bookId.value)
    } catch (error) {
      throw createActionError(error, '删除失败')
    } finally {
      isDeleting.value = false
    }
  }

  async function generateSection(section: CharacterStudioGenerationSection) {
    if (!bookId.value || !currentDocument.value) return
    generatingSection.value = section
    clearErrorMessage()
    try {
      await persistCurrentDocument()
      if (!currentDocument.value) return
      const document = await generateCharacterStudioSection(
        currentDocument.value.id,
        currentDocument.value.revision,
        section
      )
      await runWithoutAutosave(async () => {
        currentDocument.value = document
        markDocumentSynced(document)
        invalidateDocumentDerivedCaches()
      })
      await loadWorkspace(bookId.value)
      await rehydrateChatAfterDocumentMutation(document.id)
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
      await persistCurrentDocument()
      if (!currentDocument.value) return
      const response = await validateCharacterStudioDocument(
        currentDocument.value.id,
        currentDocument.value.revision
      )
      diagnostics.value = {
        valid: response.valid,
        errors: response.errors,
        warnings: response.warnings,
        checks: response.checks,
      }
      const refreshedDocument = response.document
      await runWithoutAutosave(async () => {
        currentDocument.value = refreshedDocument
        markDocumentSynced(refreshedDocument)
      })
    } catch (error) {
      throw createActionError(error, '诊断失败')
    } finally {
      isValidating.value = false
    }
  }

  function applyPendingPatch() {
    if (!currentDocument.value || !pendingAgentPatch.value) return
    clearErrorMessage()
    try {
      const nextDocument = applyCharacterStudioAgentPatch(
        currentDocument.value,
        pendingAgentPatch.value
      )
      if (
        buildAutosaveFingerprint(nextDocument) === buildAutosaveFingerprint(currentDocument.value)
      ) {
        pendingAgentPatch.value = null
        return
      }
      patchSnapshot.value = deepClone(currentDocument.value)
      currentDocument.value = nextDocument
      pendingAgentPatch.value = null
      invalidateDocumentDerivedCaches()
      scheduleAutosave()
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : '应用 patch 失败'
    }
  }

  function undoLastPatch() {
    if (!patchSnapshot.value) return
    const snapshot = patchSnapshot.value
    const latestDocument = currentDocument.value
    currentDocument.value =
      latestDocument?.id === snapshot.id
        ? rebaseUnsavedDocument(snapshot, latestDocument)
        : snapshot
    patchSnapshot.value = null
    pendingAgentPatch.value = null
    invalidateDocumentDerivedCaches()
    scheduleAutosave()
  }

  async function sendAgentMessage(message: string) {
    if (
      !bookId.value ||
      !currentDocument.value ||
      !message.trim() ||
      isAgentBusy.value
    ) return
    isAgentBusy.value = true
    clearErrorMessage()
    let pendingMessageIndex = -1
    try {
      await persistCurrentDocument()
      if (!currentDocument.value) return
      pendingMessageIndex = agentMessages.value.length
      agentMessages.value.push({ role: 'user', content: message })
      const content = await runCharacterStudioAgent(currentDocument.value.id, message)
      const output = parseCharacterStudioAgentOutput(content)
      agentMessages.value.push({ role: 'assistant', content })
      pendingAgentPatch.value = output.patch
      agentHtmlPreview.value = output.htmlPreview
    } catch (error) {
      const pendingMessage = agentMessages.value[pendingMessageIndex]
      if (
        pendingMessageIndex >= 0 &&
        agentMessages.value.length === pendingMessageIndex + 1 &&
        pendingMessage?.role === 'user' &&
        pendingMessage.content === message
      ) {
        agentMessages.value.splice(pendingMessageIndex, 1)
      }
      throw createActionError(error, 'Agent 调用失败')
    } finally {
      isAgentBusy.value = false
    }
  }

  function applyChatStatePayload(payload: {
    index_revision?: number
    active_session?: CharacterStudioChatSession | null
    archived_sessions?: CharacterStudioChatSessionSummary[]
    available_greetings?: CharacterStudioGreetingOption[]
    prompt_preview?: string
  }) {
    if ('active_session' in payload) {
      const nextSession = payload.active_session ?? null
      if (activeChatSession.value?.session_id !== nextSession?.session_id) {
        chatPromptPreview.value = ''
      }
      activeChatSession.value = nextSession
      if (nextSession) chatIndexRevision.value = nextSession.index_revision
    }
    if (typeof payload.index_revision === 'number') {
      chatIndexRevision.value = payload.index_revision
    }
    if ('archived_sessions' in payload && payload.archived_sessions) {
      archivedChatSessions.value = payload.archived_sessions
    }
    if ('available_greetings' in payload && payload.available_greetings) {
      availableChatGreetings.value = payload.available_greetings
    }
    if (typeof payload.prompt_preview === 'string') {
      chatPromptPreview.value = payload.prompt_preview
    }
  }

  async function loadChatState(docId: string) {
    if (!bookId.value || !docId) return
    const requestId = ++chatStateLoadRequestId
    const requestedBookId = bookId.value
    isChatLoading.value = true
    clearErrorMessage()
    try {
      const state = await getCharacterStudioChatState(docId)
      if (!isActiveChatStateRequest(requestId, requestedBookId, docId)) return
      applyChatStatePayload(state)
    } catch (error) {
      if (!isActiveChatStateRequest(requestId, requestedBookId, docId)) return
      throw createActionError(error, '加载聊天状态失败')
    } finally {
      if (requestId === chatStateLoadRequestId) {
        isChatLoading.value = false
      }
    }
  }

  async function createChatSession(greetingId?: string) {
    if (!bookId.value || !currentDocument.value || isChatStreaming.value) return
    isChatMutating.value = true
    clearErrorMessage()
    try {
      await persistCurrentDocument()
      if (!currentDocument.value) return
      const state = await createCharacterStudioChatSession(
        currentDocument.value.id,
        requireChatIndexRevision(),
        greetingId
      )
      applyChatStatePayload(state)
    } catch (error) {
      throw createActionError(error, '创建聊天会话失败')
    } finally {
      isChatMutating.value = false
      void flushPendingChatRehydrate()
    }
  }

  async function switchChatSession(sessionId: string) {
    if (!bookId.value || !currentDocument.value || !sessionId || isChatStreaming.value) return
    isChatMutating.value = true
    clearErrorMessage()
    try {
      const state = await switchCharacterStudioChatSession(
        currentDocument.value.id,
        sessionId,
        requireChatIndexRevision()
      )
      applyChatStatePayload(state)
    } catch (error) {
      throw createActionError(error, '切换聊天会话失败')
    } finally {
      isChatMutating.value = false
      void flushPendingChatRehydrate()
    }
  }

  async function deleteArchivedChatSession(sessionId: string, revision: number) {
    if (!bookId.value || !currentDocument.value || !sessionId || isChatStreaming.value) return
    isChatMutating.value = true
    clearErrorMessage()
    try {
      const state = await deleteCharacterStudioChatSession(
        currentDocument.value.id,
        sessionId,
        revision
      )
      applyChatStatePayload(state)
    } catch (error) {
      throw createActionError(error, '删除归档会话失败')
    } finally {
      isChatMutating.value = false
      void flushPendingChatRehydrate()
    }
  }

  async function editChatMessage(messageId: string, content: string) {
    if (
      !bookId.value ||
      !currentDocument.value ||
      !activeChatSession.value ||
      isChatStreaming.value
    )
      return
    isChatMutating.value = true
    clearErrorMessage()
    try {
      const session = await editCharacterStudioChatMessage(
        activeChatSession.value.session_id,
        activeChatSession.value.revision,
        messageId,
        content
      )
      applyChatStatePayload({ active_session: session })
    } catch (error) {
      throw createActionError(error, '编辑消息失败')
    } finally {
      isChatMutating.value = false
      void flushPendingChatRehydrate()
    }
  }

  async function deleteChatMessage(messageId: string) {
    if (
      !bookId.value ||
      !currentDocument.value ||
      !activeChatSession.value ||
      isChatStreaming.value
    )
      return
    isChatMutating.value = true
    clearErrorMessage()
    try {
      const session = await deleteCharacterStudioChatMessage(
        activeChatSession.value.session_id,
        activeChatSession.value.revision,
        messageId
      )
      applyChatStatePayload({ active_session: session })
    } catch (error) {
      throw createActionError(error, '删除消息失败')
    } finally {
      isChatMutating.value = false
      void flushPendingChatRehydrate()
    }
  }

  async function summarizeChatSession() {
    if (
      !bookId.value ||
      !currentDocument.value ||
      !activeChatSession.value ||
      isChatStreaming.value
    )
      return
    isChatSummarizing.value = true
    clearErrorMessage()
    try {
      const session = await summarizeCharacterStudioChatSession(
        activeChatSession.value.session_id,
        activeChatSession.value.revision
      )
      applyChatStatePayload({ active_session: session })
    } catch (error) {
      throw createActionError(error, '总结聊天失败')
    } finally {
      isChatSummarizing.value = false
      void flushPendingChatRehydrate()
    }
  }

  async function exportChatSession() {
    if (!bookId.value || !currentDocument.value || !activeChatSession.value) return
    isChatExporting.value = true
    clearErrorMessage()
    try {
      await downloadStudioChatTranscript(activeChatSession.value.session_id)
    } catch (error) {
      throw createActionError(error, '导出聊天记录失败')
    } finally {
      isChatExporting.value = false
      void flushPendingChatRehydrate()
    }
  }

  async function importChatSession(file: File) {
    if (!bookId.value || !currentDocument.value || isChatStreaming.value) return
    isChatImporting.value = true
    clearErrorMessage()
    try {
      await persistCurrentDocument()
      if (!currentDocument.value) return
      const state = await importCharacterStudioChatSession(
        currentDocument.value.id,
        requireChatIndexRevision(),
        file
      )
      applyChatStatePayload(state)
    } catch (error) {
      throw createActionError(error, '导入聊天记录失败')
    } finally {
      isChatImporting.value = false
      void flushPendingChatRehydrate()
    }
  }

  async function loadChatPromptPreview() {
    if (!bookId.value || !currentDocument.value || !activeChatSession.value) return
    const requestId = ++chatPromptPreviewRequestId
    const requestedBookId = bookId.value
    const requestedDocId = currentDocument.value.id
    const requestedSessionId = activeChatSession.value.session_id
    isChatPromptLoading.value = true
    clearErrorMessage()
    chatPromptPreview.value = ''
    chatPromptPreviewError.value = ''
    try {
      await persistCurrentDocument()
      if (!activeChatSession.value) return
      const promptPreview = await getCharacterStudioChatPromptPreview(requestedSessionId)
      if (
        !isActiveChatPromptPreviewRequest(
          requestId,
          requestedBookId,
          requestedDocId,
          requestedSessionId
        )
      )
        return
      chatPromptPreview.value = promptPreview
    } catch (error) {
      if (
        !isActiveChatPromptPreviewRequest(
          requestId,
          requestedBookId,
          requestedDocId,
          requestedSessionId
        )
      )
        return
      chatPromptPreviewError.value = error instanceof Error ? error.message : '加载提示词预览失败'
      throw createActionError(error, '加载提示词预览失败')
    } finally {
      if (requestId === chatPromptPreviewRequestId) {
        isChatPromptLoading.value = false
      }
    }
  }

  async function importFile(file: File) {
    if (!bookId.value) return
    isImportingFile.value = true
    clearErrorMessage()
    try {
      const document = await importCharacterStudioFile(bookId.value, file)
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
      await persistCurrentDocument()
      if (!currentDocument.value) return
      const document = await importWorldbookIntoCharacterStudioDocument(
        currentDocument.value.id,
        currentDocument.value.revision,
        file
      )
      await runWithoutAutosave(async () => {
        currentDocument.value = document
        markDocumentSynced(document)
        invalidateDocumentDerivedCaches()
      })
      await loadWorkspace(bookId.value)
      await rehydrateChatAfterDocumentMutation(document.id)
    } catch (error) {
      throw createActionError(error, '世界书导入失败')
    } finally {
      isImportingWorldbook.value = false
      void flushPendingChatRehydrate()
    }
  }

  async function downloadCurrent(format: string) {
    if (!bookId.value || !currentDocument.value) return
    downloadingFormat.value = format
    clearErrorMessage()
    try {
      await persistCurrentDocument()
      if (!currentDocument.value) return
      await downloadStudioDocumentExport(currentDocument.value.id, format)
    } catch (error) {
      throw createActionError(error, '导出失败')
    } finally {
      downloadingFormat.value = null
    }
  }

  async function sendChatMessage(content: string, attachments: File[] = []): Promise<void> {
    await persistCurrentDocument()
    await sendChatMessageWithoutDocumentFlush(content, attachments)
  }

  async function regenerateChatMessage(messageId: string): Promise<void> {
    await persistCurrentDocument()
    await regenerateChatMessageWithoutDocumentFlush(messageId)
  }

  return {
    bookId,
    documents,
    candidates,
    hasTimeline,
    currentDocument,
    chatIndexRevision,
    activeChatSession,
    archivedChatSessions,
    availableChatGreetings,
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
    activeChatOperationId,
    acceptedChatSubmissionCount,
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
    creatingCandidateId,
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
    createChatSession,
    switchChatSession,
    deleteArchivedChatSession,
    abortActiveChatOperation,
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
