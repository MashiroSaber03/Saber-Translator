import { defineStore } from 'pinia'
import { computed, nextTick, ref } from 'vue'
import {
  createCharacterStudioDocument,
  deleteCharacterStudioDocument,
  downloadCharacterStudioExport,
  downloadCharacterStudioWorldbook,
  generateCharacterStudioSection,
  getCharacterStudioDocument,
  getCharacterStudioIndex,
  importCharacterStudioFile,
  importWorldbookIntoCharacterStudioDocument,
  previewCharacterStudioChat,
  resetCharacterStudioPreview,
  runCharacterStudioAgent,
  saveCharacterStudioDocument,
  validateCharacterStudioDocument,
} from '@/api/characterStudio'
import type {
  CharacterStudioCandidate,
  CharacterStudioDocument,
  CharacterStudioEditorPendingState,
  CharacterStudioSummary,
  ExportDiagnostic,
  PreviewSessionState,
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
  const previewSession = ref<PreviewSessionState | null>(null)
  const diagnostics = ref<ExportDiagnostic | null>(null)
  const agentMessages = ref<Array<{ role: 'user' | 'assistant'; content: string }>>([])
  const agentHtmlPreview = ref('')
  const pendingAgentPatch = ref<Record<string, unknown> | null>(null)
  const activeEditorTab = ref<'overview' | 'character' | 'greetings' | 'lorebook' | 'scripts' | 'export'>('overview')
  const activeScriptTab = ref<'regex' | 'tasks'>('regex')
  const leftDrawerOpen = ref(false)
  const rightDrawerOpen = ref(false)
  const isWorkspaceLoading = ref(false)
  const isDocumentLoading = ref(false)
  const isSaving = ref(false)
  const isPreviewing = ref(false)
  const isAgentBusy = ref(false)
  const isCreatingManual = ref(false)
  const isImportingFile = ref(false)
  const isImportingWorldbook = ref(false)
  const isDeleting = ref(false)
  const isValidating = ref(false)
  const isResettingPreview = ref(false)
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
    isPreviewing.value,
    isAgentBusy.value,
    isCreatingManual.value,
    isImportingFile.value,
    isImportingWorldbook.value,
    isDeleting.value,
    isValidating.value,
    isResettingPreview.value,
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
    if (generatingSection.value) {
      return {
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
    if (isResettingPreview.value) return '正在重置预览会话'
    if (isPreviewing.value) return '正在生成预览回复'
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

  async function loadWorkspace(nextBookId: string) {
    if (!nextBookId) return
    isWorkspaceLoading.value = true
    errorMessage.value = ''
    bookId.value = nextBookId
    try {
      const response = await getCharacterStudioIndex(nextBookId)
      if (!response.success) {
        throw new Error(response.error || '加载角色工坊失败')
      }
      documents.value = response.documents || []
      candidates.value = response.candidates || []
      hasTimeline.value = response.has_timeline !== false
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : '加载角色工坊失败'
    } finally {
      isWorkspaceLoading.value = false
    }
  }

  async function openDocument(docId: string) {
    if (!bookId.value || !docId) return
    isDocumentLoading.value = true
    openingDocumentId.value = docId
    clearErrorMessage()
    try {
      const response = await getCharacterStudioDocument(bookId.value, docId)
      if (!response.success || !response.document) {
        throw new Error(response.error || '加载角色文档失败')
      }
      await runWithoutAutosave(async () => {
        currentDocument.value = response.document
        markDocumentSynced(response.document)
        previewSession.value = {
          doc_id: docId,
          messages: response.preview_session?.messages || response.document.previewState.messages || [],
          variables: response.preview_session?.variables || response.document.previewState.variables || {},
          log: response.preview_session?.log || [],
        }
        diagnostics.value = null
        agentMessages.value = []
        pendingAgentPatch.value = null
        agentHtmlPreview.value = ''
        patchSnapshot.value = null
        activeEditorTab.value = 'overview'
        activeScriptTab.value = 'regex'
      })
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
      await loadWorkspace(bookId.value)
      await openDocument(response.document.id)
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
      await loadWorkspace(bookId.value)
      await openDocument(response.document.id)
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
      await runWithoutAutosave(async () => {
        currentDocument.value = response.document
        markDocumentSynced(response.document)
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
      previewSession.value = null
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
      await runWithoutAutosave(async () => {
        currentDocument.value = response.document
        markDocumentSynced(response.document)
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

  async function resetPreview() {
    if (!bookId.value || !currentDocument.value) return
    isResettingPreview.value = true
    clearErrorMessage()
    try {
      const response = await resetCharacterStudioPreview(bookId.value, currentDocument.value.id)
      if (!response.success) {
        throw new Error(response.error || '重置预览失败')
      }
      previewSession.value = {
        doc_id: response.doc_id,
        messages: response.messages || [],
        variables: response.variables || {},
        log: response.log || [],
      }
    } catch (error) {
      throw createActionError(error, '重置预览失败')
    } finally {
      isResettingPreview.value = false
    }
  }

  async function sendPreviewMessage(message: string) {
    if (!bookId.value || !currentDocument.value || !message.trim()) return
    isPreviewing.value = true
    clearErrorMessage()
    try {
      const response = await previewCharacterStudioChat(bookId.value, currentDocument.value.id, message)
      if (!response.success) {
        throw new Error(response.error || '预览聊天失败')
      }
      previewSession.value = {
        doc_id: response.doc_id,
        messages: response.messages || [],
        variables: response.variables || {},
        log: response.log || [],
      }
    } catch (error) {
      throw createActionError(error, '预览聊天失败')
    } finally {
      isPreviewing.value = false
    }
  }

  function extractPatch(content: string): Record<string, unknown> | null {
    const match = content.match(/```json:patch\s*([\s\S]*?)```/i)
    if (!match) return null
    try {
      return JSON.parse(match[1]!.trim()) as Record<string, unknown>
    } catch {
      return null
    }
  }

  function extractHtmlPreview(content: string): string {
    const match = content.match(/```html\s*([\s\S]*?)```/i)
    return match?.[1]?.trim() || ''
  }

  function setByPath(target: Record<string, unknown>, path: string, value: unknown) {
    const keys = path.split('.')
    let current: Record<string, unknown> = target
    for (let i = 0; i < keys.length - 1; i += 1) {
      const key = keys[i]!
      const next = current[key]
      if (!next || typeof next !== 'object') {
        current[key] = {}
      }
      current = current[key] as Record<string, unknown>
    }
    current[keys[keys.length - 1]!] = value
  }

  function cloneDocument(document: CharacterStudioDocument): CharacterStudioDocument {
    return JSON.parse(JSON.stringify(document)) as CharacterStudioDocument
  }

  function applyPendingPatch() {
    if (!currentDocument.value || !pendingAgentPatch.value) return
    patchSnapshot.value = cloneDocument(currentDocument.value)
    const nextDocument = cloneDocument(currentDocument.value)
    const patch = pendingAgentPatch.value
    const frozenSections = new Set(nextDocument.status.frozen_sections || [])
    const setPayload = patch.set
    if (setPayload && typeof setPayload === 'object' && !Array.isArray(setPayload)) {
      Object.entries(setPayload as Record<string, unknown>).forEach(([path, value]) => {
        const section = path.split('.')[0]
        if (
          (section === 'identity' && frozenSections.has('identity')) ||
          (section === 'coreMessages' && frozenSections.has('greetings')) ||
          (section === 'lorebook' && frozenSections.has('lorebook')) ||
          (section === 'regexScripts' && frozenSections.has('regex')) ||
          (section === 'stateTasks' && frozenSections.has('state-tasks'))
        ) {
          return
        }
        setByPath(nextDocument as unknown as Record<string, unknown>, path, value)
      })
    }
    if (typeof patch.greeting_add === 'string' && !frozenSections.has('greetings')) {
      nextDocument.coreMessages.alternate_greetings.push(patch.greeting_add)
    }
    if (patch.worldbook_add && typeof patch.worldbook_add === 'object' && !Array.isArray(patch.worldbook_add) && !frozenSections.has('lorebook')) {
      nextDocument.lorebook.entries.push({
        id: `entry_${Date.now()}`,
        comment: String((patch.worldbook_add as Record<string, unknown>).comment || '新条目'),
        keys: Array.isArray((patch.worldbook_add as Record<string, unknown>).keys) ? ((patch.worldbook_add as Record<string, unknown>).keys as string[]) : [],
        secondary_keys: [],
        content: String((patch.worldbook_add as Record<string, unknown>).content || ''),
        enabled: true,
        constant: false,
        selective: true,
        priority: 100,
        position: 'before_char',
        depth: 4,
        children: [],
      })
    }
    if (patch.regex_add && typeof patch.regex_add === 'object' && !Array.isArray(patch.regex_add) && !frozenSections.has('regex')) {
      nextDocument.regexScripts.push({
        id: `regex_${Date.now()}`,
        scriptName: String((patch.regex_add as Record<string, unknown>).scriptName || '新脚本'),
        findRegex: String((patch.regex_add as Record<string, unknown>).findRegex || ''),
        replaceString: String((patch.regex_add as Record<string, unknown>).replaceString || ''),
        placement: Array.isArray((patch.regex_add as Record<string, unknown>).placement)
          ? ((patch.regex_add as Record<string, unknown>).placement as number[])
          : [2],
        markdownOnly: Boolean((patch.regex_add as Record<string, unknown>).markdownOnly ?? false),
        promptOnly: Boolean((patch.regex_add as Record<string, unknown>).promptOnly ?? false),
        runOnEdit: Boolean((patch.regex_add as Record<string, unknown>).runOnEdit ?? true),
        disabled: Boolean((patch.regex_add as Record<string, unknown>).disabled ?? false),
      })
    }
    if (patch.task_add && typeof patch.task_add === 'object' && !Array.isArray(patch.task_add) && !frozenSections.has('state-tasks')) {
      nextDocument.stateTasks.push({
        id: `task_${Date.now()}`,
        name: String((patch.task_add as Record<string, unknown>).name || '新任务'),
        triggerTiming: String((patch.task_add as Record<string, unknown>).triggerTiming || 'initialization'),
        interval: Number((patch.task_add as Record<string, unknown>).interval || 0),
        commands: String((patch.task_add as Record<string, unknown>).commands || ''),
        disabled: Boolean((patch.task_add as Record<string, unknown>).disabled ?? false),
      })
    }
    currentDocument.value = nextDocument
    pendingAgentPatch.value = null
    scheduleAutosave()
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

  async function importFile(file: File) {
    if (!bookId.value) return
    isImportingFile.value = true
    clearErrorMessage()
    try {
      const response = await importCharacterStudioFile(bookId.value, file)
      if (!response.success || !response.document) {
        throw new Error(response.error || '导入失败')
      }
      await loadWorkspace(bookId.value)
      await openDocument(response.document.id)
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
      await runWithoutAutosave(async () => {
        currentDocument.value = response.document
        markDocumentSynced(response.document)
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
    previewSession,
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
    leftDrawerOpen,
    rightDrawerOpen,
    isWorkspaceLoading,
    isDocumentLoading,
    isSaving,
    isPreviewing,
    isAgentBusy,
    isCreatingManual,
    isImportingFile,
    isImportingWorldbook,
    isDeleting,
    isValidating,
    isResettingPreview,
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
    createManualDocument,
    createDocumentFromCandidate,
    persistCurrentDocument,
    deleteCurrentDocument,
    generateSection,
    validateCurrentDocument,
    resetPreview,
    sendPreviewMessage,
    sendAgentMessage,
    applyPendingPatch,
    undoLastPatch,
    importFile,
    importWorldbook,
    downloadCurrent,
  }
})
