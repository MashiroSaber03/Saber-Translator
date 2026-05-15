import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
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
  const previewCollapsed = ref(false)
  const leftDrawerOpen = ref(false)
  const rightDrawerOpen = ref(false)
  const isWorkspaceLoading = ref(false)
  const isDocumentLoading = ref(false)
  const isSaving = ref(false)
  const isPreviewing = ref(false)
  const isAgentBusy = ref(false)
  const errorMessage = ref('')
  const selectedLibrarySearch = ref('')
  const _suspendAutosave = ref(false)
  let autosaveTimer: ReturnType<typeof setTimeout> | null = null
  const patchSnapshot = ref<CharacterStudioDocument | null>(null)

  const canUndoPatch = computed(() => patchSnapshot.value !== null)

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
    errorMessage.value = ''
    try {
      const response = await getCharacterStudioDocument(bookId.value, docId)
      if (!response.success || !response.document) {
        throw new Error(response.error || '加载角色文档失败')
      }
      _suspendAutosave.value = true
      currentDocument.value = response.document
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
      _suspendAutosave.value = false
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : '加载角色文档失败'
    } finally {
      isDocumentLoading.value = false
    }
  }

  async function createManualDocument(title: string = '新角色') {
    if (!bookId.value) return
    const response = await createCharacterStudioDocument(bookId.value, { title })
    if (!response.success || !response.document) {
      throw new Error(response.error || '创建角色失败')
    }
    await loadWorkspace(bookId.value)
    await openDocument(response.document.id)
  }

  async function createDocumentFromCandidate(candidateName: string) {
    if (!bookId.value) return
    const response = await createCharacterStudioDocument(bookId.value, { candidate_name: candidateName })
    if (!response.success || !response.document) {
      throw new Error(response.error || '创建角色失败')
    }
    await loadWorkspace(bookId.value)
    await openDocument(response.document.id)
  }

  async function persistCurrentDocument() {
    if (!bookId.value || !currentDocument.value) return
    isSaving.value = true
    try {
      const response = await saveCharacterStudioDocument(bookId.value, currentDocument.value.id, currentDocument.value as unknown as Record<string, unknown>)
      if (!response.success || !response.document) {
        throw new Error(response.error || '保存失败')
      }
      _suspendAutosave.value = true
      currentDocument.value = response.document
      _suspendAutosave.value = false
      await loadWorkspace(bookId.value)
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

  async function deleteCurrentDocument() {
    if (!bookId.value || !currentDocument.value) return
    const docId = currentDocument.value.id
    const response = await deleteCharacterStudioDocument(bookId.value, docId)
    if (!response.success) {
      throw new Error(response.error || '删除失败')
    }
    currentDocument.value = null
    previewSession.value = null
    diagnostics.value = null
    agentMessages.value = []
    pendingAgentPatch.value = null
    agentHtmlPreview.value = ''
    patchSnapshot.value = null
    await loadWorkspace(bookId.value)
  }

  async function generateSection(section: string) {
    if (!bookId.value || !currentDocument.value) return
    const response = await generateCharacterStudioSection(bookId.value, currentDocument.value.id, section)
    if (!response.success || !response.document) {
      throw new Error(response.error || '生成失败')
    }
    currentDocument.value = response.document
    await loadWorkspace(bookId.value)
  }

  async function validateCurrentDocument() {
    if (!bookId.value || !currentDocument.value) return
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
  }

  async function resetPreview() {
    if (!bookId.value || !currentDocument.value) return
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
  }

  async function sendPreviewMessage(message: string) {
    if (!bookId.value || !currentDocument.value || !message.trim()) return
    isPreviewing.value = true
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
    } finally {
      isAgentBusy.value = false
    }
  }

  async function importFile(file: File) {
    if (!bookId.value) return
    const response = await importCharacterStudioFile(bookId.value, file)
    if (!response.success || !response.document) {
      throw new Error(response.error || '导入失败')
    }
    await loadWorkspace(bookId.value)
    await openDocument(response.document.id)
  }

  async function importWorldbook(file: File) {
    if (!bookId.value || !currentDocument.value) return
    const response = await importWorldbookIntoCharacterStudioDocument(bookId.value, currentDocument.value.id, file)
    if (!response.success || !response.document) {
      throw new Error(response.error || '世界书导入失败')
    }
    currentDocument.value = response.document
    await loadWorkspace(bookId.value)
  }

  async function downloadCurrent(format: string) {
    if (!bookId.value || !currentDocument.value) return
    const { blob, filename } = format === 'worldbook'
      ? await downloadCharacterStudioWorldbook(bookId.value, currentDocument.value.id)
      : await downloadCharacterStudioExport(bookId.value, currentDocument.value.id, format)
    downloadBlob(blob, filename)
  }

  watch(currentDocument, () => {
    scheduleAutosave()
  }, { deep: true })

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
    activeEditorTab,
    activeScriptTab,
    previewCollapsed,
    leftDrawerOpen,
    rightDrawerOpen,
    isWorkspaceLoading,
    isDocumentLoading,
    isSaving,
    isPreviewing,
    isAgentBusy,
    errorMessage,
    selectedLibrarySearch,
    filteredDocuments,
    filteredCandidates,
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
