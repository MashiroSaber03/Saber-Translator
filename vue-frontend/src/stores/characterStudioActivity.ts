export interface CharacterStudioActivityState {
  isWorkspaceLoading: boolean
  isDocumentLoading: boolean
  isSaving: boolean
  isChatLoading: boolean
  isChatStreaming: boolean
  isChatMutating: boolean
  isChatSummarizing: boolean
  isChatImporting: boolean
  isChatExporting: boolean
  isChatPromptLoading: boolean
  isAgentBusy: boolean
  isCreatingManual: boolean
  isImportingFile: boolean
  isImportingWorldbook: boolean
  isDeleting: boolean
  isValidating: boolean
  openingDocumentId: string
  creatingCandidateName: string
  generatingSection: string | null
  downloadingFormat: string | null
}

const GENERATED_SECTION_LABELS: Record<string, string> = {
  full: '正在补全整张角色卡',
  identity: '正在补全角色设定',
  review: '正在审查当前角色',
  translate: '正在翻译整卡',
  greetings: '正在生成问候语',
  lorebook: '正在生成世界书',
  regex: '正在生成正则脚本',
  'state-tasks': '正在生成状态任务',
}

const DOWNLOAD_FORMAT_LABELS: Record<string, string> = {
  v3: '正在导出 V3 JSON',
  v2: '正在导出 V2 JSON',
  png: '正在导出 PNG',
  worldbook: '正在导出世界书',
}

export function hasCharacterStudioBusyAction(state: CharacterStudioActivityState): boolean {
  return [
    state.isWorkspaceLoading,
    state.isDocumentLoading,
    state.isSaving,
    state.isChatLoading,
    state.isChatStreaming,
    state.isChatMutating,
    state.isChatSummarizing,
    state.isChatImporting,
    state.isChatExporting,
    state.isChatPromptLoading,
    state.isAgentBusy,
    state.isCreatingManual,
    state.isImportingFile,
    state.isImportingWorldbook,
    state.isDeleting,
    state.isValidating,
    Boolean(state.openingDocumentId),
    Boolean(state.creatingCandidateName),
    Boolean(state.generatingSection),
    Boolean(state.downloadingFormat),
  ].some(Boolean)
}

export function getCharacterStudioActionLabel(state: CharacterStudioActivityState): string {
  if (state.isDocumentLoading) return '正在打开角色文档'
  if (state.openingDocumentId) return '正在切换角色文档'
  if (state.isWorkspaceLoading) return '正在加载角色工坊'
  if (state.isCreatingManual) return '正在新建角色文档'
  if (state.creatingCandidateName) return `正在从候选创建「${state.creatingCandidateName}」`
  if (state.isImportingFile) return '正在导入角色卡'
  if (state.isImportingWorldbook) return '正在导入世界书'
  if (state.isChatLoading) return '正在加载聊天会话'
  if (state.isChatStreaming) return '正在生成聊天回复'
  if (state.isChatMutating) return '正在处理聊天记录'
  if (state.isChatSummarizing) return '正在总结聊天'
  if (state.isChatImporting) return '正在导入聊天记录'
  if (state.isChatExporting) return '正在导出聊天记录'
  if (state.isChatPromptLoading) return '正在加载提示词预览'
  if (state.generatingSection) {
    return GENERATED_SECTION_LABELS[state.generatingSection] || '正在生成内容'
  }
  if (state.isValidating) return '正在执行角色诊断'
  if (state.isSaving) return '正在保存角色文档'
  if (state.downloadingFormat) {
    return DOWNLOAD_FORMAT_LABELS[state.downloadingFormat] || '正在导出文件'
  }
  if (state.isDeleting) return '正在删除角色文档'
  if (state.isAgentBusy) return '正在请求卡片助手'
  return ''
}
