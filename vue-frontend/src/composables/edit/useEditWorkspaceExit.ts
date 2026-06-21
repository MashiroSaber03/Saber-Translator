import { computed, ref, type Ref } from 'vue'
import { useSettingsStore } from '@/stores/settings'

export type ExitDialogState = 'closed' | 'confirm' | 'saving' | 'error'

interface SessionLoadingProgress {
  current: number
  total: number
  message: string
}

interface UseEditWorkspaceExitOptions {
  isBookshelfMode: Ref<boolean>
  currentBookId: Ref<string | null>
  currentChapterId: Ref<string | null>
  isSessionSaving: Ref<boolean>
  sessionLoadingProgress: Ref<SessionLoadingProgress>
  sessionSaveError: Ref<string | null>
  saveBubbleStatesToImage: () => void
  saveChapterSession: (bookId: string, chapterId: string) => Promise<boolean>
  emitExit: () => void
}

export function useEditWorkspaceExit(options: UseEditWorkspaceExitOptions) {
  const settingsStore = useSettingsStore()

  const exitDialogState = ref<ExitDialogState>('closed')
  const exitDialogError = ref('')

  const shouldPromptSaveOnExit = computed(() =>
    options.isBookshelfMode.value && settingsStore.settings.autoSaveInBookshelfMode
  )
  const exitSaveCurrent = computed(() => options.sessionLoadingProgress.value.current)
  const exitSaveTotal = computed(() => options.sessionLoadingProgress.value.total)
  const exitSaveHasProgress = computed(() => exitSaveTotal.value > 0)
  const exitSaveProgressPercent = computed(() => {
    if (!exitSaveHasProgress.value) return 0
    return Math.round((exitSaveCurrent.value / exitSaveTotal.value) * 100)
  })
  const exitSaveMessage = computed(() => {
    const message = options.sessionLoadingProgress.value.message?.trim()
    if (message) {
      return message
    }
    return options.isSessionSaving.value ? '正在保存章节进度，完成后将自动退出编辑模式...' : '正在准备保存...'
  })

  function closeExitDialog(): void {
    exitDialogState.value = 'closed'
    exitDialogError.value = ''
  }

  function openExitDialog(): void {
    exitDialogState.value = 'confirm'
    exitDialogError.value = ''
  }

  function exitEditMode(): void {
    if (exitDialogState.value === 'saving') {
      return
    }
    closeExitDialog()
    options.saveBubbleStatesToImage()
    options.emitExit()
  }

  function exitWithoutSaving(): void {
    exitEditMode()
  }

  async function saveAndExit(): Promise<void> {
    if (exitDialogState.value === 'saving') {
      return
    }

    options.saveBubbleStatesToImage()

    const bookId = options.currentBookId.value
    const chapterId = options.currentChapterId.value
    if (!bookId || !chapterId) {
      exitDialogError.value = '当前不在章节上下文中，无法执行整章保存'
      exitDialogState.value = 'error'
      return
    }

    exitDialogError.value = ''
    exitDialogState.value = 'saving'

    try {
      const success = await options.saveChapterSession(bookId, chapterId)
      if (!success) {
        exitDialogError.value = options.sessionSaveError.value || '保存失败，请重试'
        exitDialogState.value = 'error'
        return
      }

      closeExitDialog()
      options.emitExit()
    } catch (error) {
      console.error('[EditWorkspace] 保存后退出失败:', error)
      exitDialogError.value = error instanceof Error
        ? error.message
        : (options.sessionSaveError.value || '保存失败，请重试')
      exitDialogState.value = 'error'
    }
  }

  return {
    exitDialogState,
    exitDialogError,
    shouldPromptSaveOnExit,
    exitSaveCurrent,
    exitSaveTotal,
    exitSaveHasProgress,
    exitSaveProgressPercent,
    exitSaveMessage,
    closeExitDialog,
    openExitDialog,
    exitEditMode,
    exitWithoutSaving,
    saveAndExit,
  }
}
