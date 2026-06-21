import type { Ref } from 'vue'
import type { BrushMode } from '@/composables/useBrush'
import type { ExitDialogState } from './useEditWorkspaceExit'

interface UseEditWorkspaceKeyboardShortcutsOptions {
  exitDialogState: Ref<ExitDialogState>
  brushMode: Ref<BrushMode>
  hasSelection: Ref<boolean>
  isBrushKeyDown: Ref<boolean>
  closeExitDialog: () => void
  exitEditMode: () => void
  deleteSelectedBubbles: () => void
  goToPreviousImage: () => void
  goToNextImage: () => void
  applyAndNext: () => Promise<void> | void
  toggleBrushMode: (mode: 'repair' | 'restore') => void
  exitBrushMode: () => void
  zoomIn: () => void
  zoomOut: () => void
  resetZoom: () => void
}

export function useEditWorkspaceKeyboardShortcuts(options: UseEditWorkspaceKeyboardShortcutsOptions) {
  function handleKeyDown(event: KeyboardEvent): void {
    const target = event.target as HTMLElement
    const key = event.key.toLowerCase()

    if (options.exitDialogState.value !== 'closed') {
      if (event.key === 'Escape' && options.exitDialogState.value !== 'saving') {
        options.closeExitDialog()
        event.preventDefault()
      }
      return
    }

    if (key === 'r' || key === 'u' || key === 'a' || key === 'd') {
      if (target.tagName === 'TEXTAREA') return
      if (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'BUTTON') {
        target.blur()
      }
    } else if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') {
      return
    }

    switch (event.key) {
      case 'Escape':
        options.exitEditMode()
        break
      case 'Delete':
      case 'Backspace':
        if (!options.brushMode.value && options.hasSelection.value) {
          options.deleteSelectedBubbles()
          event.preventDefault()
        }
        break
      case 'a':
      case 'A':
        if (!options.brushMode.value) {
          options.goToPreviousImage()
          event.preventDefault()
        }
        break
      case 'd':
      case 'D':
        if (!options.brushMode.value) {
          options.goToNextImage()
          event.preventDefault()
        }
        break
      case 'Enter':
        if (event.ctrlKey && !options.brushMode.value) {
          void options.applyAndNext()
          event.preventDefault()
        }
        break
      case 'r':
      case 'R':
        if (!options.isBrushKeyDown.value) {
          options.toggleBrushMode('repair')
          event.preventDefault()
        }
        break
      case 'u':
      case 'U':
        if (!options.isBrushKeyDown.value) {
          options.toggleBrushMode('restore')
          event.preventDefault()
        }
        break
      case '+':
      case '=':
        options.zoomIn()
        event.preventDefault()
        break
      case '-':
        options.zoomOut()
        event.preventDefault()
        break
      case '0':
        options.resetZoom()
        event.preventDefault()
        break
    }
  }

  function handleKeyUp(event: KeyboardEvent): void {
    if (event.key === 'r' || event.key === 'R' || event.key === 'u' || event.key === 'U') {
      options.exitBrushMode()
      event.preventDefault()
    }
  }

  return {
    handleKeyDown,
    handleKeyUp,
  }
}
