import type { Ref } from 'vue'
import type { BrushMode } from '@/composables/useBrush'

interface UseEditWorkspaceKeyboardShortcutsOptions {
  brushMode: Ref<BrushMode>
  hasSelection: Ref<boolean>
  isBrushKeyDown: Ref<boolean>
  exitEditMode: () => Promise<void> | void
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
  function isEditableTarget(target: EventTarget | null): boolean {
    if (!(target instanceof Element)) return false
    return Boolean(target.closest(
      'input, textarea, select, button, [contenteditable]:not([contenteditable="false"])',
    ))
  }

  function handleKeyDown(event: KeyboardEvent): void {
    if (isEditableTarget(event.target)) return

    switch (event.key) {
      case 'Escape':
        if (options.brushMode.value) {
          options.exitBrushMode()
        } else {
          void options.exitEditMode()
        }
        event.preventDefault()
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
          void options.goToPreviousImage()
          event.preventDefault()
        }
        break
      case 'd':
      case 'D':
        if (!options.brushMode.value) {
          void options.goToNextImage()
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
    if (isEditableTarget(event.target)) return
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
