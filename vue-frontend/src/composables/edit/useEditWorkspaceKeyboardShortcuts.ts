import type { Ref } from 'vue'
import type { BrushMode } from '@/composables/useBrush'

interface UseEditWorkspaceKeyboardShortcutsOptions {
  isPickingColor: Readonly<Ref<boolean>>
  cancelColorPick: () => void
  brushMode: Ref<BrushMode>
  hasSelection: Ref<boolean>
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
  let heldBrushKey: 'r' | 'u' | null = null

  function cancelBrushShortcut(): void {
    if (!heldBrushKey) return
    const mode = heldBrushKey === 'r' ? 'repair' : 'restore'
    heldBrushKey = null
    if (options.brushMode.value === mode) options.exitBrushMode()
  }

  function isEditableTarget(target: EventTarget | null): boolean {
    if (!(target instanceof Element)) return false
    return Boolean(target.closest(
      'input, textarea, select, button, [role="dialog"], [contenteditable]:not([contenteditable="false"])',
    ))
  }

  function handleKeyDown(event: KeyboardEvent): void {
    if (event.defaultPrevented) return
    if (options.isPickingColor.value) {
      if (event.key === 'Escape') {
        options.cancelColorPick()
        event.preventDefault()
        event.stopPropagation()
      } else if (event.key !== 'Tab') {
        if (isEditableTarget(event.target)) options.cancelColorPick()
        else event.preventDefault()
      }
      return
    }
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
      case 'u':
      case 'U': {
        if (!heldBrushKey && !options.brushMode.value) {
          const key = event.key.toLowerCase() === 'r' ? 'r' : 'u'
          const mode = key === 'r' ? 'repair' : 'restore'
          options.toggleBrushMode(mode)
          if (options.brushMode.value === mode) heldBrushKey = key
          event.preventDefault()
        }
        break
      }
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
    if (event.key.toLowerCase() !== heldBrushKey) return
    // Release the shortcut even if focus moved into an input or color popover.
    cancelBrushShortcut()
    event.preventDefault()
  }

  return {
    cancelBrushShortcut,
    handleKeyDown,
    handleKeyUp,
  }
}
