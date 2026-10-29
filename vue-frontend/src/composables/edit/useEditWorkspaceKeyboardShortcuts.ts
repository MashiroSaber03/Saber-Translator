import type { Ref } from 'vue'
import type { BrushMode } from '@/composables/useBrush'

interface UseEditWorkspaceKeyboardShortcutsOptions {
  enabled?: Readonly<Ref<boolean>>
  isPickingColor: Readonly<Ref<boolean>>
  cancelColorPick: () => void
  brushMode: Ref<BrushMode>
  hasSelection: Ref<boolean>
  exitEditMode: () => Promise<void> | void
  deleteSelectedBubbles: () => void
  duplicateSelectedBubbles?: () => void
  selectAllBubbles?: () => void
  nudgeSelectedText?: (dx: number, dy: number) => void
  goToPreviousImage: () => void
  goToNextImage: () => void
  applyAndNext: (focusText?: boolean) => Promise<void> | void
  save?: () => unknown
  selectBubble?: (delta: number, focusText: boolean) => void
  toggleBrushMode: (mode: 'repair' | 'restore') => void
  exitBrushMode: () => void
  zoomIn: () => void
  zoomOut: () => void
  resetZoom: () => void
  maskActive?: Readonly<Ref<boolean>>
  exitMaskMode?: () => void
  applyMask?: () => Promise<void>
  undo?: () => void
  redo?: () => void
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
      'input, textarea, select, [role="combobox"], [role="dialog"], [contenteditable]:not([contenteditable="false"])',
    ))
  }

  function handleKeyDown(event: KeyboardEvent): void {
    if (options.enabled?.value === false) return
    if (event.defaultPrevented) return
    if (event.isComposing || event.keyCode === 229) return
    if (options.isPickingColor.value) {
      if (event.key === 'Escape') {
        options.cancelColorPick()
        event.preventDefault()
        event.stopPropagation()
      } else if (event.key !== 'Tab') {
        if (isEditableTarget(event.target) ||
            (event.target instanceof Element && event.target.closest('button'))) options.cancelColorPick()
        else event.preventDefault()
      }
      return
    }
    const editable = isEditableTarget(event.target)
    if (event.target instanceof Element && event.target.closest('[role="dialog"]')) return
    const textInput = event.target instanceof Element && !!event.target.closest('textarea, [contenteditable="true"]')
    if ((event.ctrlKey || event.metaKey) && !event.altKey && !event.shiftKey && event.key.toLowerCase() === 's' && options.save) {
      options.save()
      event.preventDefault()
      return
    }
    if ((!editable || textInput) && event.ctrlKey && !event.altKey && !event.shiftKey && event.key === 'Enter' && !options.brushMode.value) {
      if (options.maskActive?.value) void options.applyMask?.()
      else void options.applyAndNext(Boolean(textInput))
      event.preventDefault()
      return
    }
    if ((!editable || textInput) && event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey && !options.maskActive?.value && !options.brushMode.value
        && (event.key === 'ArrowDown' || event.key === 'ArrowUp') && options.selectBubble) {
      options.selectBubble(event.key === 'ArrowDown' ? 1 : -1, Boolean(textInput))
      event.preventDefault()
      return
    }
    if (editable) return
    const canEditText = !options.maskActive?.value && !options.brushMode.value && options.hasSelection.value
    if ((event.ctrlKey || event.metaKey) && !event.altKey) {
      const key = event.key.toLowerCase()
      if (key === 'a' && !event.shiftKey && !options.maskActive?.value && !options.brushMode.value && options.selectAllBubbles) {
        options.selectAllBubbles()
        event.preventDefault()
        return
      }
      if (key === 'd' && !event.shiftKey && canEditText && options.duplicateSelectedBubbles) {
        if (!event.repeat) options.duplicateSelectedBubbles()
        event.preventDefault()
        return
      }
      if (key === 'z' || key === 'y') {
        if (key === 'y' || event.shiftKey) options.redo?.()
        else options.undo?.()
        event.preventDefault()
        return
      }
    }
    if ((event.ctrlKey || event.metaKey || event.altKey) &&
        !(event.key === 'Enter' && event.ctrlKey && !event.metaKey && !event.altKey)) return
    switch (event.key) {
      case 'ArrowLeft':
      case 'ArrowRight':
      case 'ArrowUp':
      case 'ArrowDown': {
        if (!canEditText || !options.nudgeSelectedText) break
        const step = event.shiftKey ? 10 : 1
        options.nudgeSelectedText(event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0,
          event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0)
        event.preventDefault()
        break
      }
      case 'Escape':
        if (options.brushMode.value) {
          options.exitBrushMode()
        } else if (options.maskActive?.value) {
          options.exitMaskMode?.()
        } else {
          void options.exitEditMode()
        }
        event.preventDefault()
        break
      case 'Delete':
      case 'Backspace':
        if (!options.maskActive?.value && !options.brushMode.value && options.hasSelection.value) {
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

  // Only these explicit combinations precede the rich-input library. Ordinary
  // keys still bubble after dropdowns, popovers and the input handle them.
  function handleShortcutCapture(event: KeyboardEvent): void {
    if (((event.ctrlKey || event.metaKey) && (event.key === 'Enter' || event.key.toLowerCase() === 's'))
      || (event.altKey && (event.key === 'ArrowUp' || event.key === 'ArrowDown'))) handleKeyDown(event)
  }

  return {
    handleShortcutCapture,
    cancelBrushShortcut,
    handleKeyDown,
    handleKeyUp,
  }
}
