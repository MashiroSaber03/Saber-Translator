import { ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import { useEditWorkspaceKeyboardShortcuts } from '@/composables/edit/useEditWorkspaceKeyboardShortcuts'
import type { BrushMode } from '@/composables/useBrush'

function harness() {
  const enabled = ref(true)
  const brushMode = ref<BrushMode>(null)
  const actions = {
    cancelColorPick: vi.fn(),
    exitEditMode: vi.fn(),
    deleteSelectedBubbles: vi.fn(),
    goToPreviousImage: vi.fn(),
    goToNextImage: vi.fn(),
    applyAndNext: vi.fn(),
    toggleBrushMode: vi.fn((mode: 'repair' | 'restore') => { brushMode.value = mode }),
    exitBrushMode: vi.fn(() => { brushMode.value = null }),
    zoomIn: vi.fn(),
    zoomOut: vi.fn(),
    resetZoom: vi.fn(),
    exitMaskMode: vi.fn(),
    applyMask: vi.fn(),
  }
  const keyboard = useEditWorkspaceKeyboardShortcuts({
    enabled,
    ...actions,
    isPickingColor: ref(false),
    hasSelection: ref(true),
    brushMode,
    maskActive: ref(true),
  })
  return { ...keyboard, actions, brushMode, enabled }
}

describe('mask editing shares ordinary workspace shortcuts', () => {
  it('does not run hidden ordinary commands while fine editing owns the workspace', () => {
    const h = harness()
    h.enabled.value = false
    for (const key of ['Delete', 'Backspace', 'Escape', 'a', 'd', 'r', 'u', '+', '-', '0']) {
      const event = new KeyboardEvent('keydown', { key, cancelable: true })
      h.handleKeyDown(event)
      expect(event.defaultPrevented).toBe(false)
    }
    h.handleKeyDown(new KeyboardEvent('keydown', { key: 'Enter', ctrlKey: true }))
    for (const action of Object.values(h.actions)) expect(action).not.toHaveBeenCalled()
    h.enabled.value = true
    h.handleKeyDown(new KeyboardEvent('keydown', { key: 'r' }))
    expect(h.brushMode.value).toBe('repair')
  })
  it.each([['r', 'repair'], ['u', 'restore']] as const)('holds %s only until the key is released', (key, mode) => {
    const h = harness()
    h.handleKeyDown(new KeyboardEvent('keydown', { key, cancelable: true }))
    expect(h.brushMode.value).toBe(mode)
    h.handleKeyDown(new KeyboardEvent('keydown', { key, repeat: true }))
    expect(h.actions.toggleBrushMode).toHaveBeenCalledOnce()
    h.handleKeyUp(new KeyboardEvent('keyup', { key }))
    expect(h.brushMode.value).toBeNull()
  })

  it('keeps navigation and zoom available when no brush is active', () => {
    const h = harness()
    for (const key of ['a', 'd', '+', '-', '0']) h.handleKeyDown(new KeyboardEvent('keydown', { key }))
    for (const action of ['goToPreviousImage', 'goToNextImage', 'zoomIn', 'zoomOut', 'resetZoom'] as const)
      expect(h.actions[action]).toHaveBeenCalledOnce()
  })

  it('cancels the brush before leaving mask editing, and cannot delete hidden text boxes', () => {
    const h = harness()
    h.handleKeyDown(new KeyboardEvent('keydown', { key: 'r' }))
    h.handleKeyDown(new KeyboardEvent('keydown', { key: 'Escape' }))
    expect(h.brushMode.value).toBeNull()
    expect(h.actions.exitMaskMode).not.toHaveBeenCalled()
    h.handleKeyDown(new KeyboardEvent('keydown', { key: 'Escape' }))
    expect(h.actions.exitMaskMode).toHaveBeenCalledOnce()
    for (const key of ['Delete', 'Backspace']) h.handleKeyDown(new KeyboardEvent('keydown', { key }))
    expect(h.actions.deleteSelectedBubbles).not.toHaveBeenCalled()
    expect(h.actions.exitEditMode).not.toHaveBeenCalled()
  })

  it('applies the mask with Ctrl+Enter instead of switching pages', () => {
    const h = harness()
    h.handleKeyDown(new KeyboardEvent('keydown', { key: 'Enter', ctrlKey: true }))
    expect(h.actions.applyMask).toHaveBeenCalledOnce()
    expect(h.actions.applyAndNext).not.toHaveBeenCalled()
  })

  it('keeps brush shortcuts available after clicking toolbar buttons', () => {
    const h = harness()
    const button = document.createElement('button')
    button.addEventListener('keydown', h.handleKeyDown)
    h.brushMode.value = 'restore'
    button.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    expect(h.brushMode.value).toBeNull()
    expect(h.actions.exitMaskMode).not.toHaveBeenCalled()
    button.dispatchEvent(new KeyboardEvent('keydown', { key: 'r' }))
    expect(h.brushMode.value).toBe('repair')
    h.handleKeyUp(new KeyboardEvent('keyup', { key: 'r' }))
    expect(h.brushMode.value).toBeNull()
  })

  it.each(['input', 'textarea', 'select', '[role="combobox"]', '[role="dialog"] button'])(
    'does not intercept keys in %s', selector => {
      const h = harness()
      const root = document.createElement('div')
      if (selector === '[role="combobox"]') root.innerHTML = '<button role="combobox"></button>'
      else if (selector === '[role="dialog"] button') root.innerHTML = '<div role="dialog"><button></button></div>'
      else root.innerHTML = `<${selector}></${selector}>`
      const target = root.querySelector(selector)!
      target.addEventListener('keydown', h.handleKeyDown as EventListener)
      for (const key of ['r', 'Escape', 'a', 'Delete'])
        target.dispatchEvent(new KeyboardEvent('keydown', { key }))
      for (const action of Object.values(h.actions)) expect(action).not.toHaveBeenCalled()
    },
  )

  it.each([
    { key: 'r', ctrlKey: true }, { key: 'u', metaKey: true },
    { key: 'a', ctrlKey: true }, { key: 'd', altKey: true },
    { key: '+', ctrlKey: true }, { key: '-', metaKey: true },
  ])('leaves browser shortcut $key with modifiers untouched', options => {
    const h = harness()
    const event = new KeyboardEvent('keydown', { ...options, cancelable: true })
    h.handleKeyDown(event)
    expect(event.defaultPrevented).toBe(false)
    expect(h.brushMode.value).toBeNull()
    for (const action of Object.values(h.actions)) expect(action).not.toHaveBeenCalled()
  })
})
