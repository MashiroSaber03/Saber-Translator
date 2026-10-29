import { ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import RichTextInput from '@/components/edit/RichTextInput.vue'
import { createBubbleState } from '@/utils/bubbleFactory'
import { useEditWorkspaceKeyboardShortcuts } from '@/composables/edit/useEditWorkspaceKeyboardShortcuts'
import type { BrushMode } from '@/composables/useBrush'

function harness() {
  const enabled = ref(true)
  const brushMode = ref<BrushMode>(null)
  const maskActive = ref(true)
  const actions = {
    cancelColorPick: vi.fn(),
    exitEditMode: vi.fn(),
    deleteSelectedBubbles: vi.fn(),
    duplicateSelectedBubbles: vi.fn(),
    selectAllBubbles: vi.fn(),
    nudgeSelectedText: vi.fn(),
    goToPreviousImage: vi.fn(),
    goToNextImage: vi.fn(),
    applyAndNext: vi.fn(),
    save: vi.fn(),
    selectBubble: vi.fn(),
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
    maskActive,
  })
  return { ...keyboard, actions, brushMode, enabled, maskActive }
}

describe('mask editing shares ordinary workspace shortcuts', () => {
  it('handles Ctrl+Enter before the real rich-input library suppresses native Enter, without taking over normal input or Escape', async () => {
    const h = harness()
    h.maskActive.value = false
    const wrapper = mount(RichTextInput, { props: { bubble: createBubbleState({ translatedText: '校对文字' }), fonts: [] }, attachTo: document.body })
    document.addEventListener('keydown', h.handleShortcutCapture, true)
    document.addEventListener('keydown', h.handleKeyDown)
    try {
      await wrapper.get('[role="textbox"]').trigger('keydown', { key: 'Enter', keyCode: 13, ctrlKey: true })
      expect(h.actions.applyAndNext).toHaveBeenCalledWith(true)
      expect(wrapper.emitted('update')).toBeUndefined()
      await wrapper.get('[role="textbox"]').trigger('keydown', { key: 'Enter', keyCode: 13 })
      expect(h.actions.applyAndNext).toHaveBeenCalledOnce()
      expect((wrapper.emitted('update')!.at(-1)![0] as { translatedText: string }).translatedText).toContain('\n')
      h.handleShortcutCapture(new KeyboardEvent('keydown', { key: 'Escape' }))
      expect(h.actions.exitEditMode).not.toHaveBeenCalled()
    } finally {
      document.removeEventListener('keydown', h.handleShortcutCapture, true)
      document.removeEventListener('keydown', h.handleKeyDown)
      wrapper.unmount()
    }
  })
  it('supports submit, save and bubble navigation while proofreading text, but leaves IME composition alone', () => {
    const h = harness()
    h.maskActive.value = false
    const input = document.createElement('div')
    input.setAttribute('contenteditable', 'true')
    const dispatch = (key: string, init: KeyboardEventInit) => {
      const event = new KeyboardEvent('keydown', { key, cancelable: true, ...init })
      Object.defineProperty(event, 'target', { value: input })
      h.handleKeyDown(event)
      return event
    }
    expect(dispatch('Enter', { ctrlKey: true, isComposing: true }).defaultPrevented).toBe(false)
    expect(h.actions.applyAndNext).not.toHaveBeenCalled()
    expect(dispatch('Enter', { ctrlKey: true }).defaultPrevented).toBe(true)
    expect(h.actions.applyAndNext).toHaveBeenCalledOnce()
    expect(dispatch('ArrowDown', { altKey: true }).defaultPrevented).toBe(true)
    expect(h.actions.selectBubble).toHaveBeenCalledWith(1, true)
    dispatch('s', { ctrlKey: true })
    expect(h.actions.save).toHaveBeenCalledOnce()
    dispatch('u', {})
    expect(h.actions.toggleBrushMode).not.toHaveBeenCalled()
  })
  it('duplicates selected text boxes and nudges text without changing page navigation', () => {
    const h = harness()
    h.maskActive.value = false
    h.handleKeyDown(new KeyboardEvent('keydown', { key: 'a', ctrlKey: true }))
    expect(h.actions.selectAllBubbles).toHaveBeenCalledOnce()
    const duplicate = new KeyboardEvent('keydown', { key: 'd', ctrlKey: true, cancelable: true })
    h.handleKeyDown(duplicate)
    h.handleKeyDown(new KeyboardEvent('keydown', { key: 'd', ctrlKey: true, repeat: true }))
    expect(duplicate.defaultPrevented).toBe(true)
    expect(h.actions.duplicateSelectedBubbles).toHaveBeenCalledOnce()
    expect(h.actions.goToNextImage).not.toHaveBeenCalled()
    h.handleKeyDown(new KeyboardEvent('keydown', { key: 'ArrowLeft' }))
    h.handleKeyDown(new KeyboardEvent('keydown', { key: 'ArrowDown', shiftKey: true }))
    expect(h.actions.nudgeSelectedText.mock.calls).toEqual([[-1, 0], [0, 10]])
  })

  it('leaves text-box shortcuts inactive in mask mode and while painting', () => {
    const h = harness()
    const keys = () => {
      h.handleKeyDown(new KeyboardEvent('keydown', { key: 'd', ctrlKey: true }))
      h.handleKeyDown(new KeyboardEvent('keydown', { key: 'a', ctrlKey: true }))
      h.handleKeyDown(new KeyboardEvent('keydown', { key: 'ArrowRight' }))
    }
    keys()
    h.maskActive.value = false
    h.brushMode.value = 'repair'
    keys()
    expect(h.actions.duplicateSelectedBubbles).not.toHaveBeenCalled()
    expect(h.actions.nudgeSelectedText).not.toHaveBeenCalled()
    expect(h.actions.selectAllBubbles).not.toHaveBeenCalled()
  })
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
      for (const key of ['r', 'Escape', 'a', 'Delete', 'ArrowLeft'])
        target.dispatchEvent(new KeyboardEvent('keydown', { key }))
      target.dispatchEvent(new KeyboardEvent('keydown', { key: 'd', ctrlKey: true }))
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
