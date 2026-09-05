import { ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import * as fc from 'fast-check'
import { useEditWorkspaceKeyboardShortcuts } from '@/composables/edit/useEditWorkspaceKeyboardShortcuts'
import type { BrushMode } from '@/composables/useBrush'

type ShortcutOptions = Parameters<typeof useEditWorkspaceKeyboardShortcuts>[0]
type ShortcutCallback = keyof Pick<
  ShortcutOptions,
  | 'exitEditMode'
  | 'deleteSelectedBubbles'
  | 'goToPreviousImage'
  | 'goToNextImage'
  | 'applyAndNext'
  | 'toggleBrushMode'
  | 'exitBrushMode'
  | 'zoomIn'
  | 'zoomOut'
  | 'resetZoom'
>

interface KeyEventOptions {
  ctrlKey?: boolean
  target?: HTMLElement
  type?: 'keydown' | 'keyup'
}

const brushModeArb = fc.constantFrom<BrushMode>(null, 'repair', 'restore')

function createShortcutHarness(overrides: Partial<ShortcutOptions> = {}) {
  const options: ShortcutOptions = {
    isPickingColor: ref(false),
    cancelColorPick: vi.fn(),
    brushMode: ref<BrushMode>(null),
    hasSelection: ref(true),
    exitEditMode: vi.fn(),
    deleteSelectedBubbles: vi.fn(),
    goToPreviousImage: vi.fn(),
    goToNextImage: vi.fn(),
    applyAndNext: vi.fn(),
    toggleBrushMode: vi.fn((mode: 'repair' | 'restore') => { options.brushMode.value = mode }),
    exitBrushMode: vi.fn(() => { options.brushMode.value = null }),
    zoomIn: vi.fn(),
    zoomOut: vi.fn(),
    resetZoom: vi.fn(),
    ...overrides,
  }

  return {
    options,
    ...useEditWorkspaceKeyboardShortcuts(options),
  }
}

function createKeyEvent(key: string, options: KeyEventOptions = {}): KeyboardEvent {
  const event = new KeyboardEvent(options.type ?? 'keydown', {
    key,
    ctrlKey: options.ctrlKey ?? false,
    cancelable: true,
    bubbles: true,
  })
  Object.defineProperty(event, 'target', {
    value: options.target ?? document.createElement('div'),
  })
  return event
}

function expectCallback(options: ShortcutOptions, callback: ShortcutCallback, times: number): void {
  expect(options[callback]).toHaveBeenCalledTimes(times)
}

describe('edit keyboard shortcut properties', () => {
  it('gives color picking priority over editing shortcuts and cancels without exiting', () => {
    const { options, handleKeyDown } = createShortcutHarness({ isPickingColor: ref(true) })
    for (const key of ['a', 'd', 'Delete', 'r', 'u', 'Enter']) handleKeyDown(createKeyEvent(key, { ctrlKey: true }))
    expect(options.goToNextImage).not.toHaveBeenCalled()
    expect(options.goToPreviousImage).not.toHaveBeenCalled()
    expect(options.deleteSelectedBubbles).not.toHaveBeenCalled()
    expect(options.toggleBrushMode).not.toHaveBeenCalled()
    expect(options.applyAndNext).not.toHaveBeenCalled()
    handleKeyDown(createKeyEvent('Escape'))
    expect(options.cancelColorPick).toHaveBeenCalledOnce()
    expect(options.exitEditMode).not.toHaveBeenCalled()
  })

  it('allows Tab and button activation while picking, and leaves modal Escape to the dialog', () => {
    const { options, handleKeyDown } = createShortcutHarness({ isPickingColor: ref(true) })
    const button = document.createElement('button')
    const tab = createKeyEvent('Tab', { target: button })
    handleKeyDown(tab)
    expect(tab.defaultPrevented).toBe(false)
    expect(options.cancelColorPick).not.toHaveBeenCalled()
    const enter = createKeyEvent('Enter', { target: button })
    handleKeyDown(enter)
    expect(enter.defaultPrevented).toBe(false)
    expect(options.cancelColorPick).toHaveBeenCalledOnce()

    const modal = document.createElement('div')
    modal.setAttribute('role', 'dialog')
    const normal = createShortcutHarness()
    normal.handleKeyDown(createKeyEvent('Escape', { target: modal }))
    expect(normal.options.exitEditMode).not.toHaveBeenCalled()
  })

  it('routes single-key edit actions through the current composable contract', () => {
    const cases = [
      { key: 'a', callback: 'goToPreviousImage' },
      { key: 'A', callback: 'goToPreviousImage' },
      { key: 'd', callback: 'goToNextImage' },
      { key: 'D', callback: 'goToNextImage' },
      { key: '+', callback: 'zoomIn' },
      { key: '=', callback: 'zoomIn' },
      { key: '-', callback: 'zoomOut' },
      { key: '0', callback: 'resetZoom' },
    ] as const

    fc.assert(
      fc.property(fc.constantFrom(...cases), ({ key, callback }) => {
        const { options, handleKeyDown } = createShortcutHarness()
        const event = createKeyEvent(key)

        handleKeyDown(event)

        expectCallback(options, callback, 1)
        expect(event.defaultPrevented).toBe(true)
      }),
      { numRuns: 100 },
    )
  })

  it('deletes bubbles only when there is an edit selection and no brush mode', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('Delete', 'Backspace'),
        fc.boolean(),
        brushModeArb,
        (key, hasSelection, brushMode) => {
          const { options, handleKeyDown } = createShortcutHarness({
            hasSelection: ref(hasSelection),
            brushMode: ref(brushMode),
          })
          const event = createKeyEvent(key)

          handleKeyDown(event)

          const shouldDelete = hasSelection && brushMode === null
          expectCallback(options, 'deleteSelectedBubbles', shouldDelete ? 1 : 0)
          expect(event.defaultPrevented).toBe(shouldDelete)
        },
      ),
      { numRuns: 100 },
    )
  })

  it('applies and advances only for Ctrl Enter outside brush mode', () => {
    fc.assert(
      fc.property(fc.boolean(), brushModeArb, (ctrlKey, brushMode) => {
        const { options, handleKeyDown } = createShortcutHarness({
          brushMode: ref(brushMode),
        })
        const event = createKeyEvent('Enter', { ctrlKey })

        handleKeyDown(event)

        const shouldApply = ctrlKey && brushMode === null
        expectCallback(options, 'applyAndNext', shouldApply ? 1 : 0)
        expect(event.defaultPrevented).toBe(shouldApply)
      }),
      { numRuns: 100 },
    )
  })

  it('flushes through the exit callback when Escape is pressed', () => {
    const { options, handleKeyDown } = createShortcutHarness()
    handleKeyDown(createKeyEvent('Escape'))
    expectCallback(options, 'exitEditMode', 1)
  })

  it('never hijacks edit shortcuts from editable controls', () => {
    const textarea = document.createElement('textarea')
    const input = document.createElement('input')
    const button = document.createElement('button')
    const contentEditable = document.createElement('div')
    contentEditable.setAttribute('contenteditable', 'true')
    const inputBlur = vi.spyOn(input, 'blur')
    const buttonBlur = vi.spyOn(button, 'blur')

    const textareaHarness = createShortcutHarness()
    textareaHarness.handleKeyDown(createKeyEvent('a', { target: textarea }))
    expectCallback(textareaHarness.options, 'goToPreviousImage', 0)

    const inputHarness = createShortcutHarness()
    inputHarness.handleKeyDown(createKeyEvent('a', { target: input }))
    expectCallback(inputHarness.options, 'goToPreviousImage', 0)
    expect(inputBlur).not.toHaveBeenCalled()

    const buttonHarness = createShortcutHarness()
    buttonHarness.handleKeyDown(createKeyEvent('d', { target: button }))
    expectCallback(buttonHarness.options, 'goToNextImage', 0)
    expect(buttonBlur).not.toHaveBeenCalled()

    const contentEditableHarness = createShortcutHarness()
    contentEditableHarness.handleKeyDown(createKeyEvent('Backspace', { target: contentEditable }))
    expectCallback(contentEditableHarness.options, 'deleteSelectedBubbles', 0)

    const enterHarness = createShortcutHarness()
    enterHarness.handleKeyDown(createKeyEvent('Enter', { ctrlKey: true, target: input }))
    expectCallback(enterHarness.options, 'applyAndNext', 0)
  })

  it.each(['input', 'button', 'dialog', 'slider'])(
    'releases a held brush shortcut after focus moves into a %s',
    targetType => {
      const { options, handleKeyDown, handleKeyUp } = createShortcutHarness()
      const target = document.createElement(['input', 'button'].includes(targetType) ? targetType : 'div')
      if (targetType === 'dialog') target.setAttribute('role', 'dialog')
      if (targetType === 'slider') {
        const dialog = document.createElement('div')
        dialog.setAttribute('role', 'dialog')
        target.setAttribute('role', 'slider')
        dialog.append(target)
      }
      handleKeyDown(createKeyEvent('r'))
      const release = createKeyEvent('r', { target, type: 'keyup' })

      handleKeyUp(release)

      expectCallback(options, 'exitBrushMode', 1)
      expect(release.defaultPrevented).toBe(true)
    },
  )

  it('uses Escape to leave brush mode before leaving the edit workspace', () => {
    const { options, handleKeyDown } = createShortcutHarness({
      brushMode: ref<BrushMode>('repair'),
    })
    const event = createKeyEvent('Escape')

    handleKeyDown(event)

    expectCallback(options, 'exitBrushMode', 1)
    expectCallback(options, 'exitEditMode', 0)
    expect(event.defaultPrevented).toBe(true)
  })

  it('toggles temporary brush modes and exits them on keyup', () => {
    const cases = [
      { key: 'r', mode: 'repair' },
      { key: 'R', mode: 'repair' },
      { key: 'u', mode: 'restore' },
      { key: 'U', mode: 'restore' },
    ] as const

    fc.assert(
      fc.property(fc.constantFrom(...cases), brushModeArb, ({ key, mode }, activeMode) => {
        const { options, handleKeyDown, handleKeyUp } = createShortcutHarness({
          brushMode: ref(activeMode),
        })
        const keydownEvent = createKeyEvent(key)

        handleKeyDown(keydownEvent)

        expect(options.toggleBrushMode).toHaveBeenCalledTimes(activeMode ? 0 : 1)
        if (!activeMode) {
          expect(options.toggleBrushMode).toHaveBeenCalledWith(mode)
        }
        expect(keydownEvent.defaultPrevented).toBe(!activeMode)

        const keyupEvent = createKeyEvent(key, { type: 'keyup' })
        handleKeyUp(keyupEvent)

        expectCallback(options, 'exitBrushMode', activeMode ? 0 : 1)
        expect(keyupEvent.defaultPrevented).toBe(!activeMode)
      }),
      { numRuns: 100 },
    )
  })

  it('keeps the active shortcut until its own key is released', () => {
    const { options, handleKeyDown, handleKeyUp } = createShortcutHarness()
    handleKeyDown(createKeyEvent('r'))
    handleKeyDown(createKeyEvent('r'))
    handleKeyDown(createKeyEvent('u'))
    handleKeyUp(createKeyEvent('u', { type: 'keyup' }))
    expect(options.toggleBrushMode).toHaveBeenCalledOnce()
    expect(options.brushMode.value).toBe('repair')

    handleKeyUp(createKeyEvent('R', { type: 'keyup' }))
    expect(options.brushMode.value).toBeNull()
    handleKeyDown(createKeyEvent('u'))
    expect(options.brushMode.value).toBe('restore')
  })

  it('leaves a manually activated brush alone when typing in controls', () => {
    const { options, handleKeyDown, handleKeyUp, cancelBrushShortcut } = createShortcutHarness({
      brushMode: ref('repair'),
    })
    handleKeyDown(createKeyEvent('r', { target: document.createElement('input') }))
    handleKeyUp(createKeyEvent('r', { type: 'keyup' }))
    cancelBrushShortcut()
    expect(options.brushMode.value).toBe('repair')
    expect(options.exitBrushMode).not.toHaveBeenCalled()
  })

  it('releases a held shortcut on window blur and allows the next shortcut', () => {
    const { options, handleKeyDown, cancelBrushShortcut } = createShortcutHarness()
    handleKeyDown(createKeyEvent('r'))
    cancelBrushShortcut()
    expect(options.brushMode.value).toBeNull()
    handleKeyDown(createKeyEvent('u'))
    expect(options.brushMode.value).toBe('restore')
  })

  it('does not claim a blocked brush activation or close a different brush mode', () => {
    const blocked = createShortcutHarness({ toggleBrushMode: vi.fn() })
    blocked.handleKeyDown(createKeyEvent('r'))
    blocked.options.brushMode.value = 'repair'
    blocked.handleKeyUp(createKeyEvent('r', { type: 'keyup' }))
    expect(blocked.options.exitBrushMode).not.toHaveBeenCalled()

    const changed = createShortcutHarness()
    changed.handleKeyDown(createKeyEvent('r'))
    changed.options.brushMode.value = 'restore'
    changed.handleKeyUp(createKeyEvent('r', { type: 'keyup' }))
    expect(changed.options.brushMode.value).toBe('restore')
  })
})
