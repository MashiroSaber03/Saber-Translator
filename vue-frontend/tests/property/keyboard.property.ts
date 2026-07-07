import { readFileSync } from 'node:fs'
import { ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import * as fc from 'fast-check'
import { useEditWorkspaceKeyboardShortcuts } from '@/composables/edit/useEditWorkspaceKeyboardShortcuts'
import type { BrushMode } from '@/composables/useBrush'
import type { ExitDialogState } from '@/composables/edit/useEditWorkspaceExit'

type ShortcutOptions = Parameters<typeof useEditWorkspaceKeyboardShortcuts>[0]
type ShortcutCallback = keyof Pick<
  ShortcutOptions,
  | 'closeExitDialog'
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
    exitDialogState: ref<ExitDialogState>('closed'),
    brushMode: ref<BrushMode>(null),
    hasSelection: ref(true),
    isBrushKeyDown: ref(false),
    closeExitDialog: vi.fn(),
    exitEditMode: vi.fn(),
    deleteSelectedBubbles: vi.fn(),
    goToPreviousImage: vi.fn(),
    goToNextImage: vi.fn(),
    applyAndNext: vi.fn(),
    toggleBrushMode: vi.fn(),
    exitBrushMode: vi.fn(),
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
  it('uses the product shortcut composable without a copied dispatcher', () => {
    const source = readFileSync('tests/property/keyboard.property.ts', 'utf8')

    expect(source).toContain("from '@/composables/edit/useEditWorkspaceKeyboardShortcuts'")
    for (const shadowContract of [
      'function isInInput' + 'Element',
      'function match' + 'Key',
      'function handleKeyboard' + 'Event',
      'function formatKey' + 'Combo',
      'interface Keyboard' + 'Handler',
      'MockKeyboard' + 'Event',
    ]) {
      expect(source).not.toContain(shadowContract)
    }
    expect(source).not.toMatch(/={6,}/)
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

  it('treats Escape as either an exit request or an exit-dialog close request', () => {
    fc.assert(
      fc.property(fc.constantFrom<ExitDialogState>('closed', 'confirm', 'error', 'saving'), (state) => {
        const { options, handleKeyDown } = createShortcutHarness({
          exitDialogState: ref(state),
        })
        const event = createKeyEvent('Escape')

        handleKeyDown(event)

        expectCallback(options, 'exitEditMode', state === 'closed' ? 1 : 0)
        expectCallback(options, 'closeExitDialog', state === 'confirm' || state === 'error' ? 1 : 0)
        expect(event.defaultPrevented).toBe(state === 'confirm' || state === 'error')
      }),
      { numRuns: 100 },
    )
  })

  it('keeps text entry fields from swallowing editing text while preserving global navigation keys', () => {
    const textarea = document.createElement('textarea')
    const input = document.createElement('input')
    const button = document.createElement('button')
    const inputBlur = vi.spyOn(input, 'blur')
    const buttonBlur = vi.spyOn(button, 'blur')

    const textareaHarness = createShortcutHarness()
    textareaHarness.handleKeyDown(createKeyEvent('a', { target: textarea }))
    expectCallback(textareaHarness.options, 'goToPreviousImage', 0)

    const inputHarness = createShortcutHarness()
    inputHarness.handleKeyDown(createKeyEvent('a', { target: input }))
    expectCallback(inputHarness.options, 'goToPreviousImage', 1)
    expect(inputBlur).toHaveBeenCalledTimes(1)

    const buttonHarness = createShortcutHarness()
    buttonHarness.handleKeyDown(createKeyEvent('d', { target: button }))
    expectCallback(buttonHarness.options, 'goToNextImage', 1)
    expect(buttonBlur).toHaveBeenCalledTimes(1)

    const enterHarness = createShortcutHarness()
    enterHarness.handleKeyDown(createKeyEvent('Enter', { ctrlKey: true, target: input }))
    expectCallback(enterHarness.options, 'applyAndNext', 0)
  })

  it('toggles temporary brush modes and exits them on keyup', () => {
    const cases = [
      { key: 'r', mode: 'repair' },
      { key: 'R', mode: 'repair' },
      { key: 'u', mode: 'restore' },
      { key: 'U', mode: 'restore' },
    ] as const

    fc.assert(
      fc.property(fc.constantFrom(...cases), fc.boolean(), ({ key, mode }, isBrushKeyDown) => {
        const { options, handleKeyDown, handleKeyUp } = createShortcutHarness({
          isBrushKeyDown: ref(isBrushKeyDown),
        })
        const keydownEvent = createKeyEvent(key)

        handleKeyDown(keydownEvent)

        expect(options.toggleBrushMode).toHaveBeenCalledTimes(isBrushKeyDown ? 0 : 1)
        if (!isBrushKeyDown) {
          expect(options.toggleBrushMode).toHaveBeenCalledWith(mode)
        }
        expect(keydownEvent.defaultPrevented).toBe(!isBrushKeyDown)

        const keyupEvent = createKeyEvent(key, { type: 'keyup' })
        handleKeyUp(keyupEvent)

        expectCallback(options, 'exitBrushMode', 1)
        expect(keyupEvent.defaultPrevented).toBe(true)
      }),
      { numRuns: 100 },
    )
  })
})
