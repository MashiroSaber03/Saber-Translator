import { nextTick, onBeforeUnmount, watch, type Ref } from 'vue'

type DialogEntry = {
  id: symbol
  container: Ref<HTMLElement | null>
  close: () => void
  closeOnEscape: () => boolean
  previousFocus: HTMLElement | null
}

const dialogStack: DialogEntry[] = []
const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

function getTopDialog(): DialogEntry | undefined {
  return dialogStack.at(-1)
}

function getFocusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR))
    .filter(element => !element.hidden && element.getAttribute('aria-hidden') !== 'true')
}

function focusDialog(entry: DialogEntry): void {
  if (getTopDialog()?.id !== entry.id) return
  const container = entry.container.value
  if (!container) return
  const autofocusTarget = container.querySelector<HTMLElement>('[autofocus]')
  const target = autofocusTarget ?? getFocusableElements(container)[0] ?? container
  target.focus({ preventScroll: true })
}

function handleDocumentKeydown(event: KeyboardEvent): void {
  const entry = getTopDialog()
  const container = entry?.container.value
  if (!entry || !container) return

  if (event.key === 'Escape' && entry.closeOnEscape()) {
    event.preventDefault()
    entry.close()
    return
  }

  if (event.key !== 'Tab') return
  const focusable = getFocusableElements(container)
  if (focusable.length === 0) {
    event.preventDefault()
    container.focus({ preventScroll: true })
    return
  }

  const first = focusable[0]!
  const last = focusable.at(-1)!
  const activeElement = document.activeElement
  if (event.shiftKey && (activeElement === first || !container.contains(activeElement))) {
    event.preventDefault()
    last.focus({ preventScroll: true })
  } else if (!event.shiftKey && (activeElement === last || !container.contains(activeElement))) {
    event.preventDefault()
    first.focus({ preventScroll: true })
  }
}

function registerDialog(entry: DialogEntry): void {
  if (dialogStack.some(candidate => candidate.id === entry.id)) return
  entry.previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
  dialogStack.push(entry)
  if (dialogStack.length === 1) {
    document.addEventListener('keydown', handleDocumentKeydown)
  }
  void nextTick(() => focusDialog(entry))
}

function unregisterDialog(entry: DialogEntry): void {
  const index = dialogStack.findIndex(candidate => candidate.id === entry.id)
  if (index < 0) return
  const wasTopmost = index === dialogStack.length - 1
  dialogStack.splice(index, 1)
  if (dialogStack.length === 0) {
    document.removeEventListener('keydown', handleDocumentKeydown)
  }
  if (wasTopmost && entry.previousFocus?.isConnected) {
    entry.previousFocus.focus({ preventScroll: true })
  }
}

export function useDialogLifecycle(options: {
  open: Readonly<Ref<boolean>>
  container: Ref<HTMLElement | null>
  close: () => void
  closeOnEscape?: () => boolean
}) {
  const entry: DialogEntry = {
    id: Symbol('dialog'),
    container: options.container,
    close: options.close,
    closeOnEscape: options.closeOnEscape ?? (() => true),
    previousFocus: null,
  }

  watch(
    options.open,
    open => {
      if (open) registerDialog(entry)
      else unregisterDialog(entry)
    },
    { immediate: true, flush: 'post' },
  )

  onBeforeUnmount(() => unregisterDialog(entry))
}
