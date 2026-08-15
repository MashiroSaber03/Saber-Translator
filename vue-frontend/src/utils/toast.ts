import { ref, type Ref } from 'vue'

export type ToastType = 'success' | 'error' | 'info' | 'warning'

interface Toast {
  id: number
  message: string
  type: ToastType
  timer?: ReturnType<typeof setTimeout>
}

interface ToastService {
  toasts: Ref<Toast[]>
  addToast: (message: string, type?: ToastType, duration?: number) => number
  removeToast: (id: number) => void
  clearAll: () => void
  success: (message: string, duration?: number) => number
  error: (message: string, duration?: number) => number
  info: (message: string, duration?: number) => number
  warning: (message: string, duration?: number) => number
}

const DEFAULT_TOAST_DURATION = 3000

const toasts = ref<Toast[]>([])
let toastId = 0

function clearToastTimer(toast: Toast): void {
  if (!toast.timer) return
  clearTimeout(toast.timer)
  toast.timer = undefined
}

function scheduleRemoval(toast: Toast, duration: number): void {
  if (duration <= 0) return
  toast.timer = setTimeout(() => {
    removeToast(toast.id)
  }, duration)
}

function removeToastsWhere(predicate: (toast: Toast) => boolean): void {
  for (let index = toasts.value.length - 1; index >= 0; index -= 1) {
    const toast = toasts.value[index]
    if (toast && predicate(toast)) {
      clearToastTimer(toast)
      toasts.value.splice(index, 1)
    }
  }
}

const addToast = (
  message: string,
  type: ToastType = 'info',
  duration: number = DEFAULT_TOAST_DURATION,
): number => {
  const toast: Toast = {
    id: ++toastId,
    message,
    type,
  }

  scheduleRemoval(toast, duration)
  toasts.value.push(toast)
  return toast.id
}

const removeToast = (id: number): void => {
  removeToastsWhere((toast) => toast.id === id)
}

const clearAll = (): void => {
  removeToastsWhere(() => true)
}

const success = (message: string, duration?: number): number => addToast(message, 'success', duration)

const error = (message: string, duration?: number): number => addToast(message, 'error', duration)

const info = (message: string, duration?: number): number => addToast(message, 'info', duration)

const warning = (message: string, duration?: number): number => addToast(message, 'warning', duration)

export const toastService: ToastService = {
  toasts,
  addToast,
  removeToast,
  clearAll,
  success,
  error,
  info,
  warning,
}

export function useToast(): ToastService {
  return toastService
}

export function showToast(
  message: string,
  type: ToastType = 'info',
  duration: number = DEFAULT_TOAST_DURATION,
): number {
  return addToast(message, type, duration)
}
