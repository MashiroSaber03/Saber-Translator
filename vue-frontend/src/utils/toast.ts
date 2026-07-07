import { ref, type Ref } from 'vue'
import { sanitizeHtml } from './sanitizeHtml'

export type ToastType = 'success' | 'error' | 'info' | 'warning'

export interface Toast {
  id: number
  messageId: string
  message: string
  type: ToastType
  isHTML: boolean
  timer?: ReturnType<typeof setTimeout>
}

export interface ToastService {
  toasts: Ref<Toast[]>
  addToast: (message: string, type?: ToastType, duration?: number) => number
  removeToast: (id: number) => void
  clearAll: () => void
  success: (message: string, duration?: number) => number
  error: (message: string, duration?: number) => number
  info: (message: string, duration?: number) => number
  warning: (message: string, duration?: number) => number
  showGeneralMessage: (message: string, type?: ToastType, isHTML?: boolean, duration?: number, messageId?: string) => string
  clearGeneralMessageById: (messageId: string) => void
  clearAllGeneralMessages: (type?: ToastType | '') => void
}

const DEFAULT_TOAST_DURATION = 3000
const DEFAULT_GENERAL_MESSAGE_DURATION = 5000
const SAFETY_TIMEOUT = 30000

const toasts = ref<Toast[]>([])
let toastId = 0

function createMessageId(): string {
  return `msg_${Date.now()}_${Math.floor(Math.random() * 1000)}`
}

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
    messageId: createMessageId(),
    message,
    type,
    isHTML: false,
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

const showGeneralMessage = (
  message: string,
  type: ToastType = 'info',
  isHTML: boolean = false,
  duration: number = DEFAULT_GENERAL_MESSAGE_DURATION,
  messageId: string = '',
): string => {
  clearAll()

  const toast: Toast = {
    id: ++toastId,
    messageId: messageId || createMessageId(),
    message: isHTML ? sanitizeHtml(message) : message,
    type,
    isHTML,
  }

  scheduleRemoval(toast, duration > 0 ? duration : SAFETY_TIMEOUT)
  toasts.value.push(toast)
  return toast.messageId
}

const clearGeneralMessageById = (messageId: string): void => {
  if (!messageId) return
  removeToastsWhere((toast) => toast.messageId === messageId)
}

const clearAllGeneralMessages = (type: ToastType | '' = ''): void => {
  if (type === '') {
    clearAll()
    return
  }
  removeToastsWhere((toast) => toast.type === type)
}

export const toastService: ToastService = {
  toasts,
  addToast,
  removeToast,
  clearAll,
  success,
  error,
  info,
  warning,
  showGeneralMessage,
  clearGeneralMessageById,
  clearAllGeneralMessages,
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

export { showGeneralMessage, clearGeneralMessageById, clearAllGeneralMessages }
