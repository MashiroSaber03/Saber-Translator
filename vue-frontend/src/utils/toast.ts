/**
 * Toast 消息服务
 * 提供全局的消息提示功能
 * 兼容原版 showGeneralMessage 函数的所有功能
 */

import { ref, type Ref } from 'vue'

/**
 * Toast 消息类型
 */
export type ToastType = 'success' | 'error' | 'info' | 'warning'

/**
 * Toast 消息接口
 */
export interface Toast {
  /** 唯一标识（数字ID） */
  id: number
  /** 字符串ID（用于按ID清除） */
  messageId: string
  /** 消息内容 */
  message: string
  /** 消息类型 */
  type: ToastType
  /** 是否为HTML内容 */
  isHTML: boolean
  /** 自动关闭定时器 */
  timer?: ReturnType<typeof setTimeout>
  /** 安全超时定时器（防止消息永久显示） */
  safetyTimer?: ReturnType<typeof setTimeout>
}

/**
 * Toast 服务接口
 */
export interface ToastService {
  /** 消息队列 */
  toasts: Ref<Toast[]>
  /** 添加消息 */
  addToast: (message: string, type?: ToastType, duration?: number) => number
  /** 移除消息 */
  removeToast: (id: number) => void
  /** 清除所有消息 */
  clearAll: () => void
  /** 显示成功消息 */
  success: (message: string, duration?: number) => number
  /** 显示错误消息 */
  error: (message: string, duration?: number) => number
  /** 显示信息消息 */
  info: (message: string, duration?: number) => number
  /** 显示警告消息 */
  warning: (message: string, duration?: number) => number
  /** 显示通用消息（兼容原版API） */
  showGeneralMessage: (message: string, type?: ToastType, isHTML?: boolean, duration?: number, messageId?: string) => string
  /** 按ID清除消息 */
  clearGeneralMessageById: (messageId: string) => void
  /** 清除所有特定类型的消息 */
  clearAllGeneralMessages: (type?: ToastType | '') => void
}

// 消息队列
const toasts = ref<Toast[]>([])

// 消息 ID 计数器
let toastId = 0

// 安全超时时间（毫秒），即使是无限消息也会在此时间后自动消失
const SAFETY_TIMEOUT = 30000

/**
 * 清除 Toast 的所有定时器
 * @param toast - Toast 对象
 */
const clearToastTimers = (toast: Toast): void => {
  if (toast.timer) {
    clearTimeout(toast.timer)
    toast.timer = undefined
  }
  if (toast.safetyTimer) {
    clearTimeout(toast.safetyTimer)
    toast.safetyTimer = undefined
  }
}

/**
 * 添加 Toast 消息
 * @param message - 消息内容
 * @param type - 消息类型
 * @param duration - 显示时长（毫秒），0 表示不自动关闭
 * @returns Toast ID
 */
const addToast = (message: string, type: ToastType = 'info', duration: number = 3000): number => {
  const id = ++toastId
  const messageId = `msg_${Date.now()}_${Math.floor(Math.random() * 1000)}`
  const toast: Toast = { id, messageId, message, type, isHTML: false }

  // 如果设置了持续时间，添加自动关闭定时器
  if (duration > 0) {
    toast.timer = setTimeout(() => {
      removeToast(id)
    }, duration)
  } else {
    // 即使是无限消息，也添加安全超时
    toast.safetyTimer = setTimeout(() => {
      removeToast(id)
    }, SAFETY_TIMEOUT)
  }

  toasts.value.push(toast)
  return id
}

/**
 * 移除指定 ID 的 Toast
 * @param id - Toast ID
 */
const removeToast = (id: number): void => {
  const index = toasts.value.findIndex((t) => t.id === id)
  if (index !== -1) {
    const toast = toasts.value[index]
    // 清除定时器
    if (toast) {
      clearToastTimers(toast)
    }
    toasts.value.splice(index, 1)
  }
}

/**
 * 清除所有 Toast
 */
const clearAll = (): void => {
  // 清除所有定时器
  toasts.value.forEach((toast) => {
    clearToastTimers(toast)
  })
  toasts.value = []
}

/**
 * 快捷方法：显示成功消息
 */
const success = (message: string, duration?: number): number => {
  return addToast(message, 'success', duration)
}

/**
 * 快捷方法：显示错误消息
 */
const error = (message: string, duration?: number): number => {
  return addToast(message, 'error', duration)
}

/**
 * 快捷方法：显示信息消息
 */
const info = (message: string, duration?: number): number => {
  return addToast(message, 'info', duration)
}

/**
 * 快捷方法：显示警告消息
 */
const warning = (message: string, duration?: number): number => {
  return addToast(message, 'warning', duration)
}

/**
 * 显示通用消息（兼容原版 showGeneralMessage API）
 * 队列模式：立即移除所有现有消息，只显示最新的一个
 * @param message - 消息内容（可以是 HTML）
 * @param type - 消息类型
 * @param isHTML - 消息内容是否为 HTML
 * @param duration - 自动消失时间（毫秒），0 表示不自动消失
 * @param messageId - 消息唯一标识符，用于后续清除特定消息
 * @returns 消息ID
 */
const showGeneralMessage = (
  message: string,
  type: ToastType = 'info',
  isHTML: boolean = false,
  duration: number = 5000,
  messageId: string = ''
): string => {
  // 生成唯一消息ID或使用提供的ID
  const msgId = messageId || `msg_${Date.now()}_${Math.floor(Math.random() * 1000)}`
  
  // 队列模式：立即移除所有现有消息，只显示最新的一个
  clearAll()
  
  const id = ++toastId
  const toast: Toast = { id, messageId: msgId, message, type, isHTML }
  
  // 计算实际超时时间
  const actualDuration = duration > 0 ? duration : SAFETY_TIMEOUT
  
  // 设置定时消失
  toast.timer = setTimeout(() => {
    removeToast(id)
  }, actualDuration)
  
  toasts.value.push(toast)
  return msgId
}

/**
 * 按ID清除消息
 * @param messageId - 要清除的消息ID
 */
const clearGeneralMessageById = (messageId: string): void => {
  if (!messageId) return
  
  const index = toasts.value.findIndex((t) => t.messageId === messageId)
  if (index !== -1) {
    const toast = toasts.value[index]
    if (toast) {
      clearToastTimers(toast)
    }
    toasts.value.splice(index, 1)
  }
}

/**
 * 清除所有特定类型的消息
 * @param type - 消息类型，空字符串表示清除所有类型
 */
const clearAllGeneralMessages = (type: ToastType | '' = ''): void => {
  if (type === '') {
    clearAll()
  } else {
    // 找出要删除的消息索引（从后往前删除以避免索引问题）
    const indicesToRemove: number[] = []
    toasts.value.forEach((toast, index) => {
      if (toast.type === type) {
        clearToastTimers(toast)
        indicesToRemove.push(index)
      }
    })
    // 从后往前删除
    for (let i = indicesToRemove.length - 1; i >= 0; i--) {
      const idx = indicesToRemove[i]
      if (idx !== undefined) {
        toasts.value.splice(idx, 1)
      }
    }
  }
}

/**
 * Toast 服务实例
 */
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
  clearAllGeneralMessages
}

/**
 * 组合式函数：使用 Toast 服务
 * @returns Toast 服务实例
 */
export function useToast(): ToastService {
  return toastService
}

/**
 * 便捷函数：显示 Toast 消息
 * 兼容旧版 API，方便直接调用
 * @param message - 消息内容
 * @param type - 消息类型
 * @param duration - 显示时长（毫秒）
 * @returns Toast ID
 */
export function showToast(message: string, type: ToastType = 'info', duration: number = 3000): number {
  return addToast(message, type, duration)
}

// 导出便捷函数，兼容原版 API
export { showGeneralMessage, clearGeneralMessageById, clearAllGeneralMessages }
