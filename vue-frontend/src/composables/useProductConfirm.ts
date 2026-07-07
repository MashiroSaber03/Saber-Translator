import { computed, shallowRef } from 'vue'

export type ProductConfirmTone = 'primary' | 'danger'

export interface ProductConfirmOptions {
  title?: string
  message: string
  confirmText?: string
  cancelText?: string
  tone?: ProductConfirmTone
}

export type ProductConfirmAction = (options: ProductConfirmOptions) => Promise<boolean>

interface NormalizedProductConfirmOptions {
  title: string
  message: string
  confirmText: string
  cancelText: string
  tone: ProductConfirmTone
}

interface ProductConfirmRequest {
  id: number
  options: NormalizedProductConfirmOptions
  resolve: (value: boolean) => void
}

let nextRequestId = 1
const activeRequest = shallowRef<ProductConfirmRequest | null>(null)
const pendingRequests: ProductConfirmRequest[] = []

function normalizeConfirmOptions(options: ProductConfirmOptions): NormalizedProductConfirmOptions {
  return {
    title: options.title ?? '确认操作',
    message: options.message,
    confirmText: options.confirmText ?? '确定',
    cancelText: options.cancelText ?? '取消',
    tone: options.tone ?? 'primary',
  }
}

function showNextRequest() {
  if (activeRequest.value || pendingRequests.length === 0) return
  activeRequest.value = pendingRequests.shift() ?? null
}

function settleActiveRequest(value: boolean) {
  const request = activeRequest.value
  if (!request) return
  activeRequest.value = null
  request.resolve(value)
  showNextRequest()
}

export function confirmProductAction(options: ProductConfirmOptions): Promise<boolean> {
  return new Promise(resolve => {
    pendingRequests.push({
      id: nextRequestId++,
      options: normalizeConfirmOptions(options),
      resolve,
    })
    showNextRequest()
  })
}

export function useProductConfirmState() {
  return {
    activeConfirm: computed(() => activeRequest.value?.options ?? null),
    confirmActive: () => settleActiveRequest(true),
    cancelActive: () => settleActiveRequest(false),
  }
}
