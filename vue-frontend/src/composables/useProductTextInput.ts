import { computed, shallowRef } from 'vue'

export interface ProductTextInputOptions {
  title?: string
  message: string
  initialValue?: string
  placeholder?: string
  confirmText?: string
  cancelText?: string
}

interface NormalizedProductTextInputOptions {
  title: string
  message: string
  initialValue: string
  placeholder: string
  confirmText: string
  cancelText: string
}

interface ProductTextInputRequest {
  options: NormalizedProductTextInputOptions
  resolve: (value: string | null) => void
}

const activeRequest = shallowRef<ProductTextInputRequest | null>(null)
const pendingRequests: ProductTextInputRequest[] = []

function normalizeTextInputOptions(options: ProductTextInputOptions): NormalizedProductTextInputOptions {
  return {
    title: options.title ?? '输入内容',
    message: options.message,
    initialValue: options.initialValue ?? '',
    placeholder: options.placeholder ?? '',
    confirmText: options.confirmText ?? '确定',
    cancelText: options.cancelText ?? '取消',
  }
}

function showNextRequest() {
  if (activeRequest.value || pendingRequests.length === 0) return
  activeRequest.value = pendingRequests.shift() ?? null
}

function settleActiveRequest(value: string | null) {
  const request = activeRequest.value
  if (!request) return
  activeRequest.value = null
  request.resolve(value)
  showNextRequest()
}

export function requestProductTextInput(options: ProductTextInputOptions): Promise<string | null> {
  return new Promise(resolve => {
    pendingRequests.push({
      options: normalizeTextInputOptions(options),
      resolve,
    })
    showNextRequest()
  })
}

export function useProductTextInputState() {
  return {
    activeInput: computed(() => activeRequest.value?.options ?? null),
    submitActive: (value: string) => settleActiveRequest(value),
    cancelActive: () => settleActiveRequest(null),
  }
}
