import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { toastService } from '@/utils/toast'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('toast service', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    toastService.clearAll()
  })

  afterEach(() => {
    toastService.clearAll()
    vi.useRealTimers()
  })

  it('keeps the production source compact and removal logic centralized', () => {
    const content = source('src/utils/toast.ts')

    for (const staleNarration of [
      '/**',
      '@param',
      '@returns',
      'Toast 消息服务',
      'Toast 消息类型',
      'Toast 消息接口',
      'Toast 服务接口',
      '清除 Toast 的定时器',
      '添加 Toast 消息',
      '移除指定 ID 的 Toast',
      '清除所有 Toast',
      '快捷方法',
      '显示队列式通用消息',
      '按ID清除消息',
      '清除所有特定类型的消息',
      '组合式函数',
      '便捷函数',
    ]) {
      expect(content).not.toContain(staleNarration)
    }

    expect(content).toContain('function removeToastsWhere')
    expect(content.match(/function removeToastsWhere/g)).toHaveLength(1)
  })

  it('clears timers when a toast is removed manually', () => {
    const id = toastService.addToast('pending', 'info', 1000)

    toastService.removeToast(id)
    vi.advanceTimersByTime(1000)

    expect(toastService.toasts.value).toHaveLength(0)
  })

})
