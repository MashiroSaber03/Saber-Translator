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

  it('keeps toast property coverage on the real service boundary', () => {
    const content = source('tests/property/toast.property.ts')

    expect(content).toContain("from '@/utils/toast'")
    expect(content).toContain('toastService.')
    expect(content).not.toContain('createToastManager')
    expect(content).not.toContain('createExtendedToastManager')
    expect(content).not.toContain('从服务中提取用于测试')
  })

  it('sanitizes HTML general messages and expires duration zero through the safety timeout', () => {
    const messageId = toastService.showGeneralMessage(
      '<strong>ok</strong><img src=x onerror=alert(1)>',
      'warning',
      true,
      0,
      'toast_contract',
    )

    expect(messageId).toBe('toast_contract')
    expect(toastService.toasts.value).toHaveLength(1)
    expect(toastService.toasts.value[0]?.isHTML).toBe(true)
    expect(toastService.toasts.value[0]?.message).toContain('<strong>ok</strong>')
    expect(toastService.toasts.value[0]?.message).not.toContain('onerror')

    vi.advanceTimersByTime(29999)
    expect(toastService.toasts.value).toHaveLength(1)

    vi.advanceTimersByTime(1)
    expect(toastService.toasts.value).toHaveLength(0)
  })

  it('clears timers when a toast is removed manually', () => {
    const id = toastService.addToast('pending', 'info', 1000)

    toastService.removeToast(id)
    vi.advanceTimersByTime(1000)

    expect(toastService.toasts.value).toHaveLength(0)
  })

  it('clears only matching toast types when requested', () => {
    toastService.addToast('keep', 'success', 0)
    toastService.addToast('drop', 'warning', 0)

    toastService.clearAllGeneralMessages('warning')

    expect(toastService.toasts.value).toHaveLength(1)
    expect(toastService.toasts.value[0]?.message).toBe('keep')
  })
})
