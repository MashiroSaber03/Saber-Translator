import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { createProgressManager } from '@/composables/translation/core/progressManager'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('createProgressManager', () => {
  it('keeps the progress manager source free of scaffold narration', () => {
    const content = source('src/composables/translation/core/progressManager.ts')

    for (const staleNarration of [
      '进度管理器',
      '提供统一的进度报告和管理功能',
      '创建进度管理器',
      '/**',
    ]) {
      expect(content).not.toContain(staleNarration)
    }

    for (const oldIndent of [
      '    progress:',
      '    const reporter',
    ]) {
      expect(content).not.toContain(oldIndent)
    }
  })

  it('clamps auto-computed progress percentages to the valid range', () => {
    const { progress, reporter } = createProgressManager()

    reporter.init(10)
    reporter.update(12, 'overflow')

    expect(progress.value.percentage).toBe(100)
    expect(progress.value.label).toBe('overflow')
  })

  it('returns immutable progress snapshots', () => {
    const { progress, reporter } = createProgressManager()

    reporter.init(3, 'start')
    const snapshot = reporter.getProgress()
    snapshot.current = 99

    expect(progress.value.current).toBe(0)
  })
})
