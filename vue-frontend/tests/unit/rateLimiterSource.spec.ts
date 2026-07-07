import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('rate limiter source contracts', () => {
  it('keeps the property suite concise and bound to the production utility', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/property/rateLimiter.property.ts'), 'utf8')

    expect(source).toContain("from '@/utils/rateLimiter'")
    for (const staleNarration of [
      'RPM 限速器属性测试',
      '测试数据生成器',
      '生成有效的 RPM',
      '生成正整数 RPM',
      '// ============================================================',
      '/' + '**',
      '验证',
      'return true',
      '模拟失败',
      'expectedInterval = Math.ceil',
    ]) {
      expect(source).not.toContain(staleNarration)
    }
  })
})
