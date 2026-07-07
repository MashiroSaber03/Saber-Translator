import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('touch property source contracts', () => {
  it('keeps touch properties focused on exported composable helpers', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/property/touch.property.ts'), 'utf8')

    expect(source).toContain("from '@/composables/useTouch'")
    for (const shadowHelper of [
      'function calculate' + 'Distance',
      'function manual' + 'DetectSwipe',
    ]) {
      expect(source).not.toContain(shadowHelper)
    }

    for (const staleNarration of [
      '触摸手势处理属性测试',
      '测试内容',
      '辅助函数',
      '手动实现滑动方向检测',
      '属性测试',
      '距离计算',
      '// ============================================================',
      '/' + '**',
      '验证',
      'return true',
    ]) {
      expect(source).not.toContain(staleNarration)
    }
  })
})
