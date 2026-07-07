import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('prompt mode property source contract', () => {
  it('keeps prompt-mode properties as current settings-store behavior contracts', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/property/promptMode.property.ts'), 'utf8')

    for (const staleSource of [
      '/' + '**',
      '提示词模式切换' + '属性测试',
      '使用 fast-check' + ' 进行属性基测试',
      'localStorageMock',
      'Storage.prototype',
      '每次迭代重新创建 Pinia',
      '验证',
      'return ' + 'false',
      'return ' + '(',
    ]) {
      expect(source).not.toContain(staleSource)
    }

    expect(source).toContain('useSettingsStore')
    expect(source).toContain('setTranslatePromptMode')
    expect(source).toContain('setAiVisionOcrPromptMode')
    expect(source).toContain('expect(')
  })
})
