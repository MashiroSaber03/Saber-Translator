import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('edit keyboard shortcut source contracts', () => {
  it('keeps the property suite bound to the product shortcut composable', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/property/keyboard.property.ts'), 'utf8')

    expect(source).toContain("from '@/composables/edit/useEditWorkspaceKeyboardShortcuts'")

    for (const shadowContract of [
      'function isInInput' + 'Element',
      'function match' + 'Key',
      'function handleKeyboard' + 'Event',
      'function formatKey' + 'Combo',
      'interface Keyboard' + 'Handler',
      'MockKeyboard' + 'Event',
    ]) {
      expect(source).not.toContain(shadowContract)
    }

    for (const staleNarration of [
      '快捷键系统属性测试',
      '测试数据生成器',
      '辅助函数',
      '属性测试',
      '// ============================================================',
      '/' + '**',
      '验证',
      'return true',
      '模拟',
    ]) {
      expect(source).not.toContain(staleNarration)
    }
  })
})
