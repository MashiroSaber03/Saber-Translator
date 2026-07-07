import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('settings store property source contract', () => {
  it('keeps settings properties on current explicit assertion style', () => {
    const content = source('tests/property/settingsStore.property.ts')

    expect(content).toContain('function installLocalStorageMock()')
    expect(content).toContain('function createSettingsStore()')
    expect(content).not.toContain('beforeEach')
    expect(content).not.toContain('/**')
    expect(content).not.toMatch(/\/\/\s*(模拟|重置|生成|更新|验证|记录|创建|修改)/)
    expect(content).not.toMatch(/return\s+false/)
    expect(content).not.toMatch(/return\s*\(/)
    expect(content).not.toContain('设置状态管理属性测试')
  })
})
