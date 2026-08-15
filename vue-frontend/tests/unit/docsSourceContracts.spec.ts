import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('documentation source contracts', () => {
  it('keeps the current coding style guide aligned with product header and form primitives', () => {
    const codingStyle = source('CODING_STYLE.md')

    expect(codingStyle).toContain('ProductPageHeader')
    expect(codingStyle).toContain('ProductHeaderAction')
    expect(codingStyle).toContain('UiFormGrid')
    expect(codingStyle).toContain('UiSelect')
    expect(codingStyle).toContain('UiNumberField')
    expect(codingStyle).not.toContain('UiModalSection')
    expect(codingStyle).not.toContain('AppHeader')
    expect(codingStyle).toContain('全局 token 必须有生产源码中的真实消费者')
    expect(codingStyle).toContain('primitive 自身的 variant 只能设置私有 fallback 变量')
  })

  it('documents the current frontend ownership and verification contracts', () => {
    const architecture = source('docs/frontend-architecture.md')
    const readme = source('README.md')

    expect(readme).toContain('docs/frontend-architecture.md')
    expect(architecture).toContain('useAiModelDiscovery')
    expect(architecture).toContain('useCharacterStudioChat.ts')
    expect(architecture).toContain('npm run visual:test')
    expect(architecture).not.toContain('UiModalSection')
  })

})
