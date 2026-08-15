import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const frontendRoot = resolve(__dirname, '..', '..')

describe('api source contracts', () => {
  it('keeps the plugin API module free of scaffold-style section narration', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/plugin.ts'), 'utf8')

    expect(source).not.toContain('// ====================')
    expect(source).not.toContain('包含插件列表')
    expect(source).not.toContain('@param')
    expect(source).not.toContain('获取插件列表')
  })

  it('keeps the web import API on backend-owned v2 drafts', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/v2/webImport.ts'), 'utf8')

    expect(source).toContain('/api/v2/web-import')
    expect(source).toContain('createWebImportDraft')
    expect(source).toContain('commitWebImportDraft')
    expect(source).not.toContain('/api/web-import')
    expect(source).not.toContain('ReadableStream')
    expect(source).not.toContain('dataUrl')
  })

  it('keeps the continuation API module free of scaffold-style narration', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/continuation.ts'), 'utf8')

    expect(source).not.toContain('// ====================')
    expect(source).not.toContain('漫画续写 API')
    expect(source).not.toContain('@param')
    expect(source).not.toContain('获取可用于参考图选择')
  })

  it('keeps browser translation APIs on the v2 job contract', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/v2/translation.ts'), 'utf8')

    expect(source).toContain('/api/v2/chapters/')
    expect(source).toContain('/translation-jobs')
    expect(source).not.toContain('/api/parallel/')
    expect(source).not.toContain('base64')
  })

  it('keeps diagnostics on the v2 backend contract', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/v2/diagnostics.ts'), 'utf8')

    expect(source).toContain('fetchV2ModelCatalog')
    expect(source).toContain('runV2ConnectionTest')
    expect(source).not.toContain('/api/config')
  })

  it('keeps plugin agent session paths behind a local endpoint helper', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/pluginAgent.ts'), 'utf8')

    expect(source).toContain('function pluginAgentSessionEndpoint')
    expect(source.match(/encodeURIComponent\(sessionId\)/g) ?? []).toHaveLength(1)
  })

  it('keeps character studio paths behind local endpoint helpers', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/characterStudio.ts'), 'utf8')

    expect(source).toContain("from '@/api/v2/studio'")
    expect(source).not.toContain('/api/manga-insight/')
    expect(source).not.toContain('/character-studio/')
  })

  it('keeps the insight API module compact and helper-driven', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/insight.ts'), 'utf8')

    expect(source).not.toContain('// ====================')
    expect(source).not.toContain('漫画分析 API')
    expect(source).not.toContain('@param')
    expect(source).toContain("from '@/api/v2/insight'")
    expect(source).not.toContain('/api/manga-insight/')
  })

  it('uses the shared OpenAI-compatible wire type in the insight API module', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/insight.ts'), 'utf8')

    expect(source).toContain('type OpenAICompatibleOptionsWire')
    expect(source).not.toContain('interface OpenAICompatibleWireOptions')
    expect(source).not.toContain('OpenAICompatibleWireOptions')
    expect(source).toContain('): OpenAICompatibleOptionsWire')
  })
})
