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

  it('keeps the web import API module free of scaffold-style narration', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/webImport.ts'), 'utf8')

    expect(source).not.toContain('网页导入 API')
    expect(source).not.toContain('获取代理图片 URL')
    expect(source).not.toContain('提取漫画图片')
    expect(source).not.toContain('测试 AI Agent 连接')
  })

  it('keeps WebImport raw fetch usage scoped to streaming extraction', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/webImport.ts'), 'utf8')

    expect(source).toContain('function webImportEndpoint')
    expect(source.match(/await fetch\(/g) ?? []).toHaveLength(1)
    expect(source).not.toContain('response.json()')
  })

  it('keeps the continuation API module free of scaffold-style narration', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/continuation.ts'), 'utf8')

    expect(source).not.toContain('// ====================')
    expect(source).not.toContain('漫画续写 API')
    expect(source).not.toContain('@param')
    expect(source).not.toContain('获取可用于参考图选择')
  })

  it('keeps the parallel translate API module free of scaffold-style narration', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/parallelTranslate.ts'), 'utf8')

    expect(source).not.toContain('// ====================')
    expect(source).not.toContain('并行翻译 API')
    expect(source).not.toContain('为并行流水线提供')
  })

  it('keeps the translate API module free of scaffold-style narration', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/translate.ts'), 'utf8')

    expect(source).not.toContain('// ====================')
    expect(source).not.toContain('包含图片翻译')
    expect(source).not.toContain('@param')
    expect(source).not.toContain('当前后端协议字段')
  })

  it('keeps the config API module free of scaffold-style narration', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/config.ts'), 'utf8')

    expect(source).not.toContain('// ====================')
    expect(source).not.toContain('配置 API')
    expect(source).not.toContain('@param')
    expect(source).not.toContain('当前后端协议字段')
  })

  it('keeps plugin agent session paths behind a local endpoint helper', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/pluginAgent.ts'), 'utf8')

    expect(source).toContain('function pluginAgentSessionEndpoint')
    expect(source.match(/encodeURIComponent\(sessionId\)/g) ?? []).toHaveLength(1)
  })

  it('keeps character studio paths behind local endpoint helpers', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/characterStudio.ts'), 'utf8')

    expect(source).toContain('function characterStudioEndpoint')
    expect(source).toContain('function characterStudioDocumentEndpoint')
    expect(source).toContain('function characterStudioQuery')
    expect(source.match(/encodeURIComponent\(/g) ?? []).toHaveLength(1)
  })

  it('keeps the insight API module compact and helper-driven', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/insight.ts'), 'utf8')

    expect(source).not.toContain('// ====================')
    expect(source).not.toContain('漫画分析 API')
    expect(source).not.toContain('@param')
    expect(source).toContain('function insightEndpoint')
    expect(source).toContain('function insightBookEndpoint')
    expect(source).toContain('function insightQuery')
    expect(source.match(/encodeURIComponent\(/g) ?? []).toHaveLength(1)
  })

  it('uses the shared OpenAI-compatible wire type in the insight API module', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/api/insight.ts'), 'utf8')

    expect(source).toContain("import type { OpenAICompatibleOptionsWire } from '@/utils/openaiOptions'")
    expect(source).not.toContain('interface OpenAICompatibleWireOptions')
    expect(source).not.toContain('OpenAICompatibleWireOptions')
    expect(source).toContain('openai_options?: OpenAICompatibleOptionsWire')
  })
})
