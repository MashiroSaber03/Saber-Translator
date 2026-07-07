import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('constants source contracts', () => {
  it('keeps the public constants entry as a domain barrel', () => {
    const constantsBarrel = source('src/constants/index.ts')
    const exportPrefix = 'export * ' + 'from '
    const expectedLines = [
      './prompts',
      './rateLimits',
      './edit',
      './storage',
      './ocr',
      './webImport',
      './routes',
      './bookshelf',
    ].flatMap(modulePath => [`${exportPrefix}'${modulePath}'`, ''])
    expectedLines.pop()

    expect(constantsBarrel.trim().split(/\r?\n/)).toEqual(expectedLines)
  })

  it('keeps constants domain owners free of scaffold narration', () => {
    const files = [
      'src/constants/prompts.ts',
      'src/constants/rateLimits.ts',
      'src/constants/edit.ts',
      'src/constants/storage.ts',
      'src/constants/ocr.ts',
      'src/constants/webImport.ts',
      'src/constants/routes.ts',
      'src/constants/bookshelf.ts',
    ]

    for (const file of files) {
      expect(existsSync(resolve(process.cwd(), file)), file).toBe(true)
      const content = source(file)

      for (const staleNarration of [
        '/**',
        '// ============================================================',
        '常量定义文件',
        '与后端 constants.js 保持一致',
        '@param',
        '默认提示词常量',
        'RPM 默认值常量',
        '自定义服务商 ID 常量',
        '编辑模式字号预设常量',
        'localStorage 存储键常量',
        'OCR 引擎常量',
        '网页导入常量',
        '书架常量',
      ]) {
        expect(content, file).not.toContain(staleNarration)
      }
    }
  })
})
