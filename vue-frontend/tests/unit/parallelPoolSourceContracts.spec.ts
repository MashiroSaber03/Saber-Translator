import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('parallel pool source contracts', () => {
  it('keeps pool adapters compact and free of scaffold narration', () => {
    const files = [
      'src/composables/translation/parallel/pools/index.ts',
      'src/composables/translation/parallel/pools/DetectionPool.ts',
      'src/composables/translation/parallel/pools/OcrPool.ts',
      'src/composables/translation/parallel/pools/ColorPool.ts',
      'src/composables/translation/parallel/pools/AutoGlossaryPool.ts',
      'src/composables/translation/parallel/pools/TranslatePool.ts',
      'src/composables/translation/parallel/pools/InpaintPool.ts',
      'src/composables/translation/parallel/pools/RenderPool.ts',
      'src/composables/translation/parallel/pools/SavePool.ts',
    ]

    for (const file of files) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).not.toContain('/**')
      expect(source, file).not.toContain('负责')
      expect(source, file).not.toContain('池子模块导出')
      expect(source, file).not.toContain('检测池')
      expect(source, file).not.toContain('OCR池')
      expect(source, file).not.toContain('颜色提取池')
      expect(source, file).not.toContain('自动术语提取池')
      expect(source, file).not.toContain('翻译池')
      expect(source, file).not.toContain('修复池')
      expect(source, file).not.toContain('渲染池')
      expect(source, file).not.toContain('  //')
    }
  })

  it('keeps pool adapters dependent on injected pipeline services instead of the composable entry', () => {
    const files = [
      'src/composables/translation/parallel/pools/DetectionPool.ts',
      'src/composables/translation/parallel/pools/OcrPool.ts',
      'src/composables/translation/parallel/pools/ColorPool.ts',
      'src/composables/translation/parallel/pools/AutoGlossaryPool.ts',
      'src/composables/translation/parallel/pools/TranslatePool.ts',
      'src/composables/translation/parallel/pools/InpaintPool.ts',
      'src/composables/translation/parallel/pools/RenderPool.ts',
      'src/composables/translation/parallel/pools/SavePool.ts',
    ]

    for (const file of files) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).not.toContain('useParallelTranslation')
      expect(source, file).not.toContain('../useParallelTranslation')
      expect(source, file).not.toContain('@/composables/translation/parallel/useParallelTranslation')
    }
  })
})
