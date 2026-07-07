import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const stepFiles = [
  'src/composables/translation/core/steps/detection.ts',
  'src/composables/translation/core/steps/ocr.ts',
  'src/composables/translation/core/steps/color.ts',
  'src/composables/translation/core/steps/inpaint.ts',
  'src/composables/translation/core/steps/render.ts',
  'src/composables/translation/core/steps/aiTranslate.ts',
  'src/composables/translation/core/steps/translate.ts',
]

describe('translation step source contracts', () => {
  it('keeps translation public barrels and core types free of scaffold narration', () => {
    const publicBarrel = readFileSync(resolve(process.cwd(), 'src/composables/translation/index.ts'), 'utf8')
    const coreBarrel = readFileSync(resolve(process.cwd(), 'src/composables/translation/core/index.ts'), 'utf8')
    const coreTypes = readFileSync(resolve(process.cwd(), 'src/composables/translation/core/types.ts'), 'utf8')
    const exportAll = (modulePath: string) => `export * from '${modulePath}'`
    const exportNames = (names: string, modulePath: string) => `export { ${names} } from '${modulePath}'`

    expect(publicBarrel.trim()).toBe([
      exportAll('./core'),
      exportAll('./modes'),
    ].join('\n\n'))

    expect(coreBarrel.trim()).toBe([
      exportAll('./types'),
      exportNames('usePipeline', './pipeline'),
      exportNames('useSequentialPipeline, STEP_CHAIN_CONFIGS', './SequentialPipeline'),
    ].join('\n\n'))

    for (const staleNarration of [
      '翻译功能模块索引',
      '模块结构',
      '使用方式',
      '核心模块索引',
      '翻译管线核心类型定义',
      '// ============================================================',
      '翻译模式与范围',
      '进度管理',
      '批量处理选项',
      '管线配置',
      '管线执行结果',
      '保存的样式设置',
      '/** 翻译模式 */',
      '/** 执行范围 */',
      '/** 批量处理选项 */',
      '/** 管线配置 */',
      '/** 管线执行结果 */',
      '/** 保存的文本样式设置 */',
    ]) {
      expect(coreTypes).not.toContain(staleNarration)
    }

    for (const oldTopLevelIndent of [
      '    pages:',
      '    current:',
      '    batchSize:',
      '    mode:',
      '    success:',
      '    fontFamily:',
    ]) {
      expect(coreTypes).not.toContain(oldTopLevelIndent)
    }
  })

  it('keeps atomic translation steps free of scaffold-style section narration', () => {
    for (const file of stepFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).not.toContain('/**')
      expect(source, file).not.toContain('// ============================================================')
      expect(source, file).not.toContain('类型定义')
      expect(source, file).not.toContain('主函数')
      expect(source, file).not.toContain('辅助函数')
      expect(source, file).not.toContain('逐气泡翻译模式')
      expect(source, file).not.toContain('整页批量翻译模式')
      expect(source, file).not.toContain('检测步骤')
      expect(source, file).not.toContain('OCR 步骤')
      expect(source, file).not.toContain('颜色提取步骤')
      expect(source, file).not.toContain('修复步骤')
      expect(source, file).not.toContain('渲染步骤')
      expect(source, file).not.toContain('负责')
      expect(source, file).not.toContain('步骤1')
      expect(source, file).not.toContain('步骤2')
      expect(source, file).not.toContain('固定使用 Default')
      expect(source, file).not.toContain('返回生成的精确掩膜')
      expect(source, file).not.toContain('文字检测掩膜')
      expect(source, file).not.toContain('用户笔刷掩膜')
      expect(source, file).not.toContain('已解析的 results')
      expect(source, file).not.toContain('尝试从 content 解析')
    }
  })
})
