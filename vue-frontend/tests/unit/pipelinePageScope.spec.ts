import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import {
  resolvePipelineImageSelection,
  resolvePipelinePageIndexes,
} from '@/composables/translation/core/pageScope'
import type { ImageData } from '@/types/image'

function createImage(fileName: string): ImageData {
  return {
    id: fileName,
    fileName,
    originalDataURL: `data:image/png;base64,${fileName}`,
    translatedDataURL: null,
    cleanImageData: null,
    bubbleStates: null,
    translationStatus: 'pending',
    translationFailed: false,
    fontSize: 16,
    autoFontSize: false,
    fontFamily: 'fonts/STSONG.TTF',
    layoutDirection: 'auto',
    textColor: '#000000',
    fillColor: '#ffffff',
    inpaintMethod: 'solid',
    strokeEnabled: false,
    strokeColor: '#000000',
    strokeWidth: 1,
    hasUnsavedChanges: false,
  }
}

describe('pipeline page scope helpers', () => {
  it('resolves pipeline page indexes for every supported scope', () => {
    expect(resolvePipelinePageIndexes({ mode: 'standard', scope: 'current' }, 4, 2, [1])).toEqual([2])
    expect(resolvePipelinePageIndexes({ mode: 'standard', scope: 'current' }, 4, 9, [1])).toEqual([])
    expect(resolvePipelinePageIndexes({ mode: 'standard', scope: 'failed' }, 4, 0, [-1, 1, 4, 2])).toEqual([1, 2])
    expect(resolvePipelinePageIndexes({
      mode: 'standard',
      scope: 'selection',
      pageSelection: { pages: [1, 3, 99] },
    }, 4, 0, [])).toEqual([0, 2])
    expect(resolvePipelinePageIndexes({ mode: 'standard', scope: 'all' }, 3, 0, [])).toEqual([0, 1, 2])
    expect(resolvePipelinePageIndexes({ mode: 'standard', scope: 'all' }, 0, 0, [])).toEqual([])
  })

  it('projects page indexes to existing images without leaking sparse entries', () => {
    const images = [createImage('page-1.png'), undefined as unknown as ImageData, createImage('page-3.png')]

    expect(resolvePipelineImageSelection({
      mode: 'standard',
      scope: 'selection',
      pageSelection: { pages: [1, 2, 3] },
    }, images, 0, [])).toEqual([
      { image: images[0], index: 0 },
      { image: images[2], index: 2 },
    ])
  })

  it('keeps pipeline owners on the shared page-scope helper and free of scaffold narration', () => {
    const pipeline = readFileSync(resolve(process.cwd(), 'src/composables/translation/core/pipeline.ts'), 'utf8')
    const sequential = readFileSync(
      resolve(process.cwd(), 'src/composables/translation/core/SequentialPipeline.ts'),
      'utf8',
    )

    expect(pipeline).toContain("from './pageScope'")
    expect(sequential).toContain("from './pageScope'")

    for (const source of [pipeline, sequential]) {
      for (const staleNarration of [
        '/**',
        '当前执行契约',
        '核心设计理念',
        '使用示例',
        '执行翻译管线',
        '执行并行模式',
        '取消当前操作',
        '导出类型',
      ]) {
        expect(source).not.toContain(staleNarration)
      }
    }
  })
})
