import { describe, expect, it } from 'vitest'

import {
  getBaseStepChain,
  getStepLabel,
  resolveParallelPoolChain,
  resolveSequentialStepChain,
} from '@/composables/translation/core/pipelineRegistry'

describe('pipelineRegistry', () => {
  it('returns the current base step chains for all translation modes', () => {
    expect(getBaseStepChain('standard')).toEqual([
      'detection',
      'ocr',
      'color',
      'translate',
      'inpaint',
      'render',
    ])

    expect(getBaseStepChain('hq')).toEqual([
      'detection',
      'ocr',
      'color',
      'aiTranslate',
      'inpaint',
      'render',
    ])

    expect(getBaseStepChain('proofread')).toEqual([
      'aiTranslate',
      'render',
    ])

    expect(getBaseStepChain('removeText')).toEqual([
      'detection',
      'inpaint',
      'render',
    ])
  })

  it('inserts OCR after detection for removeText sequential mode when the flag is enabled', () => {
    expect(
      resolveSequentialStepChain('removeText', {
        removeTextWithOcr: true,
        autoSaveEnabled: false,
      }),
    ).toEqual([
      'detection',
      'ocr',
      'inpaint',
      'render',
    ])
  })

  it('appends save to the sequential chain when auto-save is enabled', () => {
    expect(
      resolveSequentialStepChain('standard', {
        removeTextWithOcr: false,
        autoSaveEnabled: true,
      }),
    ).toEqual([
      'detection',
      'ocr',
      'color',
      'translate',
      'inpaint',
      'render',
      'save',
    ])
  })

  it('maps aiTranslate to the translate pool while preserving the rest of the parallel chain', () => {
    expect(
      resolveParallelPoolChain('hq', {
        removeTextWithOcr: false,
        autoSaveEnabled: false,
      }),
    ).toEqual([
      'detection',
      'ocr',
      'color',
      'translate',
      'inpaint',
      'render',
    ])

    expect(
      resolveParallelPoolChain('proofread', {
        removeTextWithOcr: false,
        autoSaveEnabled: false,
      }),
    ).toEqual([
      'translate',
      'render',
    ])
  })

  it('applies both removeText OCR insertion and save appending to the parallel pool chain', () => {
    expect(
      resolveParallelPoolChain('removeText', {
        removeTextWithOcr: true,
        autoSaveEnabled: true,
      }),
    ).toEqual([
      'detection',
      'ocr',
      'inpaint',
      'render',
      'save',
    ])
  })

  it('provides the unified step labels used by the pipelines', () => {
    expect(getStepLabel('detection')).toBe('气泡检测')
    expect(getStepLabel('aiTranslate')).toBe('AI翻译')
    expect(getStepLabel('save')).toBe('保存')
  })
})
