import { describe, expect, it } from 'vitest'

import {
  getPaddleOcrVlPrompt,
  inferPaddleOcrVlPromptLanguage,
} from '@/constants'

describe('PaddleOCR-VL prompt language projection', () => {
  it('restores the selector value from a persisted generated prompt', () => {
    expect(
      inferPaddleOcrVlPromptLanguage(getPaddleOcrVlPrompt('简体中文')),
    ).toBe('chinese')
    expect(
      inferPaddleOcrVlPromptLanguage(getPaddleOcrVlPrompt('英语')),
    ).toBe('english')
  })

  it('keeps the current selector for a custom prompt', () => {
    expect(
      inferPaddleOcrVlPromptLanguage('请识别图中文字并保留排版', 'korean'),
    ).toBe('korean')
  })
})
