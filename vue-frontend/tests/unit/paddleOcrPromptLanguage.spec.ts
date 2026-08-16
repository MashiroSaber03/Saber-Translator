import { describe, expect, it } from 'vitest'

import {
  getPaddleOcrVlPrompt,
  inferPaddleOcrVlPromptLanguage,
  isPaddleOcrVlLanguage,
  PADDLEOCR_VL_LANG_MAP,
} from '@/constants'
import { createDefaultSettings } from '@/stores/settings/defaults'
import { parseCurrentSettings } from '@/stores/settings/schema'
import { paddleOcrVlSourceLanguageGroups } from '@/components/settings/ocrSettingsOptions'

describe('PaddleOCR-VL prompt language projection', () => {
  it('builds the language-aware prompt for every selectable language', () => {
    for (const [language, displayName] of Object.entries(PADDLEOCR_VL_LANG_MAP)) {
      expect(isPaddleOcrVlLanguage(language)).toBe(true)
      expect(getPaddleOcrVlPrompt(language as keyof typeof PADDLEOCR_VL_LANG_MAP))
        .toBe(`对图中的${displayName}进行OCR:`)
    }

    const selectableLanguages = paddleOcrVlSourceLanguageGroups.flatMap(
      group => group.options.map(option => option.value),
    )
    expect(new Set(selectableLanguages)).toEqual(new Set(Object.keys(PADDLEOCR_VL_LANG_MAP)))
  })

  it('restores the selector value from a persisted generated prompt', () => {
    expect(inferPaddleOcrVlPromptLanguage(getPaddleOcrVlPrompt('chinese')))
      .toBe('chinese')
    expect(inferPaddleOcrVlPromptLanguage(getPaddleOcrVlPrompt('english')))
      .toBe('english')
  })

  it('keeps the current selector for a custom prompt', () => {
    expect(
      inferPaddleOcrVlPromptLanguage('请识别图中文字并保留排版', 'korean'),
    ).toBe('korean')
  })

  it('persists Japanese as the default PaddleOCR-VL source language', () => {
    expect(createDefaultSettings().paddleOcrVl).toEqual({ sourceLanguage: 'japanese' })
  })

  it('rejects unsupported PaddleOCR-VL languages at the settings boundary', () => {
    const settings = createDefaultSettings() as unknown as Record<string, unknown>
    settings.paddleOcrVl = { sourceLanguage: 'unsupported' }

    expect(parseCurrentSettings(settings)).toBeNull()
  })
})
