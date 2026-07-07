import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const { parallelTranslateMock, translateSingleTextMock } = vi.hoisted(() => ({
  parallelTranslateMock: vi.fn(),
  translateSingleTextMock: vi.fn(),
}))

vi.mock('@/api/parallelTranslate', () => ({
  parallelTranslate: parallelTranslateMock,
}))

vi.mock('@/api/translate', () => ({
  translateSingleText: translateSingleTextMock,
}))

import { executeTranslate } from '@/composables/translation/core/steps/translate'
import { useSettingsStore } from '@/stores/settings'
import { createEmptyBookTranslationConstraints } from '@/utils/bookTranslationConstraints'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'
import type { GlossaryEntry, NonTranslateEntry } from '@/types/translationConstraints'

function createGlossaryEntry(note = ''): GlossaryEntry {
  return {
    source: 'Alice',
    target: '爱丽丝',
    note,
    matchMode: 'text',
  }
}

function createNonTranslateEntry(note = ''): NonTranslateEntry {
  return {
    pattern: '<keep>',
    note,
    matchMode: 'text',
  }
}

function createEnabledConstraints(options: {
  glossaryNote?: string
  nonTranslateNote?: string
  includeNonTranslate?: boolean
} = {}): BookTranslationConstraints {
  const constraints = createEmptyBookTranslationConstraints()
  constraints.glossary.enabled = true
  constraints.glossary.entries = [createGlossaryEntry(options.glossaryNote)]

  if (options.includeNonTranslate ?? true) {
    constraints.non_translate.enabled = true
    constraints.non_translate.entries = [createNonTranslateEntry(options.nonTranslateNote)]
  }

  return constraints
}

describe('executeTranslate', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    parallelTranslateMock.mockReset()
    translateSingleTextMock.mockReset()
  })

  it('keeps translation constraint fixtures typed to the current schema', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/translateStep.spec.ts'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('forwards glossary and non-translate settings in batch mode and returns warnings', async () => {
    const settingsStore = useSettingsStore()
    settingsStore.settings.translation.translationMode = 'batch'
    const constraints = createEnabledConstraints({
      glossaryNote: '主角',
      nonTranslateNote: '占位符',
    })

    parallelTranslateMock.mockResolvedValue({
      success: true,
      translated_texts: ['爱丽丝 <keep>'],
      textbox_texts: [],
      warnings: [{ imageIndex: 0, bubbleIndex: 0, source: 'Alice', expectedTarget: '爱丽丝', actualTranslation: '爱丽丝 <keep>' }],
    })

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    let result
    try {
      result = await executeTranslate({
        imageIndex: 0,
        originalTexts: ['Alice <keep>'],
        settingsSnapshot: settingsStore.settings,
        bookTranslationConstraints: constraints,
        isBookshelfMode: true,
      })
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }

    expect(parallelTranslateMock).toHaveBeenCalledWith(
      expect.objectContaining({
        glossary_settings: constraints.glossary,
        non_translate_settings: constraints.non_translate,
      }),
    )
    expect(result.warnings).toEqual([
      { imageIndex: 0, bubbleIndex: 0, source: 'Alice', expectedTarget: '爱丽丝', actualTranslation: '爱丽丝 <keep>' },
    ])
  })

  it('forwards glossary and non-translate settings in single mode', async () => {
    const settingsStore = useSettingsStore()
    settingsStore.settings.translation.translationMode = 'single'
    const constraints = createEnabledConstraints()

    translateSingleTextMock.mockResolvedValue({
      success: true,
      data: {
        translated_text: '爱丽丝 <keep>',
        warnings: [],
      },
    })

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    try {
      await executeTranslate({
        imageIndex: 0,
        originalTexts: ['Alice <keep>'],
        settingsSnapshot: settingsStore.settings,
        bookTranslationConstraints: constraints,
        isBookshelfMode: true,
      })
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }

    expect(translateSingleTextMock).toHaveBeenCalledWith(
      expect.objectContaining({
        glossary_settings: constraints.glossary,
        non_translate_settings: constraints.non_translate,
      }),
    )
  })

  it('forwards glossary and non-translate settings to textbox prompt requests in single mode', async () => {
    const settingsStore = useSettingsStore()
    settingsStore.settings.translation.translationMode = 'single'
    settingsStore.settings.useTextboxPrompt = true
    settingsStore.settings.textboxPrompt = 'textbox prompt'
    const constraints = createEnabledConstraints()

    translateSingleTextMock
      .mockResolvedValueOnce({
        success: true,
        data: {
          translated_text: '爱丽丝 <keep>',
          warnings: [],
        },
      })
      .mockResolvedValueOnce({
        success: true,
        data: {
          translated_text: 'textbox 爱丽丝 <keep>',
          warnings: [],
        },
      })

    await executeTranslate({
      imageIndex: 0,
      originalTexts: ['Alice <keep>'],
      settingsSnapshot: settingsStore.settings,
      bookTranslationConstraints: constraints,
      isBookshelfMode: true,
    })

    expect(translateSingleTextMock).toHaveBeenCalledTimes(2)
    expect(translateSingleTextMock).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        prompt_content: 'textbox prompt',
        glossary_settings: constraints.glossary,
        non_translate_settings: constraints.non_translate,
      }),
    )
  })

  it('uses textbox prompt warnings as the effective warnings when textbox output is enabled', async () => {
    const settingsStore = useSettingsStore()
    settingsStore.settings.translation.translationMode = 'single'
    settingsStore.settings.useTextboxPrompt = true
    settingsStore.settings.textboxPrompt = 'textbox prompt'
    const constraints = createEnabledConstraints({ includeNonTranslate: false })

    translateSingleTextMock
      .mockResolvedValueOnce({
        success: true,
        data: {
          translated_text: '阿莉斯',
          warnings: [
            {
              source: 'Alice',
              expectedTarget: '爱丽丝',
              actualTranslation: '阿莉斯',
            },
          ],
        },
      })
      .mockResolvedValueOnce({
        success: true,
        data: {
          translated_text: '爱丽丝',
          warnings: [],
        },
      })

    const result = await executeTranslate({
      imageIndex: 0,
      originalTexts: ['Alice'],
      settingsSnapshot: settingsStore.settings,
      bookTranslationConstraints: constraints,
      isBookshelfMode: true,
    })

    expect(result.textboxTexts).toEqual(['爱丽丝'])
    expect(result.warnings).toEqual([])
  })

  it('falls back to primary warnings when textbox output is empty', async () => {
    const settingsStore = useSettingsStore()
    settingsStore.settings.translation.translationMode = 'single'
    settingsStore.settings.useTextboxPrompt = true
    settingsStore.settings.textboxPrompt = 'textbox prompt'
    const constraints = createEnabledConstraints({ includeNonTranslate: false })

    translateSingleTextMock
      .mockResolvedValueOnce({
        success: true,
        data: {
          translated_text: '阿莉斯',
          warnings: [
            {
              source: 'Alice',
              expectedTarget: '爱丽丝',
              actualTranslation: '阿莉斯',
            },
          ],
        },
      })
      .mockResolvedValueOnce({
        success: true,
        data: {
          translated_text: '',
          warnings: [],
        },
      })

    const result = await executeTranslate({
      imageIndex: 0,
      originalTexts: ['Alice'],
      settingsSnapshot: settingsStore.settings,
      bookTranslationConstraints: constraints,
      isBookshelfMode: true,
    })

    expect(result.textboxTexts).toEqual([''])
    expect(result.warnings).toEqual([
      {
        imageIndex: 0,
        bubbleIndex: 0,
        source: 'Alice',
        expectedTarget: '爱丽丝',
        actualTranslation: '阿莉斯',
      },
    ])
  })
})
