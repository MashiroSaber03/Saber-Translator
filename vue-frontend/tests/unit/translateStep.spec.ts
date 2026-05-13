import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

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
import { useSettingsStore } from '@/stores/settingsStore'
import { createEmptyBookTranslationConstraints } from '@/utils/bookTranslationConstraints'

describe('executeTranslate', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    parallelTranslateMock.mockReset()
    translateSingleTextMock.mockReset()
  })

  it('forwards glossary and non-translate settings in batch mode and returns warnings', async () => {
    const settingsStore = useSettingsStore()
    settingsStore.settings.translation.translationMode = 'batch'
    const constraints = createEmptyBookTranslationConstraints()
    constraints.glossary.enabled = true
    constraints.glossary.entries = [
      { source: 'Alice', target: '爱丽丝', note: '主角', matchMode: 'text' } as any,
    ]
    constraints.non_translate.enabled = true
    constraints.non_translate.entries = [
      { pattern: '<keep>', note: '占位符', matchMode: 'text' } as any,
    ]

    parallelTranslateMock.mockResolvedValue({
      success: true,
      translated_texts: ['爱丽丝 <keep>'],
      textbox_texts: [],
      warnings: [{ imageIndex: 0, bubbleIndex: 0, source: 'Alice', expectedTarget: '爱丽丝', actualTranslation: '爱丽丝 <keep>' }],
    })

    const result = await executeTranslate({
      imageIndex: 0,
      originalTexts: ['Alice <keep>'],
      settingsSnapshot: settingsStore.settings,
      bookTranslationConstraints: constraints,
      isBookshelfMode: true,
    })

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
    const constraints = createEmptyBookTranslationConstraints()
    constraints.glossary.enabled = true
    constraints.glossary.entries = [
      { source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' } as any,
    ]
    constraints.non_translate.enabled = true
    constraints.non_translate.entries = [
      { pattern: '<keep>', note: '', matchMode: 'text' } as any,
    ]

    translateSingleTextMock.mockResolvedValue({
      success: true,
      data: {
        translated_text: '爱丽丝 <keep>',
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
    const constraints = createEmptyBookTranslationConstraints()
    constraints.glossary.enabled = true
    constraints.glossary.entries = [
      { source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' } as any,
    ]
    constraints.non_translate.enabled = true
    constraints.non_translate.entries = [
      { pattern: '<keep>', note: '', matchMode: 'text' } as any,
    ]

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
    const constraints = createEmptyBookTranslationConstraints()
    constraints.glossary.enabled = true
    constraints.glossary.entries = [
      { source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' } as any,
    ]

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
    const constraints = createEmptyBookTranslationConstraints()
    constraints.glossary.enabled = true
    constraints.glossary.entries = [
      { source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' } as any,
    ]

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
