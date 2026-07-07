import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const { extractGlossaryEntriesMock, saveBookConstraintsMock } = vi.hoisted(() => ({
  extractGlossaryEntriesMock: vi.fn(),
  saveBookConstraintsMock: vi.fn(),
}))

vi.mock('@/api/translate', () => ({
  extractGlossaryEntries: extractGlossaryEntriesMock,
}))

import { executeAutoGlossary } from '@/composables/translation/core/steps/autoGlossary'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import { useSettingsStore } from '@/stores/settings'
import { createEmptyBookTranslationConstraints } from '@/utils/bookTranslationConstraints'

describe('executeAutoGlossary', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    extractGlossaryEntriesMock.mockReset()
    saveBookConstraintsMock.mockReset()
  })

  it('keeps constraint cloning on the shared helper', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/composables/translation/core/steps/autoGlossary.ts'), 'utf8')

    expect(source).toContain("from '@/utils/deepClone'")
    expect(source).not.toContain('function cloneConstraints')
    expect(source).not.toContain('JSON.parse(JSON.stringify(constraints))')
  })

  it('skips outside bookshelf mode', async () => {
    const settingsStore = useSettingsStore()
    const constraints = createEmptyBookTranslationConstraints()
    constraints.glossary.enabled = true
    constraints.glossary.autoExtractEnabled = true

    const result = await executeAutoGlossary({
      originalTexts: ['Alice'],
      settingsSnapshot: settingsStore.settings,
      bookTranslationConstraints: constraints,
      isBookshelfMode: false,
    })

    expect(extractGlossaryEntriesMock).not.toHaveBeenCalled()
    expect(result.autoGlossaryStats).toEqual({
      added: 0,
      duplicates: 0,
      failedPages: 0,
    })
    expect(result.bookTranslationConstraints).toEqual(constraints)
    expect(result.bookTranslationConstraints).not.toBe(constraints)
    expect(result.bookTranslationConstraints.glossary.entries).not.toBe(constraints.glossary.entries)
  })

  it('extracts, saves and returns updated constraints when enabled', async () => {
    const settingsStore = useSettingsStore()
    settingsStore.settings.translation.provider = 'siliconflow'
    settingsStore.settings.translation.apiKey = 'test-key'
    settingsStore.settings.translation.modelName = 'test-model'
    settingsStore.settings.translation.openaiOptions.request.forceJsonOutput = false
    settingsStore.settings.translation.openaiOptions.request.extraBody = { top_p: 0.95 }
    settingsStore.settings.translation.openaiOptions.execution.useStream = true

    const store = useBookTranslationConstraintsStore()
    store.loadBookConstraints('book-1', createEmptyBookTranslationConstraints())
    store.constraints.glossary.enabled = true
    store.constraints.glossary.autoExtractEnabled = true
    store.constraints.glossary.autoExtractPrompt = '请提取实体\n\nOCR 文本：\n{ocr_text}'
    store.saveBookConstraints = saveBookConstraintsMock.mockImplementation(async (nextConstraints) => {
      store.constraints = JSON.parse(JSON.stringify(nextConstraints))
      return true
    })

    extractGlossaryEntriesMock.mockResolvedValue({
      success: true,
      new_entries: [
        { source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' },
      ],
      candidate_count: 2,
      duplicate_count: 1,
    })

    const result = await executeAutoGlossary({
      originalTexts: ['Alice and Bob'],
      settingsSnapshot: settingsStore.settings,
      bookTranslationConstraints: store.constraints,
      isBookshelfMode: true,
    })

    expect(extractGlossaryEntriesMock).toHaveBeenCalled()
    expect(extractGlossaryEntriesMock).toHaveBeenCalledWith(expect.objectContaining({
      prompt: '请提取实体\n\nOCR 文本：\n{ocr_text}',
      openai_options: {
        request: {
          force_json_output: false,
          extra_body: { top_p: 0.95 },
        },
        execution: expect.objectContaining({
          use_stream: true,
        }),
      },
    }))
    expect(saveBookConstraintsMock).toHaveBeenCalled()
    expect(saveBookConstraintsMock.mock.calls[0]?.[0]).not.toBe(store.constraints)
    expect(result.bookTranslationConstraints.glossary.entries).toEqual([
      { source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' },
    ])
    expect(result.bookTranslationConstraints).not.toBe(store.constraints)
    expect(result.bookTranslationConstraints.glossary.entries).not.toBe(store.constraints.glossary.entries)
    expect(result.autoGlossaryStats).toEqual({
      added: 1,
      duplicates: 1,
      failedPages: 0,
    })
  })
})
