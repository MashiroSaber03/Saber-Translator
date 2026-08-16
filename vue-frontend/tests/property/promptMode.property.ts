import { afterEach, describe, expect, it, vi } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useSettingsStore } from '@/stores/settings'
import promptDefaults from '../../../src/shared/prompt_defaults_factory.json'
import {
  DEFAULT_AI_VISION_OCR_JSON_PROMPT,
  DEFAULT_AI_VISION_OCR_PROMPT,
  DEFAULT_AUTO_GLOSSARY_PROMPT,
  DEFAULT_HQ_TRANSLATE_PROMPT,
  DEFAULT_PROOFREADING_PROMPT,
  DEFAULT_SINGLE_BUBBLE_JSON_PROMPT,
  DEFAULT_SINGLE_BUBBLE_PROMPT,
  DEFAULT_TRANSLATE_JSON_PROMPT,
  DEFAULT_TRANSLATE_PROMPT,
  DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT,
  getPaddleOcrVlPrompt,
} from '@/constants'
import type { TranslationMode } from '@/types/settings'

type SettingsStore = ReturnType<typeof useSettingsStore>

function installMemoryStorage(initial: Record<string, string> = {}): Record<string, string> {
  const state = { ...initial }
  const storage: Storage = {
    get length() {
      return Object.keys(state).length
    },
    clear: () => {
      for (const key of Object.keys(state)) {
        delete state[key]
      }
    },
    getItem: (key: string) => {
      return Object.prototype.hasOwnProperty.call(state, key) ? state[key]! : null
    },
    key: (index: number) => {
      return Object.keys(state)[index] ?? null
    },
    removeItem: (key: string) => {
      delete state[key]
    },
    setItem: (key: string, value: string) => {
      state[key] = value
    },
  }
  vi.stubGlobal('localStorage', storage)
  return state
}

function createStore(): SettingsStore {
  setActivePinia(createPinia())
  return useSettingsStore()
}

function expectedTranslatePrompt(store: SettingsStore, mode: TranslationMode, forceJsonOutput: boolean): string {
  if (mode === 'single') {
    return forceJsonOutput
      ? store.settings.translation.singleJsonPrompt
      : store.settings.translation.singleNormalPrompt
  }
  return forceJsonOutput
    ? store.settings.translation.batchJsonPrompt
    : store.settings.translation.batchNormalPrompt
}

describe('prompt mode properties', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('keeps bundled prompt defaults distinct and available', () => {
    for (const prompt of [
      DEFAULT_TRANSLATE_PROMPT,
      DEFAULT_TRANSLATE_JSON_PROMPT,
      DEFAULT_AI_VISION_OCR_PROMPT,
      DEFAULT_AI_VISION_OCR_JSON_PROMPT,
    ]) {
      expect(prompt).toEqual(expect.any(String))
      expect(prompt.length).toBeGreaterThan(0)
    }

    expect(DEFAULT_TRANSLATE_PROMPT).not.toBe(DEFAULT_TRANSLATE_JSON_PROMPT)
    expect(DEFAULT_AI_VISION_OCR_PROMPT).not.toBe(DEFAULT_AI_VISION_OCR_JSON_PROMPT)
  })

  it('uses the shared factory prompt resource without frontend copies', () => {
    expect({
      singleNormal: DEFAULT_SINGLE_BUBBLE_PROMPT,
      singleJson: DEFAULT_SINGLE_BUBBLE_JSON_PROMPT,
      batchNormal: DEFAULT_TRANSLATE_PROMPT,
      batchJson: DEFAULT_TRANSLATE_JSON_PROMPT,
      aiVisionOcrNormal: DEFAULT_AI_VISION_OCR_PROMPT,
      aiVisionOcrJson: DEFAULT_AI_VISION_OCR_JSON_PROMPT,
      hqTranslation: DEFAULT_HQ_TRANSLATE_PROMPT,
      proofreading: DEFAULT_PROOFREADING_PROMPT,
      autoGlossary: DEFAULT_AUTO_GLOSSARY_PROMPT,
      webImportExtraction: DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT,
    }).toEqual(promptDefaults)
  })

  it('selects the matching translation prompt variant for every mode', () => {
    fc.assert(
      fc.property(
        fc.constantFrom<TranslationMode>('batch', 'single'),
        fc.boolean(),
        (translationMode, forceJsonOutput) => {
          installMemoryStorage()
          const store = createStore()
          store.updateTranslationService({ translationMode })

          store.setTranslatePromptMode(forceJsonOutput)

          expect(store.settings.translation.translationMode).toBe(translationMode)
          expect(store.settings.translation.openaiOptions.request.forceJsonOutput).toBe(forceJsonOutput)
          expect(store.settings.translatePrompt).toBe(
            expectedTranslatePrompt(store, translationMode, forceJsonOutput),
          )

          store.setTranslatePromptMode(!forceJsonOutput)
          expect(store.settings.translatePrompt).toBe(
            expectedTranslatePrompt(store, translationMode, !forceJsonOutput),
          )

          store.setTranslatePromptMode(forceJsonOutput)
          expect(store.settings.translatePrompt).toBe(
            expectedTranslatePrompt(store, translationMode, forceJsonOutput),
          )
        },
      ),
    )
  })

  it('selects the matching AI vision OCR prompt and uses the selected PaddleOCR-VL language', () => {
    fc.assert(
      fc.property(fc.boolean(), fc.string({ minLength: 1, maxLength: 200 }), (forceJsonOutput, customPrompt) => {
        installMemoryStorage()
        const store = createStore()

        store.setAiVisionOcrPromptMode(forceJsonOutput ? 'json' : 'normal')

        expect(store.settings.aiVisionOcr.openaiOptions.request.forceJsonOutput).toBe(forceJsonOutput)
        expect(store.settings.aiVisionOcr.promptMode).toBe(forceJsonOutput ? 'json' : 'normal')
        expect(store.settings.aiVisionOcr.prompt).toBe(
          forceJsonOutput ? DEFAULT_AI_VISION_OCR_JSON_PROMPT : DEFAULT_AI_VISION_OCR_PROMPT,
        )

        store.updateAiVisionOcr({ prompt: customPrompt })
        store.updatePaddleOcrVl({ sourceLanguage: 'korean' })
        store.setAiVisionOcrPromptMode('paddleocr_vl')

        expect(store.settings.aiVisionOcr.promptMode).toBe('paddleocr_vl')
        expect(store.settings.aiVisionOcr.openaiOptions.request.forceJsonOutput).toBe(false)
        expect(store.settings.aiVisionOcr.prompt).toBe(getPaddleOcrVlPrompt('korean'))

        store.updatePaddleOcrVl({ sourceLanguage: 'chinese_cht' })
        expect(store.settings.aiVisionOcr.prompt).toBe(getPaddleOcrVlPrompt('chinese_cht'))
      }),
    )
  })

  it('keeps custom prompt text in the active draft without changing prompt modes', () => {
    fc.assert(
      fc.property(
        fc.constantFrom<TranslationMode>('batch', 'single'),
        fc.boolean(),
        fc.string({ minLength: 1, maxLength: 300 }),
        fc.string({ minLength: 1, maxLength: 300 }),
        (translationMode, forceJsonOutput, translatePrompt, aiVisionPrompt) => {
          installMemoryStorage()
          const store = createStore()
          store.updateTranslationService({ translationMode })
          store.setTranslatePromptMode(forceJsonOutput)

          store.setTranslatePrompt(translatePrompt)
          store.updateAiVisionOcr({ prompt: aiVisionPrompt })

          expect(store.settings.translation.translationMode).toBe(translationMode)
          expect(store.settings.translation.openaiOptions.request.forceJsonOutput).toBe(forceJsonOutput)
          expect(store.settings.translatePrompt).toBe(translatePrompt)
          expect(store.settings.aiVisionOcr.prompt).toBe(aiVisionPrompt)
        },
      ),
    )
  })

  it('changes prompt modes without mutating neighboring provider fields', () => {
    fc.assert(
      fc.property(
        fc.boolean(),
        fc.string({ minLength: 1, maxLength: 50 }),
        fc.string({ minLength: 1, maxLength: 50 }),
        (forceJsonOutput, apiKey, modelName) => {
          installMemoryStorage()
          const store = createStore()

          store.updateTranslationService({ apiKey, modelName, forceJsonOutput: false })
          store.setTranslatePromptMode(forceJsonOutput)

          expect(store.settings.translation.apiKey).toBe(apiKey)
          expect(store.settings.translation.modelName).toBe(modelName)
          expect(store.settings.translation.openaiOptions.request.forceJsonOutput).toBe(forceJsonOutput)
        },
      ),
    )
  })

  it('keeps translation and AI vision OCR JSON modes independent', () => {
    fc.assert(
      fc.property(fc.boolean(), fc.boolean(), (translateJsonMode, aiVisionJsonMode) => {
        installMemoryStorage()
        const store = createStore()

        store.setTranslatePromptMode(translateJsonMode)
        store.setAiVisionOcrPromptMode(aiVisionJsonMode ? 'json' : 'normal')
        store.setTranslatePromptMode(!translateJsonMode)

        expect(store.settings.translation.openaiOptions.request.forceJsonOutput).toBe(!translateJsonMode)
        expect(store.settings.aiVisionOcr.openaiOptions.request.forceJsonOutput).toBe(aiVisionJsonMode)

        store.setAiVisionOcrPromptMode(aiVisionJsonMode ? 'normal' : 'json')

        expect(store.settings.translation.openaiOptions.request.forceJsonOutput).toBe(!translateJsonMode)
        expect(store.settings.aiVisionOcr.openaiOptions.request.forceJsonOutput).toBe(!aiVisionJsonMode)
      }),
    )
  })
})
