import { afterEach, describe, expect, it, vi } from 'vitest'
import * as fc from 'fast-check'
import { createPinia, setActivePinia } from 'pinia'
import { STORAGE_KEY_THEME, STORAGE_KEY_TRANSLATION_SETTINGS } from '@/constants'
import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'

type ThemePreference = 'light' | 'dark' | 'system'

const themePreferenceArb = fc.constantFrom<ThemePreference>('light', 'dark', 'system')
const nextThemePreference: Record<ThemePreference, ThemePreference> = {
  light: 'dark',
  dark: 'system',
  system: 'light',
}
const validFontSizeArb = fc.integer({ min: 10, max: 100 })
const validColorArb = fc.hexaString({ minLength: 6, maxLength: 6 }).map(hex => `#${hex}`)
const validFontFamilyArb = fc.constantFrom(
  'fonts/STSONG.TTF',
  'fonts/SimHei.ttf',
  'fonts/msyh.ttc',
  'Arial',
  'sans-serif'
)
const validLayoutDirectionArb = fc.constantFrom<'auto' | 'vertical' | 'horizontal'>('auto', 'vertical', 'horizontal')
const validInpaintMethodArb = fc.constantFrom<'solid' | 'lama_mpe' | 'litelama'>('solid', 'lama_mpe', 'litelama')
const ocrEngineArb = fc.constantFrom<'manga_ocr' | 'paddle_ocr' | 'baidu_ocr' | 'ai_vision'>(
  'manga_ocr',
  'paddle_ocr',
  'baidu_ocr',
  'ai_vision'
)
const textDetectorArb = fc.constantFrom<'ctd' | 'yolo' | 'default'>('ctd', 'yolo', 'default')
const translationProviderArb = fc.constantFrom<'siliconflow' | 'deepseek' | 'volcano' | 'gemini' | 'ollama'>(
  'siliconflow',
  'deepseek',
  'volcano',
  'gemini',
  'ollama'
)
const hqProviderArb = fc.constantFrom<'siliconflow' | 'deepseek' | 'volcano' | 'gemini' | 'custom'>(
  'siliconflow',
  'deepseek',
  'volcano',
  'gemini',
  'custom'
)

function installLocalStorageMock() {
  const storage: Record<string, string> = {}

  vi.spyOn(Storage.prototype, 'getItem').mockImplementation((key: string) => storage[key] ?? null)
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation((key: string, value: string) => {
    storage[key] = value
  })
  vi.spyOn(Storage.prototype, 'removeItem').mockImplementation((key: string) => {
    delete storage[key]
  })

  return storage
}

function clearStorage(storage: Record<string, string>): void {
  for (const key of Object.keys(storage)) {
    delete storage[key]
  }
}

function createSettingsStore() {
  setActivePinia(createPinia())
  return useSettingsStore()
}

function reloadPersistedSettings(storage: Record<string, string>) {
  expect(storage[STORAGE_KEY_TRANSLATION_SETTINGS]).toBeDefined()
  const newStore = createSettingsStore()
  newStore.loadFromStorage()
  return newStore
}

describe('settings store properties', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('persists text style updates through the current settings schema', () => {
    const storage = installLocalStorageMock()

    fc.assert(
      fc.property(
        validFontSizeArb,
        validColorArb,
        validColorArb,
        validFontFamilyArb,
        validLayoutDirectionArb,
        validInpaintMethodArb,
        fc.boolean(),
        fc.integer({ min: 1, max: 10 }),
        (fontSize, textColor, fillColor, fontFamily, layoutDirection, inpaintMethod, strokeEnabled, strokeWidth) => {
          clearStorage(storage)
          const store = createSettingsStore()

          store.updateTextStyle({
            fontSize,
            textColor,
            fillColor,
            fontFamily,
            layoutDirection,
            inpaintMethod,
            strokeEnabled,
            strokeWidth,
          })

          const newStore = reloadPersistedSettings(storage)
          expect(newStore.settings.textStyle.fontSize).toBe(fontSize)
          expect(newStore.settings.textStyle.textColor).toBe(textColor)
          expect(newStore.settings.textStyle.fillColor).toBe(fillColor)
          expect(newStore.settings.textStyle.fontFamily).toBe(fontFamily)
          expect(newStore.settings.textStyle.layoutDirection).toBe(layoutDirection)
          expect(newStore.settings.textStyle.inpaintMethod).toBe(inpaintMethod)
          expect(newStore.settings.textStyle.strokeEnabled).toBe(strokeEnabled)
          expect(newStore.settings.textStyle.strokeWidth).toBe(strokeWidth)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('persists OCR engine, language, and detector settings', () => {
    const storage = installLocalStorageMock()

    fc.assert(
      fc.property(
        ocrEngineArb,
        fc.constantFrom('ja', 'zh', 'en', 'ko'),
        textDetectorArb,
        (ocrEngine, sourceLanguage, textDetector) => {
          clearStorage(storage)
          const store = createSettingsStore()

          store.setOcrEngine(ocrEngine)
          store.updateSettings({ sourceLanguage })
          store.setTextDetector(textDetector)

          const newStore = reloadPersistedSettings(storage)
          expect(newStore.settings.ocrEngine).toBe(ocrEngine)
          expect(newStore.settings.sourceLanguage).toBe(sourceLanguage)
          expect(newStore.settings.textDetector).toBe(textDetector)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('normalizes invalid current-schema detector settings to the default detector', () => {
    const storage = installLocalStorageMock()
    const invalidSettings = createDefaultSettings()
    ;(invalidSettings as Record<string, unknown>).textDetector = 'invalid-detector'
    storage[STORAGE_KEY_TRANSLATION_SETTINGS] = JSON.stringify(invalidSettings)

    const store = createSettingsStore()
    store.loadFromStorage()

    expect(store.settings.textDetector).toBe('default')
  })

  it('persists translation provider execution settings', () => {
    const storage = installLocalStorageMock()

    fc.assert(
      fc.property(
        translationProviderArb,
        fc.string({ minLength: 0, maxLength: 50 }),
        fc.string({ minLength: 0, maxLength: 50 }),
        fc.integer({ min: 0, max: 100 }),
        fc.integer({ min: 1, max: 10 }),
        (provider, apiKey, modelName, rpmLimit, businessRetries) => {
          clearStorage(storage)
          const store = createSettingsStore()

          store.updateTranslationService({
            provider,
            apiKey,
            modelName,
            rpmLimit,
            businessRetries,
          })

          const newStore = reloadPersistedSettings(storage)
          expect(newStore.settings.translation.provider).toBe(provider)
          expect(newStore.settings.translation.apiKey).toBe(apiKey)
          expect(newStore.settings.translation.modelName).toBe(modelName)
          expect(newStore.settings.translation.openaiOptions.execution.rpmLimit).toBe(rpmLimit)
          expect(newStore.settings.translation.openaiOptions.execution.businessRetries).toBe(businessRetries)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('cycles theme preference and persists each selected theme', () => {
    const storage = installLocalStorageMock()

    fc.assert(
      fc.property(themePreferenceArb, initialTheme => {
        clearStorage(storage)
        const store = createSettingsStore()

        store.setTheme(initialTheme)
        expect(store.theme).toBe(initialTheme)

        store.toggleTheme()
        const expectedTheme = nextThemePreference[initialTheme]
        expect(store.theme).toBe(expectedTheme)
        expect(storage[STORAGE_KEY_THEME]).toBe(expectedTheme)

        store.toggleTheme()
        store.toggleTheme()
        expect(store.theme).toBe(initialTheme)
      }),
      { numRuns: 100 }
    )
  })

  it('loads persisted theme preference into a new settings store', () => {
    const storage = installLocalStorageMock()

    fc.assert(
      fc.property(themePreferenceArb, theme => {
        clearStorage(storage)
        const store = createSettingsStore()

        store.setTheme(theme)
        expect(storage[STORAGE_KEY_THEME]).toBe(theme)

        const newStore = createSettingsStore()
        newStore.loadThemeFromStorage()
        expect(newStore.theme).toBe(theme)
      }),
      { numRuns: 100 }
    )
  })

  it('resets edited text style fields to their defaults', () => {
    const storage = installLocalStorageMock()

    fc.assert(
      fc.property(validFontSizeArb, validColorArb, (fontSize, textColor) => {
        clearStorage(storage)
        const store = createSettingsStore()
        const defaultFontSize = store.settings.textStyle.fontSize
        const defaultTextColor = store.settings.textStyle.textColor

        store.updateTextStyle({ fontSize, textColor })
        expect(store.settings.textStyle.fontSize).toBe(fontSize)
        expect(store.settings.textStyle.textColor).toBe(textColor)

        store.resetToDefaults()
        expect(store.settings.textStyle.fontSize).toBe(defaultFontSize)
        expect(store.settings.textStyle.textColor).toBe(defaultTextColor)
      }),
      { numRuns: 100 }
    )
  })

  it('persists high-quality translation provider settings', () => {
    const storage = installLocalStorageMock()

    fc.assert(
      fc.property(
        hqProviderArb,
        fc.integer({ min: 1, max: 10 }),
        fc.integer({ min: 1, max: 20 }),
        fc.boolean(),
        (provider, batchSize, rpmLimit, forceJsonOutput) => {
          clearStorage(storage)
          const store = createSettingsStore()

          store.updateHqTranslation({
            provider,
            batchSize,
            rpmLimit,
            forceJsonOutput,
          })

          const newStore = reloadPersistedSettings(storage)
          expect(newStore.settings.hqTranslation.provider).toBe(provider)
          expect(newStore.settings.hqTranslation.batchSize).toBe(batchSize)
          expect(newStore.settings.hqTranslation.openaiOptions.execution.rpmLimit).toBe(rpmLimit)
          expect(newStore.settings.hqTranslation.openaiOptions.request.forceJsonOutput).toBe(forceJsonOutput)
        }
      ),
      { numRuns: 100 }
    )
  })
})
