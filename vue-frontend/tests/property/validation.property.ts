import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import * as fc from 'fast-check'
import { AI_PROVIDER_MANIFEST } from '@/config/aiProviders'
import { useValidation } from '@/composables/useValidation'
import { useSettingsStore } from '@/stores/settings'
import type {
  HqTranslationProvider,
  OcrEngine,
  ProofreadingRound,
  TranslationProvider,
} from '@/types/settings'

function constantFromList<T>(values: T[]): fc.Arbitrary<T> {
  if (values.length === 0) {
    throw new Error('Expected at least one generated value')
  }
  return fc.constantFrom(...(values as [T, ...T[]]))
}

const translationProviderIds = AI_PROVIDER_MANIFEST
  .filter(provider => provider.capabilities.includes('translation'))
  .map(provider => provider.id as TranslationProvider)
const translationApiKeyProviderIds = AI_PROVIDER_MANIFEST
  .filter(provider => provider.capabilities.includes('translation') && provider.requiresApiKey)
  .map(provider => provider.id as TranslationProvider)
const localTranslationProviderIds = AI_PROVIDER_MANIFEST
  .filter(provider => provider.capabilities.includes('translation') && provider.isLocal)
  .map(provider => provider.id as TranslationProvider)
const translationBaseUrlProviderIds = AI_PROVIDER_MANIFEST
  .filter(provider => provider.capabilities.includes('translation') && provider.requiresBaseUrl)
  .map(provider => provider.id as TranslationProvider)
const hqTranslationProviderIds = AI_PROVIDER_MANIFEST
  .filter(provider => provider.capabilities.includes('hqTranslation'))
  .map(provider => provider.id as HqTranslationProvider)
const hqApiKeyProviderIds = AI_PROVIDER_MANIFEST
  .filter(provider => provider.capabilities.includes('hqTranslation') && provider.requiresApiKey)
  .map(provider => provider.id as HqTranslationProvider)

describe('validation properties', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
  })

  const translationProviderArb = constantFromList(translationProviderIds)
  const translationApiKeyProviderArb = constantFromList(translationApiKeyProviderIds)
  const localTranslationProviderArb = constantFromList(localTranslationProviderIds)
  const translationBaseUrlProviderArb = constantFromList(translationBaseUrlProviderIds)
  const hqTranslationProviderArb = constantFromList(hqTranslationProviderIds)
  const hqApiKeyProviderArb = constantFromList(hqApiKeyProviderIds)
  const nonEmptyStringArb = fc
    .string({ minLength: 1, maxLength: 50 })
    .filter(value => value.trim().length > 0)
  const emptyOrWhitespaceArb = fc.constantFrom('', ' ', '  ', '\t', '\n')
  const validProofreadingRoundArb: fc.Arbitrary<ProofreadingRound> = fc.record({
    name: nonEmptyStringArb,
    provider: hqTranslationProviderArb,
    apiKey: nonEmptyStringArb,
    modelName: nonEmptyStringArb,
    customBaseUrl: nonEmptyStringArb,
    batchSize: fc.integer({ min: 1, max: 10 }),
    rpmLimit: fc.integer({ min: 0, max: 100 }),
    forceJsonOutput: fc.boolean(),
    prompt: fc.string(),
  })
  const invalidProofreadingRoundArb: fc.Arbitrary<ProofreadingRound> = fc.oneof(
    fc.record({
      name: nonEmptyStringArb,
      provider: fc.constant('' as HqTranslationProvider),
      apiKey: nonEmptyStringArb,
      modelName: nonEmptyStringArb,
      customBaseUrl: nonEmptyStringArb,
      batchSize: fc.integer({ min: 1, max: 10 }),
      rpmLimit: fc.integer({ min: 0, max: 100 }),
      forceJsonOutput: fc.boolean(),
      prompt: fc.string(),
    }),
    fc.record({
      name: nonEmptyStringArb,
      provider: hqApiKeyProviderArb,
      apiKey: emptyOrWhitespaceArb,
      modelName: nonEmptyStringArb,
      customBaseUrl: nonEmptyStringArb,
      batchSize: fc.integer({ min: 1, max: 10 }),
      rpmLimit: fc.integer({ min: 0, max: 100 }),
      forceJsonOutput: fc.boolean(),
      prompt: fc.string(),
    }),
    fc.record({
      name: nonEmptyStringArb,
      provider: hqTranslationProviderArb,
      apiKey: nonEmptyStringArb,
      modelName: emptyOrWhitespaceArb,
      customBaseUrl: nonEmptyStringArb,
      batchSize: fc.integer({ min: 1, max: 10 }),
      rpmLimit: fc.integer({ min: 0, max: 100 }),
      forceJsonOutput: fc.boolean(),
      prompt: fc.string(),
    }),
  )

  it('accepts complete translation configs for every manifest translation provider', () => {
    fc.assert(
      fc.property(
        translationProviderArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        (provider, apiKey, modelName, customBaseUrl) => {
          setActivePinia(createPinia())
          const settingsStore = useSettingsStore()
          const { validateTranslationConfig } = useValidation()

          settingsStore.updateTranslationService({
            provider,
            apiKey,
            modelName,
            customBaseUrl,
          })

          expect(validateTranslationConfig().valid).toBe(true)
        },
      ),
      { numRuns: 100 },
    )
  })

  it('rejects manifest translation providers that require a missing API key', () => {
    fc.assert(
      fc.property(
        translationApiKeyProviderArb,
        emptyOrWhitespaceArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        (provider, apiKey, modelName, customBaseUrl) => {
          setActivePinia(createPinia())
          const settingsStore = useSettingsStore()
          const { validateTranslationConfig } = useValidation()

          settingsStore.updateTranslationService({
            provider,
            apiKey,
            modelName,
            customBaseUrl,
          })

          const result = validateTranslationConfig()
          expect(result.valid).toBe(false)
          expect(result.message).toContain('API Key')
        },
      ),
      { numRuns: 100 },
    )
  })

  it('rejects local translation providers without a model name', () => {
    fc.assert(
      fc.property(localTranslationProviderArb, emptyOrWhitespaceArb, (provider, modelName) => {
        setActivePinia(createPinia())
        const settingsStore = useSettingsStore()
        const { validateTranslationConfig } = useValidation()

        settingsStore.updateTranslationService({
          provider,
          apiKey: '',
          modelName,
        })

        const result = validateTranslationConfig()
        expect(result.valid).toBe(false)
        expect(result.message).toContain('模型名称')
      }),
      { numRuns: 100 },
    )
  })

  it('rejects manifest translation providers that require a missing base URL', () => {
    fc.assert(
      fc.property(
        translationBaseUrlProviderArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        emptyOrWhitespaceArb,
        (provider, apiKey, modelName, customBaseUrl) => {
          setActivePinia(createPinia())
          const settingsStore = useSettingsStore()
          const { validateTranslationConfig } = useValidation()

          settingsStore.updateTranslationService({
            provider,
            apiKey,
            modelName,
            customBaseUrl,
          })

          const result = validateTranslationConfig()
          expect(result.valid).toBe(false)
          expect(result.message).toContain('Base URL')
        },
      ),
      { numRuns: 100 },
    )
  })

  it('accepts complete HQ translation configs for every manifest HQ provider', () => {
    fc.assert(
      fc.property(
        hqTranslationProviderArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        (provider, apiKey, modelName, customBaseUrl) => {
          setActivePinia(createPinia())
          const settingsStore = useSettingsStore()
          const { validateHqTranslationConfig } = useValidation()

          settingsStore.updateHqTranslation({
            provider,
            apiKey,
            modelName,
            customBaseUrl,
          })

          expect(validateHqTranslationConfig().valid).toBe(true)
        },
      ),
      { numRuns: 100 },
    )
  })

  it('rejects HQ providers that require a missing API key', () => {
    fc.assert(
      fc.property(
        hqApiKeyProviderArb,
        emptyOrWhitespaceArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        (provider, apiKey, modelName, customBaseUrl) => {
          setActivePinia(createPinia())
          const settingsStore = useSettingsStore()
          const { validateHqTranslationConfig } = useValidation()

          settingsStore.updateHqTranslation({
            provider,
            apiKey,
            modelName,
            customBaseUrl,
          })

          const result = validateHqTranslationConfig()
          expect(result.valid).toBe(false)
          expect(result.message).toContain('API Key')
        },
      ),
      { numRuns: 100 },
    )
  })

  it('rejects HQ providers without a model name', () => {
    fc.assert(
      fc.property(
        hqTranslationProviderArb,
        nonEmptyStringArb,
        emptyOrWhitespaceArb,
        nonEmptyStringArb,
        (provider, apiKey, modelName, customBaseUrl) => {
          setActivePinia(createPinia())
          const settingsStore = useSettingsStore()
          const { validateHqTranslationConfig } = useValidation()

          settingsStore.updateHqTranslation({
            provider,
            apiKey,
            modelName,
            customBaseUrl,
          })

          const result = validateHqTranslationConfig()
          expect(result.valid).toBe(false)
          expect(result.message).toContain('模型名称')
        },
      ),
      { numRuns: 100 },
    )
  })

  it('accepts complete proofreading round configs', () => {
    fc.assert(
      fc.property(fc.array(validProofreadingRoundArb, { minLength: 1, maxLength: 5 }), rounds => {
        setActivePinia(createPinia())
        const { validateProofreadingConfig } = useValidation()

        expect(validateProofreadingConfig(rounds).valid).toBe(true)
      }),
      { numRuns: 100 },
    )
  })

  it('rejects an empty proofreading round list', () => {
    const { validateProofreadingConfig } = useValidation()

    const result = validateProofreadingConfig([])

    expect(result.valid).toBe(false)
    expect(result.message).toContain('至少一个校对轮次')
  })

  it('rejects invalid proofreading rounds', () => {
    fc.assert(
      fc.property(invalidProofreadingRoundArb, invalidRound => {
        setActivePinia(createPinia())
        const { validateProofreadingConfig } = useValidation()

        expect(validateProofreadingConfig([invalidRound]).valid).toBe(false)
      }),
      { numRuns: 100 },
    )
  })

  it('reports missing items for invalid translation configs', () => {
    fc.assert(
      fc.property(translationApiKeyProviderArb, provider => {
        setActivePinia(createPinia())
        const settingsStore = useSettingsStore()
        const { validateTranslationConfig } = useValidation()

        settingsStore.updateTranslationService({
          provider,
          apiKey: '',
          modelName: '',
          customBaseUrl: '',
        })

        const result = validateTranslationConfig()
        expect(result.valid).toBe(false)
        expect(result.missingItems?.length).toBeGreaterThan(0)
        expect(result.missingItems?.some(item => item.includes('API Key'))).toBe(true)
      }),
      { numRuns: 100 },
    )
  })

  it('returns an empty message for valid translation configs', () => {
    fc.assert(
      fc.property(
        translationProviderArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        (provider, apiKey, modelName, customBaseUrl) => {
          setActivePinia(createPinia())
          const settingsStore = useSettingsStore()
          const { validateTranslationConfig } = useValidation()

          settingsStore.updateTranslationService({
            provider,
            apiKey,
            modelName,
            customBaseUrl,
          })

          const result = validateTranslationConfig()
          expect(result.valid).toBe(true)
          expect(result.message).toBe('')
        },
      ),
      { numRuns: 100 },
    )
  })

  it('accepts OCR engines without credential requirements', () => {
    const ocrEnginesWithoutCredentials: OcrEngine[] = ['manga_ocr', 'paddle_ocr']

    fc.assert(
      fc.property(constantFromList(ocrEnginesWithoutCredentials), engine => {
        setActivePinia(createPinia())
        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        settingsStore.setOcrEngine(engine)

        expect(validateOcrConfig().valid).toBe(true)
      }),
      { numRuns: 100 },
    )
  })

  it('rejects Baidu OCR without an API key', () => {
    fc.assert(
      fc.property(emptyOrWhitespaceArb, nonEmptyStringArb, (apiKey, secretKey) => {
        setActivePinia(createPinia())
        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        settingsStore.setOcrEngine('baidu_ocr')
        settingsStore.updateBaiduOcr({ apiKey, secretKey })

        const result = validateOcrConfig()
        expect(result.valid).toBe(false)
        expect(result.message).toContain('API Key')
      }),
      { numRuns: 100 },
    )
  })

  it('rejects Baidu OCR without a secret key', () => {
    fc.assert(
      fc.property(nonEmptyStringArb, emptyOrWhitespaceArb, (apiKey, secretKey) => {
        setActivePinia(createPinia())
        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        settingsStore.setOcrEngine('baidu_ocr')
        settingsStore.updateBaiduOcr({ apiKey, secretKey })

        const result = validateOcrConfig()
        expect(result.valid).toBe(false)
        expect(result.message).toContain('Secret Key')
      }),
      { numRuns: 100 },
    )
  })

  it('accepts complete Baidu OCR config', () => {
    fc.assert(
      fc.property(nonEmptyStringArb, nonEmptyStringArb, (apiKey, secretKey) => {
        setActivePinia(createPinia())
        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        settingsStore.setOcrEngine('baidu_ocr')
        settingsStore.updateBaiduOcr({ apiKey, secretKey })

        expect(validateOcrConfig().valid).toBe(true)
      }),
      { numRuns: 100 },
    )
  })

  it('rejects AI vision OCR without an API key when its provider requires one', () => {
    fc.assert(
      fc.property(emptyOrWhitespaceArb, nonEmptyStringArb, (apiKey, modelName) => {
        setActivePinia(createPinia())
        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        settingsStore.setOcrEngine('ai_vision')
        settingsStore.updateAiVisionOcr({
          provider: 'gemini',
          apiKey,
          modelName,
        })

        const result = validateOcrConfig()
        expect(result.valid).toBe(false)
        expect(result.message).toContain('API Key')
      }),
      { numRuns: 100 },
    )
  })

  it('rejects AI vision OCR without a model name', () => {
    fc.assert(
      fc.property(nonEmptyStringArb, emptyOrWhitespaceArb, (apiKey, modelName) => {
        setActivePinia(createPinia())
        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        settingsStore.setOcrEngine('ai_vision')
        settingsStore.updateAiVisionOcr({
          provider: 'gemini',
          apiKey,
          modelName,
        })

        const result = validateOcrConfig()
        expect(result.valid).toBe(false)
        expect(result.message).toContain('模型名称')
      }),
      { numRuns: 100 },
    )
  })

  it('accepts complete AI vision OCR config', () => {
    fc.assert(
      fc.property(nonEmptyStringArb, nonEmptyStringArb, (apiKey, modelName) => {
        setActivePinia(createPinia())
        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        settingsStore.setOcrEngine('ai_vision')
        settingsStore.updateAiVisionOcr({
          provider: 'gemini',
          apiKey,
          modelName,
        })

        expect(validateOcrConfig().valid).toBe(true)
      }),
      { numRuns: 100 },
    )
  })

  it('matches validation helper decisions with the provider manifest', () => {
    fc.assert(
      fc.property(translationProviderArb, provider => {
        setActivePinia(createPinia())
        const { requiresApiKey, isLocalProvider, requiresBaseUrl } = useValidation()
        const manifest = AI_PROVIDER_MANIFEST.find(entry => entry.id === provider)

        expect(requiresApiKey(provider)).toBe(manifest?.requiresApiKey)
        expect(isLocalProvider(provider)).toBe(manifest?.isLocal)
        expect(requiresBaseUrl(provider)).toBe(manifest?.requiresBaseUrl)
      }),
      { numRuns: 100 },
    )
  })
})
