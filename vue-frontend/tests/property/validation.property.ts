import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import * as fc from 'fast-check'

import { AI_PROVIDER_MANIFEST } from '@/config/aiProviders'
import { useValidation } from '@/composables/useValidation'
import { useSettingsStore } from '@/stores/settings'
import type {
  HqTranslationProvider,
  ProofreadingRound,
  TranslationProvider,
} from '@/types/settings'

const toast = vi.hoisted(() => ({ error: vi.fn() }))

vi.mock('@/utils/toast', () => ({
  useToast: () => toast,
}))

function constantFromList<T>(values: T[]): fc.Arbitrary<T> {
  if (values.length === 0) throw new Error('Expected at least one generated value')
  return fc.constantFrom(...(values as [T, ...T[]]))
}

function addStoredCredential(
  store: ReturnType<typeof useSettingsStore>,
  domain: string,
  provider: string,
): void {
  store.credentialSummaries.push({
    credentialId: `credential-${domain}-${provider}`,
    credentialVersionId: `version-${domain}-${provider}`,
    currentVersion: 1,
    domain,
    hasKey: true,
    provider,
    revision: 1,
  })
}

const translationProviderIds = AI_PROVIDER_MANIFEST
  .filter(provider => provider.capabilities.includes('translation'))
  .map(provider => provider.id as TranslationProvider)
const translationApiKeyProviderIds = AI_PROVIDER_MANIFEST
  .filter(provider => (
    provider.capabilities.includes('translation') && provider.requiresApiKey
  ))
  .map(provider => provider.id as TranslationProvider)
const translationBaseUrlProviderIds = AI_PROVIDER_MANIFEST
  .filter(provider => (
    provider.capabilities.includes('translation') && provider.requiresBaseUrl
  ))
  .map(provider => provider.id as TranslationProvider)
const hqTranslationProviderIds = AI_PROVIDER_MANIFEST
  .filter(provider => provider.capabilities.includes('hqTranslation'))
  .map(provider => provider.id as HqTranslationProvider)
const hqApiKeyProviderIds = AI_PROVIDER_MANIFEST
  .filter(provider => (
    provider.capabilities.includes('hqTranslation') && provider.requiresApiKey
  ))
  .map(provider => provider.id as HqTranslationProvider)

const nonEmptyStringArb = fc
  .string({ minLength: 1, maxLength: 50 })
  .filter(value => value.trim().length > 0)
const emptyStringArb = fc.constantFrom('', ' ', '  ', '\t', '\n')

describe('validation properties', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('accepts complete translation configs for every manifest provider', () => {
    fc.assert(fc.property(
      constantFromList(translationProviderIds),
      nonEmptyStringArb,
      nonEmptyStringArb,
      nonEmptyStringArb,
      (provider, apiKey, modelName, customBaseUrl) => {
        setActivePinia(createPinia())
        const store = useSettingsStore()
        store.setTranslationProvider(provider)
        store.updateTranslationService({
          apiKey,
          modelName,
          customBaseUrl,
        })

        expect(useValidation().validateBeforeTranslation('normal')).toBe(true)
      },
    ))
  })

  it('rejects translation providers with missing required credentials or base URLs', () => {
    fc.assert(fc.property(
      constantFromList(translationApiKeyProviderIds),
      emptyStringArb,
      nonEmptyStringArb,
      (provider, apiKey, modelName) => {
        setActivePinia(createPinia())
        const store = useSettingsStore()
        store.setTranslationProvider(provider)
        store.updateTranslationService({
          apiKey,
          modelName,
          customBaseUrl: 'https://provider.example/v1',
        })

        expect(useValidation().validateBeforeTranslation('normal')).toBe(false)
      },
    ))

    fc.assert(fc.property(
      constantFromList(translationBaseUrlProviderIds),
      emptyStringArb,
      (provider, customBaseUrl) => {
        setActivePinia(createPinia())
        const store = useSettingsStore()
        store.setTranslationProvider(provider)
        store.updateTranslationService({
          apiKey: 'key',
          modelName: 'model',
          customBaseUrl,
        })

        expect(useValidation().validateBeforeTranslation('normal')).toBe(false)
      },
    ))
  })

  it('accepts complete HQ configs and rejects missing required HQ credentials', () => {
    fc.assert(fc.property(
      constantFromList(hqTranslationProviderIds),
      nonEmptyStringArb,
      nonEmptyStringArb,
      nonEmptyStringArb,
      (provider, apiKey, modelName, customBaseUrl) => {
        setActivePinia(createPinia())
        const store = useSettingsStore()
        store.setHqProvider(provider)
        store.updateHqTranslation({
          apiKey,
          modelName,
          customBaseUrl,
        })

        expect(useValidation().validateBeforeTranslation('hq')).toBe(true)
      },
    ))

    fc.assert(fc.property(
      constantFromList(hqApiKeyProviderIds),
      emptyStringArb,
      (provider, apiKey) => {
        setActivePinia(createPinia())
        const store = useSettingsStore()
        store.setHqProvider(provider)
        store.updateHqTranslation({
          apiKey,
          modelName: 'model',
          customBaseUrl: 'https://provider.example/v1',
        })

        expect(useValidation().validateBeforeTranslation('hq')).toBe(false)
      },
    ))
  })

  it('validates proofreading only through the page-facing workflow entry', () => {
    const validRound: ProofreadingRound = {
      id: '11111111-1111-4111-8111-111111111111',
      name: '第一轮',
      provider: 'deepseek',
      apiKey: 'key',
      modelName: 'deepseek-chat',
      customBaseUrl: '',
      batchSize: 5,
      rpmLimit: 0,
      forceJsonOutput: false,
      prompt: '',
    }
    const validation = useValidation()

    expect(validation.validateBeforeTranslation('proofread', {
      proofreadingRounds: [validRound],
    })).toBe(true)
    expect(validation.validateBeforeTranslation('proofread', {
      proofreadingRounds: [],
    })).toBe(false)
  })

  it('uses backend credential summaries when secret inputs are intentionally blank', () => {
    const store = useSettingsStore()
    const validation = useValidation()

    store.setTranslationProvider('deepseek')
    store.updateTranslationService({ apiKey: '', modelName: 'deepseek-chat' })
    addStoredCredential(store, 'translation', 'deepseek')
    expect(validation.validateBeforeTranslation('normal')).toBe(true)

    store.setHqProvider('deepseek')
    store.updateHqTranslation({ apiKey: '', modelName: 'deepseek-chat' })
    addStoredCredential(store, 'hq', 'deepseek')
    expect(validation.validateBeforeTranslation('hq')).toBe(true)

    store.setOcrEngine('ai_vision')
    store.setAiVisionOcrProvider('gemini')
    store.updateAiVisionOcr({ apiKey: '', modelName: 'gemini-2.5-flash' })
    addStoredCredential(store, 'ai_vision_ocr', 'gemini')
    expect(validation.validateBeforeTranslation('ocr')).toBe(true)
  })

  it('validates local, Baidu, and AI vision OCR configurations', () => {
    const store = useSettingsStore()
    const validation = useValidation()

    store.setOcrEngine('manga_ocr')
    expect(validation.validateBeforeTranslation('ocr')).toBe(true)

    store.setOcrEngine('baidu_ocr')
    store.updateBaiduOcr({ apiKey: '', secretKey: '' })
    expect(validation.validateBeforeTranslation('ocr')).toBe(false)
    store.updateBaiduOcr({ apiKey: 'key', secretKey: 'secret' })
    expect(validation.validateBeforeTranslation('ocr')).toBe(true)

    store.setOcrEngine('ai_vision')
    store.setAiVisionOcrProvider('gemini')
    store.updateAiVisionOcr({ apiKey: '', modelName: 'model' })
    expect(validation.validateBeforeTranslation('ocr')).toBe(false)
    store.updateAiVisionOcr({ apiKey: 'key', modelName: 'model' })
    expect(validation.validateBeforeTranslation('ocr')).toBe(true)
  })
})
