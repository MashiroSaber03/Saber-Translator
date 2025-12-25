/**
 * 配置验证属性测试
 * 使用 fast-check 进行属性基测试，验证翻译配置验证的完整性和正确性
 *
 * Feature: vue-frontend-migration, Property 8: 配置验证完整性
 * Validates: Requirements 31.2, 31.3
 * 
 * Feature: vue-frontend-migration, Property 38: 翻译配置验证一致性
 * Validates: Requirements 31.2, 31.3
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useSettingsStore } from '@/stores/settingsStore'
import { useValidation } from '@/composables/useValidation'
import type { TranslationProvider, HqTranslationProvider, ProofreadingRound, OcrEngine } from '@/types/settings'

describe('配置验证属性测试', () => {
  // 模拟 localStorage
  let localStorageMock: Record<string, string> = {}

  beforeEach(() => {
    // 重置 localStorage 模拟
    localStorageMock = {}

    // 模拟 localStorage
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation((key: string) => {
      return localStorageMock[key] || null
    })

    vi.spyOn(Storage.prototype, 'setItem').mockImplementation((key: string, value: string) => {
      localStorageMock[key] = value
    })

    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation((key: string) => {
      delete localStorageMock[key]
    })

    // 重置 Pinia
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  // ============================================================
  // 生成器定义
  // ============================================================

  /** 需要 API Key 的服务商 */
  const providersRequiringApiKey: TranslationProvider[] = [
    'siliconflow',
    'deepseek',
    'volcano',
    'caiyun',
    'baidu',
    'youdao',
    'gemini',
    'custom_openai'
  ]

  /** 本地服务商 */
  const localProviders: TranslationProvider[] = ['ollama', 'sakura']

  /** 所有翻译服务商 */
  const allProviders: TranslationProvider[] = [...providersRequiringApiKey, ...localProviders]

  /** 高质量翻译服务商 */
  const hqProviders: HqTranslationProvider[] = [
    'siliconflow',
    'deepseek',
    'volcano',
    'gemini',
    'custom_openai'
  ]

  /** 生成有效的翻译服务商 */
  const validProviderArb = fc.constantFrom(...allProviders)

  /** 生成需要 API Key 的服务商 */
  const providerRequiringApiKeyArb = fc.constantFrom(...providersRequiringApiKey)

  /** 生成本地服务商 */
  const localProviderArb = fc.constantFrom(...localProviders)

  /** 生成高质量翻译服务商 */
  const hqProviderArb = fc.constantFrom(...hqProviders)

  /** 生成非空字符串 */
  const nonEmptyStringArb = fc.string({ minLength: 1, maxLength: 50 }).filter((s) => s.trim().length > 0)

  /** 生成空或空白字符串 */
  const emptyOrWhitespaceArb = fc.constantFrom('', ' ', '  ', '\t', '\n')

  /** 生成有效的校对轮次 */
  const validProofreadingRoundArb = fc.record({
    name: nonEmptyStringArb,
    provider: hqProviderArb,
    apiKey: nonEmptyStringArb,
    modelName: nonEmptyStringArb,
    customBaseUrl: fc.string(),
    batchSize: fc.integer({ min: 1, max: 10 }),
    sessionReset: fc.integer({ min: 1, max: 50 }),
    rpmLimit: fc.integer({ min: 0, max: 100 }),
    lowReasoning: fc.boolean(),
    noThinkingMethod: fc.constantFrom('gemini', 'volcano') as fc.Arbitrary<'gemini' | 'volcano'>,
    forceJsonOutput: fc.boolean(),
    prompt: fc.string()
  })

  /** 生成无效的校对轮次（缺少必填字段） */
  const invalidProofreadingRoundArb = fc.oneof(
    // 缺少服务商
    fc.record({
      name: nonEmptyStringArb,
      provider: fc.constant('' as HqTranslationProvider),
      apiKey: nonEmptyStringArb,
      modelName: nonEmptyStringArb,
      customBaseUrl: fc.string(),
      batchSize: fc.integer({ min: 1, max: 10 }),
      sessionReset: fc.integer({ min: 1, max: 50 }),
      rpmLimit: fc.integer({ min: 0, max: 100 }),
      lowReasoning: fc.boolean(),
      noThinkingMethod: fc.constantFrom('gemini', 'volcano') as fc.Arbitrary<'gemini' | 'volcano'>,
      forceJsonOutput: fc.boolean(),
      prompt: fc.string()
    }),
    // 缺少 API Key
    fc.record({
      name: nonEmptyStringArb,
      provider: hqProviderArb,
      apiKey: emptyOrWhitespaceArb,
      modelName: nonEmptyStringArb,
      customBaseUrl: fc.string(),
      batchSize: fc.integer({ min: 1, max: 10 }),
      sessionReset: fc.integer({ min: 1, max: 50 }),
      rpmLimit: fc.integer({ min: 0, max: 100 }),
      lowReasoning: fc.boolean(),
      noThinkingMethod: fc.constantFrom('gemini', 'volcano') as fc.Arbitrary<'gemini' | 'volcano'>,
      forceJsonOutput: fc.boolean(),
      prompt: fc.string()
    }),
    // 缺少模型名称
    fc.record({
      name: nonEmptyStringArb,
      provider: hqProviderArb,
      apiKey: nonEmptyStringArb,
      modelName: emptyOrWhitespaceArb,
      customBaseUrl: fc.string(),
      batchSize: fc.integer({ min: 1, max: 10 }),
      sessionReset: fc.integer({ min: 1, max: 50 }),
      rpmLimit: fc.integer({ min: 0, max: 100 }),
      lowReasoning: fc.boolean(),
      noThinkingMethod: fc.constantFrom('gemini', 'volcano') as fc.Arbitrary<'gemini' | 'volcano'>,
      forceJsonOutput: fc.boolean(),
      prompt: fc.string()
    })
  )

  // ============================================================
  // 普通翻译配置验证测试
  // ============================================================

  /**
   * Feature: vue-frontend-migration, Property 8: 配置验证完整性
   * Validates: Requirements 31.2, 31.3
   *
   * 对于任意完整的翻译配置，验证应当通过
   */
  it('完整的翻译配置应当验证通过', () => {
    fc.assert(
      fc.property(
        validProviderArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        (provider, apiKey, modelName, customBaseUrl) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const settingsStore = useSettingsStore()
          const { validateTranslationConfig } = useValidation()

          // 设置完整的翻译配置
          settingsStore.updateTranslationService({
            provider,
            apiKey,
            modelName,
            customBaseUrl
          })

          // 验证配置
          const result = validateTranslationConfig()

          // 完整配置应当验证通过
          return result.valid === true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 8: 配置验证完整性
   * Validates: Requirements 31.2, 31.3
   *
   * 对于需要 API Key 的服务商，缺少 API Key 时验证应当失败
   */
  it('需要 API Key 的服务商缺少 API Key 时验证应当失败', () => {
    fc.assert(
      fc.property(
        providerRequiringApiKeyArb,
        emptyOrWhitespaceArb,
        nonEmptyStringArb,
        (provider, emptyApiKey, modelName) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const settingsStore = useSettingsStore()
          const { validateTranslationConfig } = useValidation()

          // 设置缺少 API Key 的配置
          settingsStore.updateTranslationService({
            provider,
            apiKey: emptyApiKey,
            modelName
          })

          // 验证配置
          const result = validateTranslationConfig()

          // 缺少 API Key 应当验证失败
          return result.valid === false && result.message.includes('API Key')
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 8: 配置验证完整性
   * Validates: Requirements 31.2, 31.3
   *
   * 对于本地服务商，缺少模型名称时验证应当失败
   */
  it('本地服务商缺少模型名称时验证应当失败', () => {
    fc.assert(
      fc.property(localProviderArb, emptyOrWhitespaceArb, (provider, emptyModelName) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateTranslationConfig } = useValidation()

        // 设置缺少模型名称的配置
        settingsStore.updateTranslationService({
          provider,
          apiKey: '', // 本地服务商不需要 API Key
          modelName: emptyModelName
        })

        // 验证配置
        const result = validateTranslationConfig()

        // 缺少模型名称应当验证失败
        return result.valid === false && result.message.includes('模型名称')
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 8: 配置验证完整性
   * Validates: Requirements 31.2, 31.3
   *
   * 自定义 OpenAI 服务商缺少 Base URL 时验证应当失败
   */
  it('自定义 OpenAI 服务商缺少 Base URL 时验证应当失败', () => {
    fc.assert(
      fc.property(nonEmptyStringArb, nonEmptyStringArb, emptyOrWhitespaceArb, (apiKey, modelName, emptyBaseUrl) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateTranslationConfig } = useValidation()

        // 设置缺少 Base URL 的自定义 OpenAI 配置
        settingsStore.updateTranslationService({
          provider: 'custom_openai',
          apiKey,
          modelName,
          customBaseUrl: emptyBaseUrl
        })

        // 验证配置
        const result = validateTranslationConfig()

        // 缺少 Base URL 应当验证失败
        return result.valid === false && result.message.includes('Base URL')
      }),
      { numRuns: 100 }
    )
  })

  // ============================================================
  // 高质量翻译配置验证测试
  // ============================================================

  /**
   * Feature: vue-frontend-migration, Property 8: 配置验证完整性
   * Validates: Requirements 31.2, 31.3
   *
   * 对于任意完整的高质量翻译配置，验证应当通过
   */
  it('完整的高质量翻译配置应当验证通过', () => {
    fc.assert(
      fc.property(
        hqProviderArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        (provider, apiKey, modelName, customBaseUrl) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const settingsStore = useSettingsStore()
          const { validateHqTranslationConfig } = useValidation()

          // 设置完整的高质量翻译配置
          settingsStore.updateHqTranslation({
            provider,
            apiKey,
            modelName,
            customBaseUrl
          })

          // 验证配置
          const result = validateHqTranslationConfig()

          // 完整配置应当验证通过
          return result.valid === true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 8: 配置验证完整性
   * Validates: Requirements 31.2, 31.3
   *
   * 高质量翻译缺少 API Key 时验证应当失败
   */
  it('高质量翻译缺少 API Key 时验证应当失败', () => {
    fc.assert(
      fc.property(hqProviderArb, emptyOrWhitespaceArb, nonEmptyStringArb, (provider, emptyApiKey, modelName) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateHqTranslationConfig } = useValidation()

        // 设置缺少 API Key 的配置
        settingsStore.updateHqTranslation({
          provider,
          apiKey: emptyApiKey,
          modelName
        })

        // 验证配置
        const result = validateHqTranslationConfig()

        // 缺少 API Key 应当验证失败
        return result.valid === false && result.message.includes('API Key')
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 8: 配置验证完整性
   * Validates: Requirements 31.2, 31.3
   *
   * 高质量翻译缺少模型名称时验证应当失败
   */
  it('高质量翻译缺少模型名称时验证应当失败', () => {
    fc.assert(
      fc.property(hqProviderArb, nonEmptyStringArb, emptyOrWhitespaceArb, (provider, apiKey, emptyModelName) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateHqTranslationConfig } = useValidation()

        // 设置缺少模型名称的配置
        settingsStore.updateHqTranslation({
          provider,
          apiKey,
          modelName: emptyModelName
        })

        // 验证配置
        const result = validateHqTranslationConfig()

        // 缺少模型名称应当验证失败
        return result.valid === false && result.message.includes('模型名称')
      }),
      { numRuns: 100 }
    )
  })

  // ============================================================
  // AI 校对配置验证测试
  // ============================================================

  /**
   * Feature: vue-frontend-migration, Property 8: 配置验证完整性
   * Validates: Requirements 31.2, 31.3
   *
   * 对于任意完整的校对轮次配置，验证应当通过
   */
  it('完整的校对轮次配置应当验证通过', () => {
    fc.assert(
      fc.property(fc.array(validProofreadingRoundArb, { minLength: 1, maxLength: 5 }), (rounds) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const { validateProofreadingConfig } = useValidation()

        // 验证配置
        const result = validateProofreadingConfig(rounds as ProofreadingRound[])

        // 完整配置应当验证通过
        return result.valid === true
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 8: 配置验证完整性
   * Validates: Requirements 31.2, 31.3
   *
   * 空的校对轮次列表验证应当失败
   */
  it('空的校对轮次列表验证应当失败', () => {
    // 每次迭代重新创建 Pinia 实例
    setActivePinia(createPinia())
    localStorageMock = {}

    const { validateProofreadingConfig } = useValidation()

    // 验证空列表
    const result = validateProofreadingConfig([])

    // 空列表应当验证失败
    expect(result.valid).toBe(false)
    expect(result.message).toContain('至少一个校对轮次')
  })

  /**
   * Feature: vue-frontend-migration, Property 8: 配置验证完整性
   * Validates: Requirements 31.2, 31.3
   *
   * 包含无效轮次的校对配置验证应当失败
   */
  it('包含无效轮次的校对配置验证应当失败', () => {
    fc.assert(
      fc.property(invalidProofreadingRoundArb, (invalidRound) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const { validateProofreadingConfig } = useValidation()

        // 验证包含无效轮次的配置
        const result = validateProofreadingConfig([invalidRound as ProofreadingRound])

        // 无效配置应当验证失败
        return result.valid === false
      }),
      { numRuns: 100 }
    )
  })

  // ============================================================
  // 验证结果一致性测试
  // ============================================================

  /**
   * Feature: vue-frontend-migration, Property 8: 配置验证完整性
   * Validates: Requirements 31.2, 31.3
   *
   * 验证结果应当包含正确的缺失项列表
   */
  it('验证结果应当包含正确的缺失项列表', () => {
    fc.assert(
      fc.property(providerRequiringApiKeyArb, (provider) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateTranslationConfig } = useValidation()

        // 设置不完整的配置
        settingsStore.updateTranslationService({
          provider,
          apiKey: '',
          modelName: ''
        })

        // 验证配置
        const result = validateTranslationConfig()

        // 验证失败时应当有缺失项列表
        if (!result.valid) {
          return result.missingItems !== undefined && result.missingItems.length > 0
        }
        return true
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 8: 配置验证完整性
   * Validates: Requirements 31.2, 31.3
   *
   * 验证通过时不应有错误消息
   */
  it('验证通过时不应有错误消息', () => {
    fc.assert(
      fc.property(
        validProviderArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        nonEmptyStringArb,
        (provider, apiKey, modelName, customBaseUrl) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const settingsStore = useSettingsStore()
          const { validateTranslationConfig } = useValidation()

          // 设置完整的配置
          settingsStore.updateTranslationService({
            provider,
            apiKey,
            modelName,
            customBaseUrl
          })

          // 验证配置
          const result = validateTranslationConfig()

          // 验证通过时消息应为空
          if (result.valid) {
            return result.message === ''
          }
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  // ============================================================
  // OCR 配置验证测试
  // Feature: vue-frontend-migration, Property 38: 翻译配置验证一致性
  // Validates: Requirements 31.2, 31.3
  // ============================================================

  /** 不需要额外配置的 OCR 引擎 */
  const localOcrEngines: OcrEngine[] = ['manga_ocr', 'paddle_ocr']

  /** 生成本地 OCR 引擎 */
  const localOcrEngineArb = fc.constantFrom(...localOcrEngines)

  /**
   * Feature: vue-frontend-migration, Property 38: 翻译配置验证一致性
   * Validates: Requirements 31.2, 31.3
   *
   * 本地 OCR 引擎（MangaOCR、PaddleOCR）不需要额外配置，验证应当通过
   */
  it('本地 OCR 引擎验证应当通过', () => {
    fc.assert(
      fc.property(localOcrEngineArb, (engine) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        // 设置本地 OCR 引擎
        settingsStore.setOcrEngine(engine)

        // 验证配置
        const result = validateOcrConfig()

        // 本地引擎应当验证通过
        return result.valid === true
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 38: 翻译配置验证一致性
   * Validates: Requirements 31.2, 31.3
   *
   * 百度 OCR 缺少 API Key 时验证应当失败
   */
  it('百度 OCR 缺少 API Key 时验证应当失败', () => {
    fc.assert(
      fc.property(emptyOrWhitespaceArb, nonEmptyStringArb, (emptyApiKey, secretKey) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        // 设置百度 OCR 引擎，但缺少 API Key
        settingsStore.setOcrEngine('baidu_ocr')
        settingsStore.updateBaiduOcr({
          apiKey: emptyApiKey,
          secretKey: secretKey
        })

        // 验证配置
        const result = validateOcrConfig()

        // 缺少 API Key 应当验证失败
        return result.valid === false && result.message.includes('API Key')
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 38: 翻译配置验证一致性
   * Validates: Requirements 31.2, 31.3
   *
   * 百度 OCR 缺少 Secret Key 时验证应当失败
   */
  it('百度 OCR 缺少 Secret Key 时验证应当失败', () => {
    fc.assert(
      fc.property(nonEmptyStringArb, emptyOrWhitespaceArb, (apiKey, emptySecretKey) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        // 设置百度 OCR 引擎，但缺少 Secret Key
        settingsStore.setOcrEngine('baidu_ocr')
        settingsStore.updateBaiduOcr({
          apiKey: apiKey,
          secretKey: emptySecretKey
        })

        // 验证配置
        const result = validateOcrConfig()

        // 缺少 Secret Key 应当验证失败
        return result.valid === false && result.message.includes('Secret Key')
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 38: 翻译配置验证一致性
   * Validates: Requirements 31.2, 31.3
   *
   * 完整的百度 OCR 配置验证应当通过
   */
  it('完整的百度 OCR 配置验证应当通过', () => {
    fc.assert(
      fc.property(nonEmptyStringArb, nonEmptyStringArb, (apiKey, secretKey) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        // 设置完整的百度 OCR 配置
        settingsStore.setOcrEngine('baidu_ocr')
        settingsStore.updateBaiduOcr({
          apiKey: apiKey,
          secretKey: secretKey
        })

        // 验证配置
        const result = validateOcrConfig()

        // 完整配置应当验证通过
        return result.valid === true
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 38: 翻译配置验证一致性
   * Validates: Requirements 31.2, 31.3
   *
   * AI 视觉 OCR 缺少 API Key 时验证应当失败
   */
  it('AI 视觉 OCR 缺少 API Key 时验证应当失败', () => {
    fc.assert(
      fc.property(emptyOrWhitespaceArb, nonEmptyStringArb, (emptyApiKey, modelName) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        // 设置 AI 视觉 OCR 引擎，但缺少 API Key
        settingsStore.setOcrEngine('ai_vision')
        settingsStore.updateAiVisionOcr({
          provider: 'gemini',
          apiKey: emptyApiKey,
          modelName: modelName
        })

        // 验证配置
        const result = validateOcrConfig()

        // 缺少 API Key 应当验证失败
        return result.valid === false && result.message.includes('API Key')
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 38: 翻译配置验证一致性
   * Validates: Requirements 31.2, 31.3
   *
   * AI 视觉 OCR 缺少模型名称时验证应当失败
   */
  it('AI 视觉 OCR 缺少模型名称时验证应当失败', () => {
    fc.assert(
      fc.property(nonEmptyStringArb, emptyOrWhitespaceArb, (apiKey, emptyModelName) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        // 设置 AI 视觉 OCR 引擎，但缺少模型名称
        settingsStore.setOcrEngine('ai_vision')
        settingsStore.updateAiVisionOcr({
          provider: 'gemini',
          apiKey: apiKey,
          modelName: emptyModelName
        })

        // 验证配置
        const result = validateOcrConfig()

        // 缺少模型名称应当验证失败
        return result.valid === false && result.message.includes('模型名称')
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 38: 翻译配置验证一致性
   * Validates: Requirements 31.2, 31.3
   *
   * 完整的 AI 视觉 OCR 配置验证应当通过
   */
  it('完整的 AI 视觉 OCR 配置验证应当通过', () => {
    fc.assert(
      fc.property(nonEmptyStringArb, nonEmptyStringArb, (apiKey, modelName) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateOcrConfig } = useValidation()

        // 设置完整的 AI 视觉 OCR 配置
        settingsStore.setOcrEngine('ai_vision')
        settingsStore.updateAiVisionOcr({
          provider: 'gemini',
          apiKey: apiKey,
          modelName: modelName
        })

        // 验证配置
        const result = validateOcrConfig()

        // 完整配置应当验证通过
        return result.valid === true
      }),
      { numRuns: 100 }
    )
  })

  // ============================================================
  // 缺失项列表生成测试
  // Feature: vue-frontend-migration, Property 38: 翻译配置验证一致性
  // Validates: Requirements 31.2, 31.3
  // ============================================================

  /**
   * Feature: vue-frontend-migration, Property 38: 翻译配置验证一致性
   * Validates: Requirements 31.2, 31.3
   *
   * 缺失项列表应当正确反映缺失的配置
   */
  it('缺失项列表应当正确反映缺失的配置', () => {
    fc.assert(
      fc.property(providerRequiringApiKeyArb, (provider) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateTranslationConfig } = useValidation()

        // 设置服务商但不设置 API Key 和模型名称
        settingsStore.updateTranslationService({
          provider,
          apiKey: '',
          modelName: ''
        })

        // 验证配置
        const result = validateTranslationConfig()

        // 验证失败时缺失项列表应当非空
        if (!result.valid && result.missingItems) {
          // 缺失项应当包含 API Key 相关内容
          return result.missingItems.some(item => item.includes('API Key'))
        }
        return true
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 38: 翻译配置验证一致性
   * Validates: Requirements 31.2, 31.3
   *
   * 各服务商必填字段验证正确
   */
  it('各服务商必填字段验证正确', () => {
    fc.assert(
      fc.property(validProviderArb, (provider) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const settingsStore = useSettingsStore()
        const { validateTranslationConfig, requiresApiKey, isLocalProvider } = useValidation()

        // 设置服务商但不设置任何配置
        settingsStore.updateTranslationService({
          provider,
          apiKey: '',
          modelName: '',
          customBaseUrl: ''
        })

        // 验证配置
        const result = validateTranslationConfig()

        // 需要 API Key 的服务商应当验证失败
        if (requiresApiKey(provider)) {
          return result.valid === false
        }
        // 本地服务商缺少模型名称也应当验证失败
        if (isLocalProvider(provider)) {
          return result.valid === false
        }
        return true
      }),
      { numRuns: 100 }
    )
  })
})
