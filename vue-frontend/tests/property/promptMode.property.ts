/**
 * 提示词模式切换属性测试
 * 使用 fast-check 进行属性基测试，验证提示词模式切换的一致性
 *
 * Feature: frontend-behavior, Property 21: 提示词模式切换一致性
 * Validates: Requirements 22.1, 22.2
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useSettingsStore } from '@/stores/settings'
import {
  DEFAULT_TRANSLATE_PROMPT,
  DEFAULT_TRANSLATE_JSON_PROMPT,
  DEFAULT_AI_VISION_OCR_PROMPT,
  DEFAULT_AI_VISION_OCR_JSON_PROMPT,
  STORAGE_KEY_TRANSLATION_SETTINGS
} from '@/constants'

describe('提示词模式切换属性测试', () => {
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

  /**
   * Feature: frontend-behavior, Property 21: 提示词模式切换一致性
   * Validates: Requirements 22.1, 22.2
   *
   * 对于任意初始模式状态，切换翻译提示词模式后状态应正确更新
   */
  it('翻译提示词模式切换状态一致性', () => {
    fc.assert(
      fc.property(fc.boolean(), (initialJsonMode) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const store = useSettingsStore()

        // 设置初始模式
        store.updateTranslationService({ forceJsonOutput: initialJsonMode })

        // 验证初始模式
        if (store.settings.translation.openaiOptions.request.forceJsonOutput !== initialJsonMode) return false

        // 切换模式
        store.updateTranslationService({ forceJsonOutput: !initialJsonMode })

        // 验证模式已切换
        const expectedMode = !initialJsonMode
        if (store.settings.translation.openaiOptions.request.forceJsonOutput !== expectedMode) return false

        // 再次切换，应该回到初始模式
        store.updateTranslationService({ forceJsonOutput: initialJsonMode })
        return store.settings.translation.openaiOptions.request.forceJsonOutput === initialJsonMode
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: frontend-behavior, Property 21: 提示词模式切换一致性
   * Validates: Requirements 22.1, 22.2
   *
   * 对于任意初始模式状态，切换AI视觉OCR提示词模式后状态应正确更新
   */
  it('AI视觉OCR提示词模式切换状态一致性', () => {
    fc.assert(
      fc.property(fc.boolean(), (initialJsonMode) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const store = useSettingsStore()

        // 设置初始模式
        store.updateAiVisionOcr({ forceJsonOutput: initialJsonMode })

        // 验证初始模式
        if (store.settings.aiVisionOcr.openaiOptions.request.forceJsonOutput !== initialJsonMode) return false

        // 切换模式
        store.updateAiVisionOcr({ forceJsonOutput: !initialJsonMode })

        // 验证模式已切换
        const expectedMode = !initialJsonMode
        if (store.settings.aiVisionOcr.openaiOptions.request.forceJsonOutput !== expectedMode) return false

        // 再次切换，应该回到初始模式
        store.updateAiVisionOcr({ forceJsonOutput: initialJsonMode })
        return store.settings.aiVisionOcr.openaiOptions.request.forceJsonOutput === initialJsonMode
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: frontend-behavior, Property 21: 提示词模式切换一致性
   * Validates: Requirements 22.1, 22.2
   *
   * 翻译提示词模式持久化往返一致性
   */
  it('翻译提示词模式持久化往返一致性', () => {
    fc.assert(
      fc.property(fc.boolean(), (forceJsonOutput) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const store = useSettingsStore()

        // 设置模式
        store.updateTranslationService({ forceJsonOutput })

        // 验证设置已保存到 localStorage
        const savedData = localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS]
        if (!savedData) return false

        // 创建新的 store 实例并加载设置
        setActivePinia(createPinia())
        const newStore = useSettingsStore()
        newStore.loadFromStorage()

        // 验证模式已正确恢复
        return newStore.settings.translation.openaiOptions.request.forceJsonOutput === forceJsonOutput
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: frontend-behavior, Property 21: 提示词模式切换一致性
   * Validates: Requirements 22.1, 22.2
   *
   * AI视觉OCR提示词模式持久化往返一致性
   */
  it('AI视觉OCR提示词模式持久化往返一致性', () => {
    fc.assert(
      fc.property(fc.boolean(), (forceJsonOutput) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const store = useSettingsStore()

        // 设置模式
        store.updateAiVisionOcr({ forceJsonOutput })

        // 验证设置已保存到 localStorage
        const savedData = localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS]
        if (!savedData) return false

        // 创建新的 store 实例并加载设置
        setActivePinia(createPinia())
        const newStore = useSettingsStore()
        newStore.loadFromStorage()

        // 验证模式已正确恢复
        return newStore.settings.aiVisionOcr.openaiOptions.request.forceJsonOutput === forceJsonOutput
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: frontend-behavior, Property 21: 提示词模式切换一致性
   * Validates: Requirements 22.1, 22.2
   *
   * 默认提示词常量存在且不为空
   */
  it('默认提示词常量存在且不为空', () => {
    // 验证翻译提示词常量
    expect(DEFAULT_TRANSLATE_PROMPT).toBeDefined()
    expect(DEFAULT_TRANSLATE_PROMPT.length).toBeGreaterThan(0)
    
    expect(DEFAULT_TRANSLATE_JSON_PROMPT).toBeDefined()
    expect(DEFAULT_TRANSLATE_JSON_PROMPT.length).toBeGreaterThan(0)
    
    // 验证AI视觉OCR提示词常量
    expect(DEFAULT_AI_VISION_OCR_PROMPT).toBeDefined()
    expect(DEFAULT_AI_VISION_OCR_PROMPT.length).toBeGreaterThan(0)
    
    expect(DEFAULT_AI_VISION_OCR_JSON_PROMPT).toBeDefined()
    expect(DEFAULT_AI_VISION_OCR_JSON_PROMPT.length).toBeGreaterThan(0)
    
    // 验证普通模式和JSON模式的提示词不同
    expect(DEFAULT_TRANSLATE_PROMPT).not.toBe(DEFAULT_TRANSLATE_JSON_PROMPT)
    expect(DEFAULT_AI_VISION_OCR_PROMPT).not.toBe(DEFAULT_AI_VISION_OCR_JSON_PROMPT)
  })

  /**
   * Feature: frontend-behavior, Property 21: 提示词模式切换一致性
   * Validates: Requirements 22.1, 22.2
   *
   * 提示词内容保存和加载一致性
   */
  it('提示词内容保存和加载一致性', () => {
    fc.assert(
      fc.property(
        fc.string({ minLength: 1, maxLength: 500 }),
        fc.string({ minLength: 1, maxLength: 500 }),
        (translatePrompt, aiVisionPrompt) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const store = useSettingsStore()

          // 设置提示词内容
          store.setTranslatePrompt(translatePrompt)
          store.updateAiVisionOcr({ prompt: aiVisionPrompt })

          // 验证设置已保存到 localStorage
          const savedData = localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS]
          if (!savedData) return false

          // 创建新的 store 实例并加载设置
          setActivePinia(createPinia())
          const newStore = useSettingsStore()
          newStore.loadFromStorage()

          // 验证提示词内容已正确恢复
          return (
            newStore.settings.translatePrompt === translatePrompt &&
            newStore.settings.aiVisionOcr.prompt === aiVisionPrompt
          )
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: frontend-behavior, Property 21: 提示词模式切换一致性
   * Validates: Requirements 22.1, 22.2
   *
   * 模式切换不影响其他设置
   */
  it('模式切换不影响其他设置', () => {
    fc.assert(
      fc.property(
        fc.boolean(),
        fc.string({ minLength: 1, maxLength: 50 }),
        fc.string({ minLength: 1, maxLength: 50 }),
        (forceJsonOutput, apiKey, modelName) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const store = useSettingsStore()

          // 设置其他配置
          store.updateTranslationService({
            apiKey,
            modelName,
            forceJsonOutput: false
          })

          // 记录其他配置
          const originalApiKey = store.settings.translation.apiKey
          const originalModelName = store.settings.translation.modelName

          // 切换模式
          store.updateTranslationService({ forceJsonOutput })

          // 验证其他配置未变
          return (
            store.settings.translation.apiKey === originalApiKey &&
            store.settings.translation.modelName === originalModelName &&
            store.settings.translation.openaiOptions.request.forceJsonOutput === forceJsonOutput
          )
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: frontend-behavior, Property 21: 提示词模式切换一致性
   * Validates: Requirements 22.1, 22.2
   *
   * 翻译和AI视觉OCR模式独立切换
   */
  it('翻译和AI视觉OCR模式独立切换', () => {
    fc.assert(
      fc.property(
        fc.boolean(),
        fc.boolean(),
        (translateJsonMode, aiVisionJsonMode) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const store = useSettingsStore()

          // 分别设置两种模式
          store.updateTranslationService({ forceJsonOutput: translateJsonMode })
          store.updateAiVisionOcr({ forceJsonOutput: aiVisionJsonMode })

          // 验证两种模式独立
          if (store.settings.translation.openaiOptions.request.forceJsonOutput !== translateJsonMode) return false
          if (store.settings.aiVisionOcr.openaiOptions.request.forceJsonOutput !== aiVisionJsonMode) return false

          // 切换翻译模式，不影响AI视觉OCR模式
          store.updateTranslationService({ forceJsonOutput: !translateJsonMode })
          if (store.settings.aiVisionOcr.openaiOptions.request.forceJsonOutput !== aiVisionJsonMode) return false

          // 切换AI视觉OCR模式，不影响翻译模式
          store.updateAiVisionOcr({ forceJsonOutput: !aiVisionJsonMode })
          return store.settings.translation.openaiOptions.request.forceJsonOutput === !translateJsonMode
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: frontend-behavior, Property 40: 提示词模式切换一致性
   * Validates: Requirements 22.1, 16.2
   *
   * setTranslatePromptMode 函数切换模式后默认提示词正确更新
   */
  it('setTranslatePromptMode 切换模式后默认提示词正确更新', () => {
    fc.assert(
      fc.property(fc.boolean(), (forceJsonOutput) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const store = useSettingsStore()

        // 使用 setTranslatePromptMode 切换模式
        store.setTranslatePromptMode(forceJsonOutput)

        // 验证模式状态正确
        if (store.settings.translation.openaiOptions.request.forceJsonOutput !== forceJsonOutput) return false

        // 验证提示词内容正确更新
        const expectedPrompt = forceJsonOutput ? DEFAULT_TRANSLATE_JSON_PROMPT : DEFAULT_TRANSLATE_PROMPT
        return store.settings.translatePrompt === expectedPrompt
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: frontend-behavior, Property 40: 提示词模式切换一致性
   * Validates: Requirements 22.1, 16.2
   *
   * setAiVisionOcrPromptMode 函数切换模式后默认提示词正确更新
   */
  it('setAiVisionOcrPromptMode 切换模式后默认提示词正确更新', () => {
    fc.assert(
      fc.property(fc.boolean(), (forceJsonOutput) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const store = useSettingsStore()

        // 使用 setAiVisionOcrPromptMode 切换模式
        store.setAiVisionOcrPromptMode(forceJsonOutput)

        // 验证模式状态正确
        if (store.settings.aiVisionOcr.openaiOptions.request.forceJsonOutput !== forceJsonOutput) return false

        // 验证提示词内容正确更新
        const expectedPrompt = forceJsonOutput ? DEFAULT_AI_VISION_OCR_JSON_PROMPT : DEFAULT_AI_VISION_OCR_PROMPT
        return store.settings.aiVisionOcr.prompt === expectedPrompt
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: frontend-behavior, Property 40: 提示词模式切换一致性
   * Validates: Requirements 22.1, 16.2
   *
   * 提示词模式切换后状态持久化正确
   */
  it('提示词模式切换后状态持久化正确', () => {
    fc.assert(
      fc.property(
        fc.boolean(),
        fc.boolean(),
        (translateJsonMode, aiVisionJsonMode) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const store = useSettingsStore()

          // 使用专用函数切换模式
          store.setTranslatePromptMode(translateJsonMode)
          store.setAiVisionOcrPromptMode(aiVisionJsonMode)

          // 验证设置已保存到 localStorage
          const savedData = localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS]
          if (!savedData) return false

          // 创建新的 store 实例并加载设置
          setActivePinia(createPinia())
          const newStore = useSettingsStore()
          newStore.loadFromStorage()

          // 验证模式状态已正确恢复
          if (newStore.settings.translation.openaiOptions.request.forceJsonOutput !== translateJsonMode) return false
          if (newStore.settings.aiVisionOcr.openaiOptions.request.forceJsonOutput !== aiVisionJsonMode) return false

          // 验证提示词内容已正确恢复
          const expectedTranslatePrompt = translateJsonMode ? DEFAULT_TRANSLATE_JSON_PROMPT : DEFAULT_TRANSLATE_PROMPT
          const expectedAiVisionPrompt = aiVisionJsonMode ? DEFAULT_AI_VISION_OCR_JSON_PROMPT : DEFAULT_AI_VISION_OCR_PROMPT

          return (
            newStore.settings.translatePrompt === expectedTranslatePrompt &&
            newStore.settings.aiVisionOcr.prompt === expectedAiVisionPrompt
          )
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: frontend-behavior, Property 40: 提示词模式切换一致性
   * Validates: Requirements 22.1, 16.2
   *
   * 模式切换往返一致性（切换两次回到原始状态）
   */
  it('模式切换往返一致性', () => {
    fc.assert(
      fc.property(fc.boolean(), (initialMode) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const store = useSettingsStore()

        // 设置初始模式
        store.setTranslatePromptMode(initialMode)
        const initialPrompt = store.settings.translatePrompt

        // 切换到相反模式
        store.setTranslatePromptMode(!initialMode)

        // 验证提示词已变化
        if (store.settings.translatePrompt === initialPrompt) return false

        // 切换回初始模式
        store.setTranslatePromptMode(initialMode)

        // 验证提示词恢复
        return store.settings.translatePrompt === initialPrompt
      }),
      { numRuns: 100 }
    )
  })
})
