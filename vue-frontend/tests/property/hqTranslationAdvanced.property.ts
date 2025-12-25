/**
 * 高质量翻译高级选项属性测试
 * 使用 fast-check 进行属性基测试，验证高质量翻译高级选项的一致性
 *
 * Feature: vue-frontend-migration, Property 41: 高质量翻译高级选项一致性
 * Validates: Requirements 4.9
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useSettingsStore } from '@/stores/settingsStore'
import { STORAGE_KEY_TRANSLATION_SETTINGS } from '@/constants'
import type { NoThinkingMethod, HqTranslationProvider } from '@/types/settings'

describe('高质量翻译高级选项属性测试', () => {
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
   * 生成有效的高质量翻译服务商
   */
  const validHqProviderArb = fc.constantFrom(
    'siliconflow',
    'deepseek',
    'volcano',
    'gemini',
    'custom_openai'
  ) as fc.Arbitrary<HqTranslationProvider>

  /**
   * 生成有效的取消思考方法
   */
  const validNoThinkingMethodArb = fc.constantFrom('gemini', 'volcano') as fc.Arbitrary<NoThinkingMethod>

  /**
   * Feature: vue-frontend-migration, Property 41: 高质量翻译高级选项一致性
   * Validates: Requirements 4.9
   *
   * 对于任意流式调用开关状态，设置后应当正确保存和恢复
   */
  it('流式调用开关状态持久化一致性', () => {
    fc.assert(
      fc.property(fc.boolean(), (useStream) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const store = useSettingsStore()

        // 设置流式调用开关
        store.setHqUseStream(useStream)

        // 验证设置已更新
        if (store.settings.hqTranslation.useStream !== useStream) return false

        // 验证设置已保存到 localStorage
        const savedData = localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS]
        if (!savedData) return false

        // 创建新的 store 实例并加载设置
        setActivePinia(createPinia())
        const newStore = useSettingsStore()
        newStore.loadFromStorage()

        // 验证设置已正确恢复
        return newStore.settings.hqTranslation.useStream === useStream
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 41: 高质量翻译高级选项一致性
   * Validates: Requirements 4.9
   *
   * 对于任意取消思考方法，设置后应当正确保存和恢复
   */
  it('取消思考方法持久化一致性', () => {
    fc.assert(
      fc.property(validNoThinkingMethodArb, (noThinkingMethod) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const store = useSettingsStore()

        // 设置取消思考方法
        store.setHqNoThinkingMethod(noThinkingMethod)

        // 验证设置已更新
        if (store.settings.hqTranslation.noThinkingMethod !== noThinkingMethod) return false

        // 验证设置已保存到 localStorage
        const savedData = localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS]
        if (!savedData) return false

        // 创建新的 store 实例并加载设置
        setActivePinia(createPinia())
        const newStore = useSettingsStore()
        newStore.loadFromStorage()

        // 验证设置已正确恢复
        return newStore.settings.hqTranslation.noThinkingMethod === noThinkingMethod
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 41: 高质量翻译高级选项一致性
   * Validates: Requirements 4.9
   *
   * 对于任意强制JSON输出状态，设置后应当正确保存和恢复
   */
  it('强制JSON输出状态持久化一致性', () => {
    fc.assert(
      fc.property(fc.boolean(), (forceJsonOutput) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const store = useSettingsStore()

        // 设置强制JSON输出
        store.setHqForceJsonOutput(forceJsonOutput)

        // 验证设置已更新
        if (store.settings.hqTranslation.forceJsonOutput !== forceJsonOutput) return false

        // 验证设置已保存到 localStorage
        const savedData = localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS]
        if (!savedData) return false

        // 创建新的 store 实例并加载设置
        setActivePinia(createPinia())
        const newStore = useSettingsStore()
        newStore.loadFromStorage()

        // 验证设置已正确恢复
        return newStore.settings.hqTranslation.forceJsonOutput === forceJsonOutput
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 41: 高质量翻译高级选项一致性
   * Validates: Requirements 4.9
   *
   * 对于任意高级选项组合，设置后应当正确保存和恢复
   */
  it('高级选项组合持久化一致性', () => {
    fc.assert(
      fc.property(
        fc.boolean(),
        validNoThinkingMethodArb,
        fc.boolean(),
        fc.boolean(),
        (useStream, noThinkingMethod, forceJsonOutput, lowReasoning) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const store = useSettingsStore()

          // 设置所有高级选项
          store.updateHqTranslation({
            useStream,
            noThinkingMethod,
            forceJsonOutput,
            lowReasoning
          })

          // 验证设置已更新
          if (store.settings.hqTranslation.useStream !== useStream) return false
          if (store.settings.hqTranslation.noThinkingMethod !== noThinkingMethod) return false
          if (store.settings.hqTranslation.forceJsonOutput !== forceJsonOutput) return false
          if (store.settings.hqTranslation.lowReasoning !== lowReasoning) return false

          // 验证设置已保存到 localStorage
          const savedData = localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS]
          if (!savedData) return false

          // 创建新的 store 实例并加载设置
          setActivePinia(createPinia())
          const newStore = useSettingsStore()
          newStore.loadFromStorage()

          // 验证设置已正确恢复
          return (
            newStore.settings.hqTranslation.useStream === useStream &&
            newStore.settings.hqTranslation.noThinkingMethod === noThinkingMethod &&
            newStore.settings.hqTranslation.forceJsonOutput === forceJsonOutput &&
            newStore.settings.hqTranslation.lowReasoning === lowReasoning
          )
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 41: 高质量翻译高级选项一致性
   * Validates: Requirements 4.9
   *
   * 切换服务商时高级选项应当正确保存和恢复
   */
  it('服务商切换时高级选项保存恢复一致性', () => {
    fc.assert(
      fc.property(
        validHqProviderArb,
        validHqProviderArb,
        fc.boolean(),
        validNoThinkingMethodArb,
        fc.boolean(),
        (provider1, provider2, useStream, noThinkingMethod, forceJsonOutput) => {
          // 跳过相同服务商的情况
          if (provider1 === provider2) return true

          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const store = useSettingsStore()

          // 设置第一个服务商的高级选项
          store.setHqProvider(provider1)
          store.updateHqTranslation({
            useStream,
            noThinkingMethod,
            forceJsonOutput
          })

          // 记录第一个服务商的设置
          const provider1UseStream = store.settings.hqTranslation.useStream
          const provider1NoThinkingMethod = store.settings.hqTranslation.noThinkingMethod
          const provider1ForceJsonOutput = store.settings.hqTranslation.forceJsonOutput

          // 切换到第二个服务商
          store.setHqProvider(provider2)

          // 设置第二个服务商的不同高级选项
          store.updateHqTranslation({
            useStream: !useStream,
            noThinkingMethod: noThinkingMethod === 'gemini' ? 'volcano' : 'gemini',
            forceJsonOutput: !forceJsonOutput
          })

          // 切换回第一个服务商
          store.setHqProvider(provider1)

          // 验证第一个服务商的设置已恢复
          return (
            store.settings.hqTranslation.useStream === provider1UseStream &&
            store.settings.hqTranslation.noThinkingMethod === provider1NoThinkingMethod &&
            store.settings.hqTranslation.forceJsonOutput === provider1ForceJsonOutput
          )
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 41: 高质量翻译高级选项一致性
   * Validates: Requirements 4.9
   *
   * 取消思考方法参数生成正确性
   */
  it('取消思考方法参数生成正确性', () => {
    fc.assert(
      fc.property(validNoThinkingMethodArb, fc.boolean(), (noThinkingMethod, lowReasoning) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const store = useSettingsStore()

        // 设置取消思考方法和低推理模式
        store.updateHqTranslation({
          noThinkingMethod,
          lowReasoning
        })

        // 验证设置已正确保存
        const savedNoThinkingMethod = store.settings.hqTranslation.noThinkingMethod
        const savedLowReasoning = store.settings.hqTranslation.lowReasoning

        // 验证取消思考方法值有效
        const validMethods: NoThinkingMethod[] = ['gemini', 'volcano']
        if (!validMethods.includes(savedNoThinkingMethod)) return false

        // 验证低推理模式是布尔值
        if (typeof savedLowReasoning !== 'boolean') return false

        return (
          savedNoThinkingMethod === noThinkingMethod &&
          savedLowReasoning === lowReasoning
        )
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 41: 高质量翻译高级选项一致性
   * Validates: Requirements 4.9
   *
   * 强制JSON输出默认值正确性
   */
  it('强制JSON输出默认值正确性', () => {
    // 每次迭代重新创建 Pinia 实例
    setActivePinia(createPinia())
    localStorageMock = {}

    const store = useSettingsStore()

    // 验证默认值为 true（根据设计文档要求）
    expect(store.settings.hqTranslation.forceJsonOutput).toBe(true)
  })

  /**
   * Feature: vue-frontend-migration, Property 41: 高质量翻译高级选项一致性
   * Validates: Requirements 4.9
   *
   * 流式调用默认值正确性
   */
  it('流式调用默认值正确性', () => {
    // 每次迭代重新创建 Pinia 实例
    setActivePinia(createPinia())
    localStorageMock = {}

    const store = useSettingsStore()

    // 验证默认值为 false
    expect(store.settings.hqTranslation.useStream).toBe(false)
  })

  /**
   * Feature: vue-frontend-migration, Property 41: 高质量翻译高级选项一致性
   * Validates: Requirements 4.9
   *
   * 取消思考方法默认值正确性
   */
  it('取消思考方法默认值正确性', () => {
    // 每次迭代重新创建 Pinia 实例
    setActivePinia(createPinia())
    localStorageMock = {}

    const store = useSettingsStore()

    // 验证默认值为 'gemini'
    expect(store.settings.hqTranslation.noThinkingMethod).toBe('gemini')
  })
})
