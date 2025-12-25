/**
 * 设置状态管理属性测试
 * 使用 fast-check 进行属性基测试，验证设置持久化和主题切换的一致性
 *
 * Feature: vue-frontend-migration, Property 4: 设置持久化往返一致性
 * Feature: vue-frontend-migration, Property 5: 主题切换状态一致性
 * Validates: Requirements 7.2, 7.3, 10.1, 10.2
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useSettingsStore } from '@/stores/settingsStore'
import { STORAGE_KEY_TRANSLATION_SETTINGS, STORAGE_KEY_THEME } from '@/constants'

describe('设置状态管理属性测试', () => {
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
   * 生成有效的字号
   */
  const validFontSizeArb = fc.integer({ min: 10, max: 100 })

  /**
   * 生成有效的颜色值
   */
  const validColorArb = fc.hexaString({ minLength: 6, maxLength: 6 }).map((hex) => `#${hex}`)

  /**
   * 生成有效的字体名称
   */
  const validFontFamilyArb = fc.constantFrom(
    'fonts/STSONG.TTF',
    'fonts/SimHei.ttf',
    'fonts/msyh.ttc',
    'Arial',
    'sans-serif'
  )

  /**
   * 生成有效的排版方向
   */
  const validLayoutDirectionArb = fc.constantFrom('auto', 'vertical', 'horizontal')

  /**
   * 生成有效的修复方式
   */
  const validInpaintMethodArb = fc.constantFrom('solid', 'lama_mpe', 'litelama')

  /**
   * Feature: vue-frontend-migration, Property 4: 设置持久化往返一致性
   * Validates: Requirements 7.2, 7.3
   *
   * 对于任意有效的文字样式设置，保存到 localStorage 后再读取应当得到等价的设置
   */
  it('文字样式设置持久化往返一致性', () => {
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
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const store = useSettingsStore()

          // 更新文字样式设置
          store.updateTextStyle({
            fontSize,
            textColor,
            fillColor,
            fontFamily,
            layoutDirection: layoutDirection as 'auto' | 'vertical' | 'horizontal',
            inpaintMethod: inpaintMethod as 'solid' | 'lama_mpe' | 'litelama',
            strokeEnabled,
            strokeWidth
          })

          // 验证设置已保存到 localStorage
          const savedData = localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS]
          if (!savedData) return false

          // 创建新的 store 实例并加载设置
          setActivePinia(createPinia())
          const newStore = useSettingsStore()
          newStore.loadFromStorage()

          // 验证设置已正确恢复
          return (
            newStore.settings.textStyle.fontSize === fontSize &&
            newStore.settings.textStyle.textColor === textColor &&
            newStore.settings.textStyle.fillColor === fillColor &&
            newStore.settings.textStyle.fontFamily === fontFamily &&
            newStore.settings.textStyle.layoutDirection === layoutDirection &&
            newStore.settings.textStyle.inpaintMethod === inpaintMethod &&
            newStore.settings.textStyle.strokeEnabled === strokeEnabled &&
            newStore.settings.textStyle.strokeWidth === strokeWidth
          )
        }
      ),
      { numRuns: 100 }
    )
  })


  /**
   * Feature: vue-frontend-migration, Property 4: 设置持久化往返一致性
   * Validates: Requirements 7.2, 7.3
   *
   * 对于任意有效的OCR设置，保存到 localStorage 后再读取应当得到等价的设置
   */
  it('OCR设置持久化往返一致性', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('manga_ocr', 'paddle_ocr', 'baidu_ocr', 'ai_vision'),
        fc.constantFrom('ja', 'zh', 'en', 'ko'),
        fc.constantFrom('ctd', 'yolo', 'yolov5', 'default'),
        (ocrEngine, sourceLanguage, textDetector) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const store = useSettingsStore()

          // 更新OCR设置
          store.setOcrEngine(ocrEngine as 'manga_ocr' | 'paddle_ocr' | 'baidu_ocr' | 'ai_vision')
          store.updateSettings({ sourceLanguage })
          store.setTextDetector(textDetector as 'ctd' | 'yolo' | 'yolov5' | 'default')

          // 验证设置已保存到 localStorage
          const savedData = localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS]
          if (!savedData) return false

          // 创建新的 store 实例并加载设置
          setActivePinia(createPinia())
          const newStore = useSettingsStore()
          newStore.loadFromStorage()

          // 验证设置已正确恢复
          return (
            newStore.settings.ocrEngine === ocrEngine &&
            newStore.settings.sourceLanguage === sourceLanguage &&
            newStore.settings.textDetector === textDetector
          )
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 4: 设置持久化往返一致性
   * Validates: Requirements 7.2, 7.3
   *
   * 对于任意有效的翻译服务设置，保存到 localStorage 后再读取应当得到等价的设置
   */
  it('翻译服务设置持久化往返一致性', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('siliconflow', 'deepseek', 'volcano', 'gemini', 'ollama'),
        fc.string({ minLength: 0, maxLength: 50 }),
        fc.string({ minLength: 0, maxLength: 50 }),
        fc.integer({ min: 0, max: 100 }),
        fc.integer({ min: 1, max: 10 }),
        (provider, apiKey, modelName, rpmLimit, maxRetries) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const store = useSettingsStore()

          // 更新翻译服务设置
          store.updateTranslationService({
            provider: provider as 'siliconflow' | 'deepseek' | 'volcano' | 'gemini' | 'ollama',
            apiKey,
            modelName,
            rpmLimit,
            maxRetries
          })

          // 验证设置已保存到 localStorage
          const savedData = localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS]
          if (!savedData) return false

          // 创建新的 store 实例并加载设置
          setActivePinia(createPinia())
          const newStore = useSettingsStore()
          newStore.loadFromStorage()

          // 验证设置已正确恢复
          return (
            newStore.settings.translation.provider === provider &&
            newStore.settings.translation.apiKey === apiKey &&
            newStore.settings.translation.modelName === modelName &&
            newStore.settings.translation.rpmLimit === rpmLimit &&
            newStore.settings.translation.maxRetries === maxRetries
          )
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 5: 主题切换状态一致性
   * Validates: Requirements 10.1, 10.2
   *
   * 对于任意初始主题状态，切换主题后状态应当正确更新
   */
  it('主题切换状态一致性', () => {
    fc.assert(
      fc.property(fc.constantFrom('light', 'dark'), (initialTheme) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const store = useSettingsStore()

        // 设置初始主题
        store.setTheme(initialTheme as 'light' | 'dark')

        // 验证初始主题
        if (store.theme !== initialTheme) return false

        // 切换主题
        store.toggleTheme()

        // 验证主题已切换
        const expectedTheme = initialTheme === 'light' ? 'dark' : 'light'
        if (store.theme !== expectedTheme) return false

        // 验证主题已保存到 localStorage
        const savedTheme = localStorageMock[STORAGE_KEY_THEME]
        if (savedTheme !== expectedTheme) return false

        // 再次切换，应该回到初始主题
        store.toggleTheme()
        return store.theme === initialTheme
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 5: 主题切换状态一致性
   * Validates: Requirements 10.1, 10.2
   *
   * 主题持久化往返一致性
   */
  it('主题持久化往返一致性', () => {
    fc.assert(
      fc.property(fc.constantFrom('light', 'dark'), (theme) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())
        localStorageMock = {}

        const store = useSettingsStore()

        // 设置主题
        store.setTheme(theme as 'light' | 'dark')

        // 验证主题已保存到 localStorage
        const savedTheme = localStorageMock[STORAGE_KEY_THEME]
        if (savedTheme !== theme) return false

        // 创建新的 store 实例并加载主题
        setActivePinia(createPinia())
        const newStore = useSettingsStore()
        newStore.loadThemeFromStorage()

        // 验证主题已正确恢复
        return newStore.theme === theme
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 4: 设置持久化往返一致性
   * Validates: Requirements 7.2, 7.3
   *
   * 重置设置后应该恢复为默认值
   */
  it('重置设置后应该恢复为默认值', () => {
    fc.assert(
      fc.property(
        validFontSizeArb,
        validColorArb,
        (fontSize, textColor) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const store = useSettingsStore()

          // 记录默认值
          const defaultFontSize = store.settings.textStyle.fontSize
          const defaultTextColor = store.settings.textStyle.textColor

          // 修改设置
          store.updateTextStyle({ fontSize, textColor })

          // 验证设置已修改
          if (store.settings.textStyle.fontSize !== fontSize) return false
          if (store.settings.textStyle.textColor !== textColor) return false

          // 重置设置
          store.resetToDefaults()

          // 验证设置已恢复为默认值
          return (
            store.settings.textStyle.fontSize === defaultFontSize &&
            store.settings.textStyle.textColor === defaultTextColor
          )
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 4: 设置持久化往返一致性
   * Validates: Requirements 7.2, 7.3
   *
   * 高质量翻译设置持久化往返一致性
   */
  it('高质量翻译设置持久化往返一致性', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('siliconflow', 'deepseek', 'volcano', 'gemini', 'custom_openai'),
        fc.integer({ min: 1, max: 10 }),
        fc.integer({ min: 1, max: 50 }),
        fc.integer({ min: 1, max: 20 }),
        fc.boolean(),
        fc.boolean(),
        (provider, batchSize, sessionReset, rpmLimit, lowReasoning, forceJsonOutput) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          localStorageMock = {}

          const store = useSettingsStore()

          // 更新高质量翻译设置
          store.updateHqTranslation({
            provider: provider as 'siliconflow' | 'deepseek' | 'volcano' | 'gemini' | 'custom_openai',
            batchSize,
            sessionReset,
            rpmLimit,
            lowReasoning,
            forceJsonOutput
          })

          // 验证设置已保存到 localStorage
          const savedData = localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS]
          if (!savedData) return false

          // 创建新的 store 实例并加载设置
          setActivePinia(createPinia())
          const newStore = useSettingsStore()
          newStore.loadFromStorage()

          // 验证设置已正确恢复
          return (
            newStore.settings.hqTranslation.provider === provider &&
            newStore.settings.hqTranslation.batchSize === batchSize &&
            newStore.settings.hqTranslation.sessionReset === sessionReset &&
            newStore.settings.hqTranslation.rpmLimit === rpmLimit &&
            newStore.settings.hqTranslation.lowReasoning === lowReasoning &&
            newStore.settings.hqTranslation.forceJsonOutput === forceJsonOutput
          )
        }
      ),
      { numRuns: 100 }
    )
  })
})
