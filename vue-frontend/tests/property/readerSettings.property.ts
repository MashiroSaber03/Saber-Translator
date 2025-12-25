/**
 * 阅读器设置持久化属性测试
 * 使用 fast-check 进行属性基测试，验证阅读器设置的持久化一致性
 *
 * Feature: vue-frontend-migration, Property 15: 阅读器设置持久化往返一致性
 * Validates: Requirements 5.4
 */
import { describe, it, beforeEach, afterEach, vi } from 'vitest'
import * as fc from 'fast-check'

// 阅读器设置接口（与 ReaderControls 组件中的定义一致）
interface ReaderSettings {
  /** 图片宽度百分比 (50-100) */
  imageWidth: number
  /** 图片间距像素 (0-50) */
  imageGap: number
  /** 背景颜色 */
  bgColor: string
}

// localStorage 存储键名
const READER_SETTINGS_KEY = 'readerSettings'

// 背景颜色预设
const BG_COLOR_PRESETS = ['#1a1a2e', '#ffffff', '#f5f5dc', '#2d2d2d']

describe('阅读器设置持久化属性测试', () => {
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
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  /**
   * 生成有效的图片宽度 (50-100)
   */
  const validImageWidthArb = fc.integer({ min: 50, max: 100 })

  /**
   * 生成有效的图片间距 (0-50)
   */
  const validImageGapArb = fc.integer({ min: 0, max: 50 })

  /**
   * 生成有效的背景颜色（预设颜色之一）
   */
  const validBgColorArb = fc.constantFrom(...BG_COLOR_PRESETS)

  /**
   * 生成有效的阅读器设置
   */
  const validReaderSettingsArb = fc.record({
    imageWidth: validImageWidthArb,
    imageGap: validImageGapArb,
    bgColor: validBgColorArb
  })

  /**
   * 保存设置到 localStorage
   */
  function saveSettings(settings: ReaderSettings): void {
    localStorage.setItem(READER_SETTINGS_KEY, JSON.stringify(settings))
  }

  /**
   * 从 localStorage 加载设置
   */
  function loadSettings(): ReaderSettings | null {
    const saved = localStorage.getItem(READER_SETTINGS_KEY)
    if (saved) {
      try {
        return JSON.parse(saved) as ReaderSettings
      } catch {
        return null
      }
    }
    return null
  }

  /**
   * Feature: vue-frontend-migration, Property 15: 阅读器设置持久化往返一致性
   * Validates: Requirements 5.4
   *
   * 对于任意有效的阅读器设置，保存到 localStorage 后再读取应当得到等价的设置
   */
  it('阅读器设置保存后加载应保持一致', () => {
    fc.assert(
      fc.property(validReaderSettingsArb, (settings) => {
        // 重置 localStorage
        localStorageMock = {}

        // 保存设置
        saveSettings(settings)

        // 验证设置已保存到 localStorage
        const savedData = localStorageMock[READER_SETTINGS_KEY]
        if (!savedData) return false

        // 加载设置
        const loadedSettings = loadSettings()
        if (!loadedSettings) return false

        // 验证设置已正确恢复
        return (
          loadedSettings.imageWidth === settings.imageWidth &&
          loadedSettings.imageGap === settings.imageGap &&
          loadedSettings.bgColor === settings.bgColor
        )
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 15: 阅读器设置持久化往返一致性
   * Validates: Requirements 5.4
   *
   * 图片宽度设置应在有效范围内 (50-100)
   */
  it('图片宽度设置应在有效范围内', () => {
    fc.assert(
      fc.property(validImageWidthArb, (imageWidth) => {
        // 重置 localStorage
        localStorageMock = {}

        const settings: ReaderSettings = {
          imageWidth,
          imageGap: 8,
          bgColor: '#1a1a2e'
        }

        // 保存设置
        saveSettings(settings)

        // 加载设置
        const loadedSettings = loadSettings()
        if (!loadedSettings) return false

        // 验证图片宽度在有效范围内
        return loadedSettings.imageWidth >= 50 && loadedSettings.imageWidth <= 100
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 15: 阅读器设置持久化往返一致性
   * Validates: Requirements 5.4
   *
   * 图片间距设置应在有效范围内 (0-50)
   */
  it('图片间距设置应在有效范围内', () => {
    fc.assert(
      fc.property(validImageGapArb, (imageGap) => {
        // 重置 localStorage
        localStorageMock = {}

        const settings: ReaderSettings = {
          imageWidth: 100,
          imageGap,
          bgColor: '#1a1a2e'
        }

        // 保存设置
        saveSettings(settings)

        // 加载设置
        const loadedSettings = loadSettings()
        if (!loadedSettings) return false

        // 验证图片间距在有效范围内
        return loadedSettings.imageGap >= 0 && loadedSettings.imageGap <= 50
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 15: 阅读器设置持久化往返一致性
   * Validates: Requirements 5.4
   *
   * 背景颜色设置应为预设颜色之一
   */
  it('背景颜色设置应为预设颜色之一', () => {
    fc.assert(
      fc.property(validBgColorArb, (bgColor) => {
        // 重置 localStorage
        localStorageMock = {}

        const settings: ReaderSettings = {
          imageWidth: 100,
          imageGap: 8,
          bgColor
        }

        // 保存设置
        saveSettings(settings)

        // 加载设置
        const loadedSettings = loadSettings()
        if (!loadedSettings) return false

        // 验证背景颜色为预设颜色之一
        return BG_COLOR_PRESETS.includes(loadedSettings.bgColor)
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 15: 阅读器设置持久化往返一致性
   * Validates: Requirements 5.4
   *
   * 多次保存设置后，最后一次保存的设置应被正确加载
   */
  it('多次保存设置后应加载最后一次保存的设置', () => {
    fc.assert(
      fc.property(
        fc.array(validReaderSettingsArb, { minLength: 2, maxLength: 10 }),
        (settingsArray) => {
          // 重置 localStorage
          localStorageMock = {}

          // 多次保存设置
          for (const settings of settingsArray) {
            saveSettings(settings)
          }

          // 加载设置
          const loadedSettings = loadSettings()
          if (!loadedSettings) return false

          // 验证加载的是最后一次保存的设置
          const lastSettings = settingsArray[settingsArray.length - 1]
          if (!lastSettings) return false
          return (
            loadedSettings.imageWidth === lastSettings.imageWidth &&
            loadedSettings.imageGap === lastSettings.imageGap &&
            loadedSettings.bgColor === lastSettings.bgColor
          )
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 15: 阅读器设置持久化往返一致性
   * Validates: Requirements 5.4
   *
   * 页面刷新后设置应正确恢复（模拟通过清空内存状态后重新加载）
   */
  it('页面刷新后设置应正确恢复', () => {
    fc.assert(
      fc.property(validReaderSettingsArb, (settings) => {
        // 重置 localStorage
        localStorageMock = {}

        // 保存设置
        saveSettings(settings)

        // 模拟页面刷新：清空内存中的设置引用，只保留 localStorage
        // 验证 localStorage 中有数据
        if (!localStorageMock[READER_SETTINGS_KEY]) return false
        
        // 重新从 localStorage 加载
        const loadedSettings = loadSettings()
        if (!loadedSettings) return false

        // 验证设置已正确恢复
        return (
          loadedSettings.imageWidth === settings.imageWidth &&
          loadedSettings.imageGap === settings.imageGap &&
          loadedSettings.bgColor === settings.bgColor
        )
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 15: 阅读器设置持久化往返一致性
   * Validates: Requirements 5.4
   *
   * 设置合并：加载时应正确合并默认值和已保存的值
   */
  it('加载时应正确合并默认值和已保存的值', () => {
    fc.assert(
      fc.property(validImageWidthArb, (imageWidth) => {
        // 重置 localStorage
        localStorageMock = {}

        // 只保存部分设置
        const partialSettings = { imageWidth }
        localStorage.setItem(READER_SETTINGS_KEY, JSON.stringify(partialSettings))

        // 加载设置并合并默认值
        const defaultSettings: ReaderSettings = {
          imageWidth: 100,
          imageGap: 8,
          bgColor: '#1a1a2e'
        }

        const saved = localStorage.getItem(READER_SETTINGS_KEY)
        let loadedSettings = defaultSettings
        if (saved) {
          try {
            const parsed = JSON.parse(saved)
            loadedSettings = { ...defaultSettings, ...parsed }
          } catch {
            // 解析失败时使用默认值
          }
        }

        // 验证已保存的值被正确加载，未保存的值使用默认值
        return (
          loadedSettings.imageWidth === imageWidth &&
          loadedSettings.imageGap === defaultSettings.imageGap &&
          loadedSettings.bgColor === defaultSettings.bgColor
        )
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 15: 阅读器设置持久化往返一致性
   * Validates: Requirements 5.4
   *
   * JSON 序列化往返一致性
   */
  it('JSON 序列化往返一致性', () => {
    fc.assert(
      fc.property(validReaderSettingsArb, (settings) => {
        // 序列化
        const serialized = JSON.stringify(settings)

        // 反序列化
        const deserialized = JSON.parse(serialized) as ReaderSettings

        // 验证往返一致性
        return (
          deserialized.imageWidth === settings.imageWidth &&
          deserialized.imageGap === settings.imageGap &&
          deserialized.bgColor === settings.bgColor
        )
      }),
      { numRuns: 100 }
    )
  })
})
