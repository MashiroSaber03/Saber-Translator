import { beforeEach, describe, expect, it, vi } from 'vitest'

const { mockFetchTextStyleDefaults } = vi.hoisted(() => ({
  mockFetchTextStyleDefaults: vi.fn()
}))

vi.mock('@/api/config', () => ({
  getTextStyleDefaults: mockFetchTextStyleDefaults
}))

import {
  TEXT_STYLE_DEFAULTS,
  getTextStyleDefaults,
  reloadTextStyleDefaultsFromBackend,
  resetTextStyleDefaultsToBundled
} from '@/defaults/textStyleDefaults'
import { createDefaultSettings } from '@/stores/settings/defaults'

const bundledDefaults = getTextStyleDefaults()

describe('textStyleDefaults runtime reload', () => {
  beforeEach(() => {
    resetTextStyleDefaultsToBundled()
    mockFetchTextStyleDefaults.mockReset()
  })

  it('重新加载后应更新运行时默认值并影响后续 settings 初始化', async () => {
    mockFetchTextStyleDefaults.mockResolvedValue({
      success: true,
      defaults: {
        ...bundledDefaults,
        fontSize: 30,
        textColor: '#112233'
      }
    })

    const reloaded = await reloadTextStyleDefaultsFromBackend()

    expect(reloaded).toBe(true)
    expect(TEXT_STYLE_DEFAULTS.fontSize).toBe(30)
    expect(TEXT_STYLE_DEFAULTS.textColor).toBe('#112233')
    expect(getTextStyleDefaults().fontSize).toBe(30)
    expect(createDefaultSettings().textStyle.fontSize).toBe(30)
    expect(createDefaultSettings().textStyle.textColor).toBe('#112233')
  })

  it('重新加载失败时应保留当前默认值', async () => {
    mockFetchTextStyleDefaults.mockRejectedValue(new Error('network error'))

    const reloaded = await reloadTextStyleDefaultsFromBackend()

    expect(reloaded).toBe(false)
    expect(TEXT_STYLE_DEFAULTS.fontSize).toBe(bundledDefaults.fontSize)
    expect(TEXT_STYLE_DEFAULTS.textColor).toBe(bundledDefaults.textColor)
  })
})
