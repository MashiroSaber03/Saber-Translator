import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const { mockFetchTextStyleDefaults } = vi.hoisted(() => ({
  mockFetchTextStyleDefaults: vi.fn()
}))

vi.mock('@/api/config', () => ({
  getTextStyleDefaults: mockFetchTextStyleDefaults
}))

import {
  TEXT_STYLE_DEFAULTS,
  getTextStyleDefaults,
  normalizeImageTextStyleFields,
  normalizeTextStyleSettings,
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

  afterEach(() => {
    vi.restoreAllMocks()
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
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    mockFetchTextStyleDefaults.mockRejectedValue(new Error('network error'))

    const reloaded = await reloadTextStyleDefaultsFromBackend()

    expect(reloaded).toBe(false)
    expect(TEXT_STYLE_DEFAULTS.fontSize).toBe(bundledDefaults.fontSize)
    expect(TEXT_STYLE_DEFAULTS.textColor).toBe(bundledDefaults.textColor)
  })

  it('keeps text-style field normalization on a shared field builder', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/defaults/textStyleDefaults.ts'), 'utf8')

    expect(source).toContain('function buildTextStyleFields')
    expect(source).not.toContain('fontSize: style.fontSize !== undefined')
    expect(source).not.toContain('fontSize: image.fontSize !== undefined')
  })

  it('normalizes settings and image text-style fields through the same parser rules', () => {
    const partialStyle = {
      fontSize: 24,
      autoFontSize: true,
      fontFamily: 'fonts/custom.ttf',
      layoutDirection: 'horizontal' as const,
      textColor: '#112233',
      fillColor: '#445566',
      inpaintMethod: 'litelama' as const,
      strokeEnabled: true,
      strokeColor: '#778899',
      strokeWidth: 3,
      lineSpacing: 1.25,
      textAlign: 'center' as const,
      useAutoTextColor: true,
    }

    expect(normalizeTextStyleSettings(partialStyle)).toMatchObject(partialStyle)
    expect(normalizeImageTextStyleFields(partialStyle)).toMatchObject(partialStyle)
    expect(() => normalizeImageTextStyleFields({ fontSize: -1 })).toThrow('fontSize must be a positive integer')
  })
})
