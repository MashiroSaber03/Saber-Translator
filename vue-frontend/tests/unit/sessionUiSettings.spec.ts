import { describe, expect, it, vi } from 'vitest'
import { createDefaultSettings } from '@/stores/settings/defaults'
import type { TranslationSettings, TranslationSettingsUpdates, TextStyleSettings } from '@/types/settings'
import { applySessionUiSettings, type SessionUiSettingsTarget } from '@/stores/sessionUiSettings'

function createTarget(): SessionUiSettingsTarget & {
  settings: TranslationSettings
  updateSettings: ReturnType<typeof vi.fn>
  updateTextStyle: ReturnType<typeof vi.fn>
} {
  const settings = createDefaultSettings()
  return {
    settings,
    updateSettings: vi.fn((updates: TranslationSettingsUpdates) => {
      Object.assign(settings, updates)
    }),
    updateTextStyle: vi.fn((updates: Partial<TextStyleSettings>) => {
      Object.assign(settings.textStyle, updates)
    }),
  }
}

describe('session UI settings contract', () => {
  it('applies current session ui_settings through a named helper', () => {
    const target = createTarget()

    applySessionUiSettings({
      targetLanguage: 'English',
      sourceLanguage: 'Japanese',
      fontSize: 32,
      autoFontSize: true,
      fontFamily: 'Noto Sans',
      layoutDirection: 'vertical',
      textColor: '#111111',
      fillColor: '#ffffff',
      useInpaintingMethod: 'litelama',
      strokeEnabled: true,
      strokeColor: '#222222',
      strokeWidth: 2,
      lineSpacing: 1.4,
      textAlign: 'center',
      useAutoTextColor: true,
    }, target)

    expect(target.updateSettings).toHaveBeenCalledWith({
      targetLanguage: 'English',
      sourceLanguage: 'Japanese',
    })
    expect(target.updateTextStyle).toHaveBeenCalledWith(expect.objectContaining({
      fontSize: 32,
      autoFontSize: true,
      fontFamily: 'Noto Sans',
      layoutDirection: 'vertical',
      inpaintMethod: 'litelama',
      strokeEnabled: true,
      lineSpacing: 1.4,
      textAlign: 'center',
      useAutoTextColor: true,
    }))
  })

  it('does not clear absent language fields while applying a partial session payload', () => {
    const target = createTarget()
    target.settings.sourceLanguage = 'Korean'

    applySessionUiSettings({
      targetLanguage: 'English',
    }, target)

    expect(target.updateSettings).toHaveBeenCalledWith({ targetLanguage: 'English' })
    expect(target.settings.sourceLanguage).toBe('Korean')
  })
})
