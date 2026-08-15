import { afterEach, describe, expect, it, vi } from 'vitest'
import * as fc from 'fast-check'
import { createPinia, setActivePinia } from 'pinia'

import { STORAGE_KEY_THEME } from '@/constants'
import { useSettingsStore } from '@/stores/settings'

type ThemePreference = 'light' | 'dark' | 'system'

function createSettingsStore() {
  setActivePinia(createPinia())
  return useSettingsStore()
}

function installStorageMock() {
  const storage: Record<string, string> = {}
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(key => storage[key] ?? null)
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation((key, value) => {
    storage[key] = value
  })
  return storage
}

describe('settings store properties', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('applies text style edits to the active backend-settings draft', () => {
    fc.assert(fc.property(
      fc.integer({ min: 10, max: 100 }),
      fc.hexaString({ minLength: 6, maxLength: 6 }).map(hex => `#${hex}`),
      fc.constantFrom<'auto' | 'vertical' | 'horizontal'>('auto', 'vertical', 'horizontal'),
      (fontSize, textColor, layoutDirection) => {
        const store = createSettingsStore()

        store.updateTextStyle({ fontSize, textColor, layoutDirection })

        expect(store.settings.textStyle).toMatchObject({
          fontSize,
          textColor,
          layoutDirection,
        })
      },
    ))
  })

  it('keeps provider execution edits in the active draft until explicit save', () => {
    fc.assert(fc.property(
      fc.constantFrom('siliconflow', 'deepseek', 'volcano', 'gemini', 'ollama'),
      fc.integer({ min: 0, max: 100 }),
      fc.integer({ min: 0, max: 10 }),
      (provider, rpmLimit, businessRetries) => {
        const store = createSettingsStore()

        store.setTranslationProvider(provider)
        store.updateTranslationService({ rpmLimit, businessRetries })

        expect(store.settings.translation.provider).toBe(provider)
        expect(store.settings.translation.openaiOptions.execution.rpmLimit).toBe(rpmLimit)
        expect(store.settings.translation.openaiOptions.execution.businessRetries).toBe(businessRetries)
      },
    ))
  })

  it('persists only the UI theme preference in browser storage', () => {
    const storage = installStorageMock()
    fc.assert(fc.property(
      fc.constantFrom<ThemePreference>('light', 'dark', 'system'),
      theme => {
        const store = createSettingsStore()
        store.setTheme(theme)

        expect(store.theme).toBe(theme)
        expect(storage[STORAGE_KEY_THEME]).toBe(theme)
      },
    ))
  })

})
