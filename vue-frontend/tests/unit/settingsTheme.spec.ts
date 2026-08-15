import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useSettingsStore } from '@/stores/settings'
import { STORAGE_KEY_THEME } from '@/constants'

describe('settings theme startup', () => {
  beforeEach(() => {
    vi.unstubAllGlobals()
    setActivePinia(createPinia())
    localStorage.clear()
    document.documentElement.removeAttribute('data-theme')
    document.body.removeAttribute('data-theme')
  })

  function installColorSchemeMedia(initialMatches: boolean) {
    let matches = initialMatches
    const listeners = new Set<(event: MediaQueryListEvent) => void>()
    const mediaQuery = '(prefers-color-scheme: dark)'

    vi.stubGlobal('matchMedia', vi.fn((query: string) => ({
      matches,
      media: query,
      onchange: null,
      addEventListener: (event: string, listener: (event: MediaQueryListEvent) => void) => {
        if (event === 'change') listeners.add(listener)
      },
      removeEventListener: (event: string, listener: (event: MediaQueryListEvent) => void) => {
        if (event === 'change') listeners.delete(listener)
      },
      addListener: (listener: (event: MediaQueryListEvent) => void) => listeners.add(listener),
      removeListener: (listener: (event: MediaQueryListEvent) => void) => listeners.delete(listener),
      dispatchEvent: () => true,
    })))

    return {
      setMatches(nextMatches: boolean) {
        matches = nextMatches
        listeners.forEach(listener => listener({ matches, media: mediaQuery } as MediaQueryListEvent))
      },
    }
  }

  it('applies the saved theme during settings initialization', () => {
    localStorage.setItem(STORAGE_KEY_THEME, 'dark')

    const settingsStore = useSettingsStore()
    settingsStore.initSettings()

    expect(settingsStore.theme).toBe('dark')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    expect(document.body.getAttribute('data-theme')).toBe('dark')
  })

  it('follows the system color scheme when the saved theme is system', () => {
    const media = installColorSchemeMedia(true)
    localStorage.setItem(STORAGE_KEY_THEME, 'system')

    const settingsStore = useSettingsStore()
    settingsStore.initSettings()

    expect(settingsStore.theme).toBe('system')
    expect(settingsStore.effectiveTheme).toBe('dark')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')

    media.setMatches(false)

    expect(settingsStore.effectiveTheme).toBe('light')
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
    expect(document.body.getAttribute('data-theme')).toBe('light')
  })

  it('applies explicit theme changes even when theme persistence is unavailable', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('theme storage unavailable')
    })

    const settingsStore = useSettingsStore()
    settingsStore.setTheme('dark')

    expect(settingsStore.theme).toBe('dark')
    expect(settingsStore.effectiveTheme).toBe('dark')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    expect(document.body.getAttribute('data-theme')).toBe('dark')
  })

})
