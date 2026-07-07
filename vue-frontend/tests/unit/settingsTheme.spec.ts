import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
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

  it('defines a semantic dark token layer for product surfaces', () => {
    const semanticTokens = readFileSync(
      resolve(process.cwd(), 'src/styles/tokens/semantic.css'),
      'utf8',
    )

    const darkLayerMatch = semanticTokens.match(/:root\[data-theme="dark"\]\s*\{([\s\S]*?)\n\}/)
    expect(darkLayerMatch).not.toBeNull()
    const darkLayer = darkLayerMatch?.[1] ?? ''

    expect(darkLayer).toContain('color-scheme: dark')
    expect(darkLayer).toMatch(/--color-surface-page:\s*#[0-9a-fA-F]{6}/)
    expect(darkLayer).toMatch(/--color-text-default:\s*#[0-9a-fA-F]{6}/)
    expect(darkLayer).toMatch(/--color-border-default:\s*#[0-9a-fA-F]{6}/)
  })

  it('maps Insight domain tokens through semantic theme roles', () => {
    const domainTokens = readFileSync(
      resolve(process.cwd(), 'src/styles/tokens/domain.css'),
      'utf8',
    )

    expect(domainTokens).not.toContain('Shared Insight')
    expect(domainTokens).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(domainTokens).toContain('--insight-surface-page: var(--color-surface-panel)')
    expect(domainTokens).toContain('--insight-status-warning: var(--color-status-warning)')
  })
})
