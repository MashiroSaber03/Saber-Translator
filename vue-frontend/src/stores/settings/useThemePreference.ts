import { ref } from 'vue'

export type ThemePreference = 'light' | 'dark' | 'system'
export type EffectiveTheme = 'light' | 'dark'

export function useThemePreference(storageKey: string) {
  const theme = ref<ThemePreference>('light')
  const effectiveTheme = ref<EffectiveTheme>('light')
  let systemThemeMedia: MediaQueryList | null = null

  function resolveSystemTheme(): EffectiveTheme {
    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    }
    return 'light'
  }

  function applyEffectiveTheme(nextTheme: EffectiveTheme): void {
    effectiveTheme.value = nextTheme
    if (typeof document !== 'undefined') {
      document.documentElement.setAttribute('data-theme', nextTheme)
      document.body.setAttribute('data-theme', nextTheme)
    }
  }

  function handleSystemThemeChange(event: MediaQueryListEvent): void {
    if (theme.value === 'system') {
      applyEffectiveTheme(event.matches ? 'dark' : 'light')
    }
  }

  function detachSystemThemeListener(): void {
    if (!systemThemeMedia) return
    systemThemeMedia.removeEventListener('change', handleSystemThemeChange)
    systemThemeMedia = null
  }

  function attachSystemThemeListener(): void {
    detachSystemThemeListener()
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
    systemThemeMedia = window.matchMedia('(prefers-color-scheme: dark)')
    systemThemeMedia.addEventListener('change', handleSystemThemeChange)
  }

  function applyThemePreference(): void {
    if (theme.value === 'system') {
      attachSystemThemeListener()
      applyEffectiveTheme(resolveSystemTheme())
      return
    }
    detachSystemThemeListener()
    applyEffectiveTheme(theme.value)
  }

  function setTheme(nextTheme: ThemePreference): void {
    theme.value = nextTheme
    try {
      localStorage.setItem(storageKey, nextTheme)
    } catch {
      // Theme persistence is best-effort; the active theme still updates.
    }
    applyThemePreference()
  }

  function toggleTheme(): void {
    setTheme(theme.value === 'light' ? 'dark' : theme.value === 'dark' ? 'system' : 'light')
  }

  function loadThemeFromStorage(): void {
    try {
      const storedTheme = localStorage.getItem(storageKey)
      if (storedTheme === 'light' || storedTheme === 'dark' || storedTheme === 'system') {
        setTheme(storedTheme)
      }
    } catch {
      return
    }
  }

  return {
    theme,
    effectiveTheme,
    setTheme,
    toggleTheme,
    loadThemeFromStorage,
    detachSystemThemeListener,
  }
}
