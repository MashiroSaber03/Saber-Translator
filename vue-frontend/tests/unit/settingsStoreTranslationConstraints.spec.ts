import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { useSettingsStore } from '@/stores/settingsStore'

describe('settingsStore translation constraints', () => {
  const storageState: Record<string, string> = {}

  beforeEach(() => {
    setActivePinia(createPinia())
    for (const key of Object.keys(storageState)) {
      delete storageState[key]
    }

    vi.spyOn(Storage.prototype, 'getItem').mockImplementation((key: string) => storageState[key] ?? null)
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation((key: string, value: string) => {
      storageState[key] = value
    })
    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation((key: string) => {
      delete storageState[key]
    })
  })

  it('persists glossary and non-translate settings through localStorage round-trip', () => {
    const store = useSettingsStore()
    store.updateSettings({
      glossary: {
        enabled: true,
        entries: [
          { source: 'Alice', target: '爱丽丝', note: '主角', matchMode: 'text' },
          { source: '^dragon$', target: '巨龙', note: '', matchMode: 'regex' }
        ]
      },
      nonTranslate: {
        enabled: true,
        entries: [
          { pattern: '<keep>', note: '占位符', matchMode: 'text' },
          { pattern: '\\{[^}]+\\}', note: '宏变量', matchMode: 'regex' }
        ]
      }
    } as any)

    setActivePinia(createPinia())
    const restoredStore = useSettingsStore()
    restoredStore.loadFromStorage()

    expect(restoredStore.settings.glossary).toEqual({
      enabled: true,
      entries: [
        { source: 'Alice', target: '爱丽丝', note: '主角', matchMode: 'text' },
        { source: '^dragon$', target: '巨龙', note: '', matchMode: 'regex' }
      ]
    })
    expect(restoredStore.settings.nonTranslate).toEqual({
      enabled: true,
      entries: [
        { pattern: '<keep>', note: '占位符', matchMode: 'text' },
        { pattern: '\\{[^}]+\\}', note: '宏变量', matchMode: 'regex' }
      ]
    })
  })
})
