import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import type { V2SettingsDocument, V2SettingsTransaction } from '@/api/v2/settings'
import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'

const settingsApiMocks = vi.hoisted(() => ({
  getV2Settings: vi.fn(),
  saveV2SettingsTransaction: vi.fn(),
}))

vi.mock('@/api/v2/settings', () => ({
  getV2Settings: settingsApiMocks.getV2Settings,
  saveV2SettingsTransaction: settingsApiMocks.saveV2SettingsTransaction,
}))

function settingsDocument(
  settings = createDefaultSettings(),
  revision = 4,
): V2SettingsDocument {
  return {
    settings: [
      {
        domain: 'translation',
        payload: settings as unknown as Record<string, unknown>,
        revision,
        schemaVersion: 7,
      },
      {
        domain: 'text_style_defaults',
        payload: settings.textStyle as unknown as Record<string, unknown>,
        revision,
        schemaVersion: 2,
      },
      {
        domain: 'workflow_preferences',
        payload: {
          rememberWorkflowModeEnabled: false,
          lastWorkflowMode: 'translate-current',
        },
        revision,
        schemaVersion: 1,
      },
      {
        domain: 'export_preferences',
        payload: { preserveOriginalFilenames: false },
        revision,
        schemaVersion: 1,
      },
    ],
    bookSettings: [],
    providerSettings: [],
    credentials: [],
  }
}

describe('settings store min text block area percent', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    settingsApiMocks.getV2Settings.mockReset()
    settingsApiMocks.saveV2SettingsTransaction.mockReset()
    settingsApiMocks.getV2Settings.mockResolvedValue(settingsDocument())
    settingsApiMocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [{ domain: 'translation', revision: 5 }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
      prompts: [],
    })
  })

  it('defaults minTextBlockAreaPercent to 0.05', () => {
    expect(useSettingsStore().settings.minTextBlockAreaPercent).toBe(0.05)
  })

  it('hydrates minTextBlockAreaPercent from backend and preserves zero', async () => {
    const settings = createDefaultSettings()
    settings.minTextBlockAreaPercent = 0
    settingsApiMocks.getV2Settings.mockResolvedValue(settingsDocument(settings))

    const store = useSettingsStore()

    expect(await store.loadFromBackend()).toBe(true)
    expect(store.settings.minTextBlockAreaPercent).toBe(0)
  })

  it('saves minTextBlockAreaPercent through the v2 transaction', async () => {
    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    store.setMinTextBlockAreaPercent(2.5)

    expect(await store.saveToBackend()).toBe(true)

    const payload = (
      settingsApiMocks.saveV2SettingsTransaction.mock.calls[0]?.[0]
    ) as V2SettingsTransaction
    expect(payload.settings?.[0]).toMatchObject({
      domain: 'translation',
      baseRevision: 4,
      schemaVersion: 7,
    })
    expect(payload.settings?.[0]?.payload.minTextBlockAreaPercent).toBe(2.5)
  })
})
