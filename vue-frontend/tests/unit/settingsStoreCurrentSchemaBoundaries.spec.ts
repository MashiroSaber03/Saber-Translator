import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
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
  revision = 8,
): V2SettingsDocument {
  return {
    settings: [{
      domain: 'translation',
      payload: settings as unknown as Record<string, unknown>,
      revision,
      schemaVersion: 3,
    }],
    bookSettings: [],
    providerSettings: [],
    credentials: [],
  }
}

describe('settings store current schema boundaries', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    settingsApiMocks.getV2Settings.mockReset()
    settingsApiMocks.saveV2SettingsTransaction.mockReset()
    settingsApiMocks.getV2Settings.mockResolvedValue(settingsDocument())
    settingsApiMocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [{ domain: 'translation', revision: 9 }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })
  })

  it('keeps settings store cloning on shared helpers', () => {
    for (const file of [
      'src/stores/settings/defaults.ts',
      'src/stores/settings/index.ts',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, file).toContain("import { deepClone } from '@/utils/deepClone'")
      expect(source, file).not.toContain('function cloneJson')
      expect(source, file).not.toContain('JSON.parse(JSON.stringify')
    }

    for (const file of [
      'src/stores/settings/modules/translation.ts',
      'src/stores/settings/modules/hqTranslation.ts',
      'src/stores/settings/modules/ocr.ts',
      'src/stores/settings/modules/pluginAgent.ts',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, file).toContain('cloneOpenAiOptions')
      expect(source, file).not.toContain('JSON.parse(JSON.stringify')
    }
  })

  it('keeps provider cache switching on shared settings helpers', () => {
    for (const file of [
      'src/stores/settings/modules/translation.ts',
      'src/stores/settings/modules/hqTranslation.ts',
      'src/stores/settings/modules/ocr.ts',
      'src/stores/settings/modules/pluginAgent.ts',
      'src/stores/settings/modules/webImport.ts',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, file).toContain("from '../providerConfigCache'")
      expect(source, file).toContain('saveProviderCacheEntry')
      expect(source, file).toContain('restoreProviderCacheEntry')
      expect(source, file).not.toMatch(/providerConfigs\.value\.\w+\[[^\]]+\]\s*=/)
    }
  })

  it('keeps removed-field probes typed without broad any escapes', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'tests/unit/settingsStoreCurrentSchemaBoundaries.spec.ts'),
      'utf8',
    )

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('does not send removed session reset fields in v2 settings transactions', async () => {
    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    store.settings.hqTranslation.openaiOptions.execution.rpmLimit = 9
    store.settings.proofreading.rounds = [{
      name: '第1轮',
      provider: 'siliconflow',
      apiKey: 'proof-key',
      modelName: 'proof-model',
      customBaseUrl: '',
      prompt: 'proof',
      batchSize: 2,
      openaiOptions: {
        request: { forceJsonOutput: false },
        execution: {
          useStream: true,
          rpmLimit: 7,
          transportRetries: 1,
          businessRetries: 1,
        },
      },
    }]

    expect(await store.saveToBackend()).toBe(true)

    const transaction = (
      settingsApiMocks.saveV2SettingsTransaction.mock.calls[0]?.[0]
    ) as V2SettingsTransaction
    const payload = transaction.settings?.[0]?.payload
    expect(payload).not.toHaveProperty('hqSessionReset')
    expect(payload?.hqTranslation).toMatchObject({
      openaiOptions: { execution: { rpmLimit: 9 } },
    })
    const proofreading = payload?.proofreading as {
      rounds: Array<Record<string, unknown>>
    }
    expect(proofreading.rounds[0]).not.toHaveProperty('sessionReset')
    expect(proofreading.rounds[0]).toMatchObject({
      openaiOptions: { execution: { rpmLimit: 7 } },
    })
  })

  it('keeps proofreading UI patches inside nested OpenAI options', () => {
    const store = useSettingsStore()
    store.setProofreadingEnabled(true)
    store.addProofreadingRound({
      name: '第1轮',
      provider: 'siliconflow',
      apiKey: 'proof-key',
      modelName: 'proof-model',
      customBaseUrl: '',
      openaiOptions: {
        request: { forceJsonOutput: false },
        execution: {
          useStream: false,
          rpmLimit: 0,
          transportRetries: 1,
          businessRetries: 0,
        },
      },
      batchSize: 2,
      prompt: 'proof',
    })

    store.updateProofreadingRound(0, {
      rpmLimit: 9,
      transportRetries: 4,
      businessRetries: 2,
      forceJsonOutput: true,
      useStream: true,
      extraBody: { top_p: 0.9 },
    })

    const round = store.settings.proofreading.rounds[0] as Record<string, unknown>
    expect(round).not.toHaveProperty('rpmLimit')
    expect(round).not.toHaveProperty('useStream')
    expect(store.settings.proofreading.rounds[0]?.openaiOptions).toEqual({
      request: {
        forceJsonOutput: true,
        extraBody: { top_p: 0.9 },
      },
      execution: {
        useStream: true,
        rpmLimit: 9,
        transportRetries: 4,
        businessRetries: 2,
      },
    })
  })

  it('rejects malformed authoritative backend settings', async () => {
    settingsApiMocks.getV2Settings.mockResolvedValue(settingsDocument({
      settingsSchemaVersion: 3,
      translation: { provider: 'custom' },
    } as unknown as ReturnType<typeof createDefaultSettings>))
    const store = useSettingsStore()

    expect(await store.loadFromBackend()).toBe(false)
    expect(store.isBackendReady).toBe(false)
    expect(store.backendError).toContain('格式无效')
    expect(store.settings).toEqual(createDefaultSettings())
  })
})
