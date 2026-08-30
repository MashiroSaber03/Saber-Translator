import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import type { V2SettingsDocument, V2SettingsTransaction } from '@/api/v2/settings'
import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'
import { parseCurrentSettings } from '@/stores/settings/schema'
import { deepClone } from '@/utils/deepClone'

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
    settings: [
      {
        domain: 'translation',
        payload: settings as unknown as Record<string, unknown>,
        revision,
        schemaVersion: 9,
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
      prompts: [],
    })
  })

  it('uses factory runtime options for a provider with no saved memory', () => {
    const store = useSettingsStore()
    const defaults = createDefaultSettings()

    store.updateTranslationService({
      translationMode: 'single',
      forceJsonOutput: true,
      rpmLimit: 91,
    })
    store.setTranslationProvider('deepseek')
    expect(store.settings.translation.translationMode).toBe(
      defaults.translation.translationMode,
    )
    expect(store.settings.translation.openaiOptions).toEqual(
      defaults.translation.openaiOptions,
    )

    store.updateHqTranslation({
      batchSize: 9,
      prompt: 'provider-specific HQ prompt',
      rpmLimit: 88,
    })
    store.setHqProvider('deepseek')
    expect(store.settings.hqTranslation).toMatchObject({
      batchSize: defaults.hqTranslation.batchSize,
      prompt: defaults.hqTranslation.prompt,
      openaiOptions: defaults.hqTranslation.openaiOptions,
    })

    store.updateAiVisionOcr({
      minImageSize: 777,
      prompt: 'provider-specific OCR prompt',
      promptMode: 'json',
      rpmLimit: 66,
    })
    store.setAiVisionOcrProvider('siliconflow')
    expect(store.settings.aiVisionOcr).toMatchObject({
      minImageSize: defaults.aiVisionOcr.minImageSize,
      prompt: defaults.aiVisionOcr.prompt,
      promptMode: defaults.aiVisionOcr.promptMode,
      openaiOptions: defaults.aiVisionOcr.openaiOptions,
    })
  })

  it('rejects extra fields instead of treating them as an older settings shape', () => {
    const topLevel = createDefaultSettings() as unknown as Record<string, unknown>
    topLevel.removedField = true

    const nested = deepClone(createDefaultSettings())
    ;(nested.translation.openaiOptions.request as Record<string, unknown>).removedField = true

    const proofreading = deepClone(createDefaultSettings())
    proofreading.proofreading.rounds = [{
      ...proofreading.hqTranslation,
      id: '11111111-1111-4111-8111-111111111111',
      name: '第1轮',
    }]
    ;(proofreading.proofreading.rounds[0] as unknown as Record<string, unknown>).sessionReset = true

    const retiredProofreadingRetry = deepClone(createDefaultSettings())
    ;(retiredProofreadingRetry.proofreading as unknown as Record<string, unknown>).maxRetries = 2

    expect(parseCurrentSettings(topLevel)).toBeNull()
    expect(parseCurrentSettings(nested)).toBeNull()
    expect(parseCurrentSettings(proofreading)).toBeNull()
    expect(parseCurrentSettings(retiredProofreadingRetry)).toBeNull()
  })

  it('rejects malformed numeric runtime options instead of coercing them', () => {
    const fractionalRetry = deepClone(createDefaultSettings())
    fractionalRetry.translation.openaiOptions.execution.transportRetries = 1.5

    const invalidTemperature = deepClone(createDefaultSettings())
    invalidTemperature.translation.openaiOptions.request.temperature = 2.1

    const fractionalBatch = deepClone(createDefaultSettings())
    fractionalBatch.hqTranslation.batchSize = 1.5

    expect(parseCurrentSettings(fractionalRetry)).toBeNull()
    expect(parseCurrentSettings(invalidTemperature)).toBeNull()
    expect(parseCurrentSettings(fractionalBatch)).toBeNull()
  })

  it('accepts positive HQ and proofreading batch sizes without fixed upper bounds', () => {
    const settings = createDefaultSettings()
    settings.hqTranslation.batchSize = 128
    settings.proofreading.rounds = [{
      ...settings.hqTranslation,
      id: '11111111-1111-4111-8111-111111111111',
      name: '第1轮',
      batchSize: 256,
    }]

    expect(parseCurrentSettings(settings)?.hqTranslation.batchSize).toBe(128)
    expect(parseCurrentSettings(settings)?.proofreading.rounds[0]?.batchSize).toBe(256)
  })

  it('rejects duplicate proofreading round identities', () => {
    const settings = createDefaultSettings()
    const id = '11111111-1111-4111-8111-111111111111'
    settings.proofreading.rounds = [
      { ...settings.hqTranslation, id, name: '第1轮' },
      { ...settings.hqTranslation, id, name: '第2轮' },
    ]

    expect(parseCurrentSettings(settings)).toBeNull()
  })

  it('accepts backend-valid text styles without frontend-only upper bounds', () => {
    const settings = createDefaultSettings()
    settings.textStyle = {
      ...settings.textStyle,
      fontSize: 1024,
      strokeWidth: 80,
      lineSpacing: 12.5,
    }

    expect(parseCurrentSettings(settings)?.textStyle).toMatchObject({
      fontSize: 1024,
      strokeWidth: 80,
      lineSpacing: 12.5,
    })
  })

  it('finishes a successful transaction without a post-commit settings reload', async () => {
    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    store.settings.translation.apiKey = 'new-secret'
    store.settings.translation.modelName = 'new-model'
    settingsApiMocks.saveV2SettingsTransaction.mockResolvedValueOnce({
      settings: [
        { domain: 'translation', revision: 9 },
        { domain: 'text_style_defaults', revision: 9 },
      ],
      bookSettings: [],
      providerSettings: [{
        domain: 'translation',
        provider: 'siliconflow',
        revision: 1,
      }],
      credentials: [{
        credentialId: '11111111-1111-4111-8111-111111111111',
        credentialVersionId: '22222222-2222-4222-8222-222222222222',
        currentVersion: 1,
        domain: 'translation',
        hasKey: true,
        provider: 'siliconflow',
        revision: 1,
        secret: { api_key: 'new-secret' },
      }],
      prompts: [],
    })

    expect(await store.saveToBackend()).toBe(true)

    expect(settingsApiMocks.getV2Settings).toHaveBeenCalledTimes(1)
    expect(store.settings.translation.apiKey).toBe('new-secret')

    expect(await store.saveToBackend()).toBe(true)
    const secondTransaction = (
      settingsApiMocks.saveV2SettingsTransaction.mock.calls[1]?.[0]
    ) as V2SettingsTransaction
    expect(secondTransaction.settings?.[0]?.baseRevision).toBe(9)
    expect(secondTransaction.providerSettings?.find(
      row => row.domain === 'translation' && row.provider === 'siliconflow',
    )).toMatchObject({
      baseRevision: 1,
      credentialVersionId: '22222222-2222-4222-8222-222222222222',
    })
    expect(secondTransaction.credentialEdits).toEqual([])
    expect(settingsApiMocks.getV2Settings).toHaveBeenCalledTimes(1)
  })

  it('keeps proofreading provider identities stable after removing a middle round', async () => {
    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    const ids = [
      '11111111-1111-4111-8111-111111111111',
      '22222222-2222-4222-8222-222222222222',
      '33333333-3333-4333-8333-333333333333',
    ]
    ids.forEach((id, index) => {
      store.addProofreadingRound({
        ...deepClone(store.settings.hqTranslation),
        id,
        name: `第${index + 1}轮`,
        apiKey: `proof-key-${index + 1}`,
      })
    })

    store.removeProofreadingRound(1)
    expect(await store.saveToBackend()).toBe(true)

    const transaction = (
      settingsApiMocks.saveV2SettingsTransaction.mock.calls[0]?.[0]
    ) as V2SettingsTransaction
    const proofreadingDomains = transaction.providerSettings
      ?.filter(row => row.domain.startsWith('proofreading_'))
      .map(row => row.domain)
    expect(proofreadingDomains).toEqual([
      `proofreading_${ids[0]}`,
      `proofreading_${ids[2]}`,
    ])
    expect(transaction.credentialEdits
      ?.filter(row => row.domain.startsWith('proofreading_'))
      .map(row => row.domain)).toEqual(proofreadingDomains)
  })

  it('does not send removed session reset fields in v2 settings transactions', async () => {
    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    store.settings.hqTranslation.openaiOptions.execution.rpmLimit = 9
    store.settings.proofreading.rounds = [{
      id: '11111111-1111-4111-8111-111111111111',
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
    const proofreadingProvider = transaction.providerSettings?.find(
      row => row.domain === 'proofreading_11111111-1111-4111-8111-111111111111',
    )
    expect(proofreadingProvider).toMatchObject({
      domain: 'proofreading_11111111-1111-4111-8111-111111111111',
      provider: 'siliconflow',
      payload: {
        modelName: 'proof-model',
        customBaseUrl: '',
        prompt: 'proof',
        batchSize: 2,
        openaiOptions: { execution: { rpmLimit: 7 } },
      },
    })
    expect(proofreadingProvider?.payload).not.toHaveProperty('apiKey')
    expect(proofreadingProvider?.payload).not.toHaveProperty('name')
    expect(proofreadingProvider?.payload).not.toHaveProperty('provider')
  })

  it('keeps proofreading UI patches inside nested OpenAI options', () => {
    const store = useSettingsStore()
    store.setProofreadingEnabled(true)
    store.addProofreadingRound({
      id: '11111111-1111-4111-8111-111111111111',
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
      settingsSchemaVersion: 9,
      translation: { provider: 'custom' },
    } as unknown as ReturnType<typeof createDefaultSettings>))
    const store = useSettingsStore()

    expect(await store.loadFromBackend()).toBe(false)
    expect(store.isBackendReady).toBe(false)
    expect(store.backendError).toContain('格式无效')
    expect(store.settings).toEqual(createDefaultSettings())
  })

  it('rejects missing or malformed current workflow preferences', async () => {
    const missing = settingsDocument()
    missing.settings = missing.settings.filter(
      entry => entry.domain !== 'workflow_preferences',
    )
    settingsApiMocks.getV2Settings.mockResolvedValueOnce(missing)

    const missingStore = useSettingsStore()
    expect(await missingStore.loadFromBackend()).toBe(false)
    expect(missingStore.backendError).toContain('工作流偏好设置缺失')

    setActivePinia(createPinia())
    const malformed = settingsDocument()
    const workflow = malformed.settings.find(
      entry => entry.domain === 'workflow_preferences',
    )
    if (!workflow) throw new Error('test fixture is missing workflow preferences')
    workflow.payload = {
      rememberWorkflowModeEnabled: 'false',
      lastWorkflowMode: 'retired-mode',
    }
    settingsApiMocks.getV2Settings.mockResolvedValueOnce(malformed)

    const malformedStore = useSettingsStore()
    expect(await malformedStore.loadFromBackend()).toBe(false)
    expect(malformedStore.backendError).toContain('工作流偏好设置格式无效')
  })
})
