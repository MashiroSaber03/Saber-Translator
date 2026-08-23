import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  getV2Settings: vi.fn(),
  saveV2SettingsTransaction: vi.fn(),
  prepareBrowserCredentialTransaction: vi.fn(),
  restoreBrowserCredentialLeases: vi.fn(),
}))

vi.mock('@/api/v2/settings', () => ({
  getV2Settings: mocks.getV2Settings,
  saveV2SettingsTransaction: mocks.saveV2SettingsTransaction,
}))

vi.mock('@/services/browserCredentials', () => ({
  prepareBrowserCredentialTransaction: mocks.prepareBrowserCredentialTransaction,
  restoreBrowserCredentialLeases: mocks.restoreBrowserCredentialLeases,
}))

import { useCustomAiProfileStore } from '@/stores/customAiProfileStore'

const PROFILE_ID = '11111111-1111-4111-8111-111111111111'

function credential(apiKey: string, revision = 1) {
  return {
    credentialId: 'credential-1',
    credentialVersionId: `credential-version-${revision}`,
    currentVersion: revision,
    domain: 'custom_ai_profile',
    hasKey: true,
    provider: PROFILE_ID,
    revision,
    secret: { api_key: apiKey },
  }
}

describe('useCustomAiProfileStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    Object.values(mocks).forEach(mock => mock.mockReset())
    mocks.restoreBrowserCredentialLeases.mockResolvedValue([])
    mocks.prepareBrowserCredentialTransaction.mockImplementation(async transaction => ({
      transaction,
      summaries: [],
    }))
    mocks.getV2Settings.mockResolvedValue({
      settings: [{
        domain: 'custom_ai_profiles',
        revision: 2,
        schemaVersion: 1,
        payload: {
          profiles: [{
            id: PROFILE_ID,
            name: '主翻译服务',
            kind: 'chat',
            baseUrl: 'https://gateway.example.com/v1',
            model: 'chat-model',
          }],
        },
      }],
      bookSettings: [],
      providerSettings: [],
      credentials: [credential('stored-key')],
    })
  })

  it('loads reusable metadata and its plaintext API key as one profile', async () => {
    const store = useCustomAiProfileStore()

    expect(await store.load()).toBe(true)
    expect(mocks.getV2Settings).toHaveBeenCalledWith([
      'custom_ai_profiles',
      'custom_ai_profile',
    ])
    expect(store.byKind('chat')).toEqual([{
      id: PROFILE_ID,
      name: '主翻译服务',
      kind: 'chat',
      baseUrl: 'https://gateway.example.com/v1',
      apiKey: 'stored-key',
      model: 'chat-model',
    }])
  })

  it('keeps API keys out of setting metadata and updates them through credentials', async () => {
    mocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [{ domain: 'custom_ai_profiles', revision: 3 }],
      bookSettings: [],
      providerSettings: [],
      credentials: [credential('replacement-key', 2)],
      prompts: [],
    })
    const store = useCustomAiProfileStore()
    expect(await store.load()).toBe(true)

    expect(await store.update({
      ...store.byKind('chat')[0]!,
      apiKey: ' replacement-key ',
      model: 'replacement-model',
    })).toBe(true)

    const transaction = mocks.saveV2SettingsTransaction.mock.calls[0]?.[0]
    expect(transaction.settings[0].payload.profiles[0]).toEqual({
      id: PROFILE_ID,
      name: '主翻译服务',
      kind: 'chat',
      baseUrl: 'https://gateway.example.com/v1',
      model: 'replacement-model',
    })
    expect(transaction.credentialEdits).toEqual([expect.objectContaining({
      domain: 'custom_ai_profile',
      provider: PROFILE_ID,
      secret: { api_key: 'replacement-key' },
      baseRevision: 1,
      credentialId: 'credential-1',
    })])
    expect(store.byKind('chat')[0]?.apiKey).toBe('replacement-key')
  })
})
