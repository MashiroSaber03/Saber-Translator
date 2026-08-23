import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  deleteBrowserCredential: vi.fn(),
  deleteV2Credential: vi.fn(),
  getV2Settings: vi.fn(),
  saveV2SettingsTransaction: vi.fn(),
  prepareBrowserCredentialTransaction: vi.fn(),
  restoreBrowserCredentialLeases: vi.fn(),
}))

vi.mock('@/api/v2/settings', () => ({
  deleteV2Credential: mocks.deleteV2Credential,
  getV2Settings: mocks.getV2Settings,
  saveV2SettingsTransaction: mocks.saveV2SettingsTransaction,
}))

vi.mock('@/services/browserCredentials', () => ({
  deleteBrowserCredential: mocks.deleteBrowserCredential,
  prepareBrowserCredentialTransaction: mocks.prepareBrowserCredentialTransaction,
  restoreBrowserCredentialLeases: mocks.restoreBrowserCredentialLeases,
}))

import { useCustomAiProfileStore } from '@/stores/customAiProfileStore'

const PROFILE_ID = '11111111-1111-4111-8111-111111111111'
const SECOND_PROFILE_ID = '22222222-2222-4222-8222-222222222222'

function credential(apiKey: string, revision = 1, profileId = PROFILE_ID) {
  return {
    credentialId: profileId === PROFILE_ID ? 'credential-1' : 'credential-2',
    credentialVersionId: `credential-version-${profileId}-${revision}`,
    currentVersion: revision,
    domain: 'custom_ai_profile',
    hasKey: true,
    provider: profileId,
    revision,
    secret: { api_key: apiKey },
  }
}

describe('useCustomAiProfileStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    Object.values(mocks).forEach(mock => mock.mockReset())
    mocks.deleteBrowserCredential.mockResolvedValue(false)
    mocks.deleteV2Credential.mockResolvedValue({ deleted: true })
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
            kind: 'chatVision',
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
    expect(store.byKind('chatVision')).toEqual([{
      id: PROFILE_ID,
      name: '主翻译服务',
      kind: 'chatVision',
      baseUrl: 'https://gateway.example.com/v1',
      apiKey: 'stored-key',
      model: 'chat-model',
    }])
  })

  it('clears plaintext profiles when the application user context changes', async () => {
    const store = useCustomAiProfileStore()
    expect(await store.load()).toBe(true)

    store.reset()

    expect(store.profiles).toEqual([])
    expect(store.isLoaded).toBe(false)
    expect(store.error).toBeNull()
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
      ...store.byKind('chatVision')[0]!,
      apiKey: ' replacement-key ',
      model: 'replacement-model',
    })).toBe(true)

    const transaction = mocks.saveV2SettingsTransaction.mock.calls[0]?.[0]
    expect(transaction.settings[0].payload.profiles[0]).toEqual({
      id: PROFILE_ID,
      name: '主翻译服务',
      kind: 'chatVision',
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
    expect(store.byKind('chatVision')[0]?.apiKey).toBe('replacement-key')
  })

  it('keeps a newly created profile loaded with its plaintext API key', async () => {
    mocks.getV2Settings.mockResolvedValue({
      settings: [{
        domain: 'custom_ai_profiles',
        revision: 1,
        schemaVersion: 1,
        payload: { profiles: [] },
      }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })
    mocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [{ domain: 'custom_ai_profiles', revision: 2 }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
      prompts: [],
    })
    const store = useCustomAiProfileStore()

    const profile = await store.create({
      name: '新服务',
      kind: 'chatVision',
      baseUrl: 'https://new.example.com/v1/',
      apiKey: 'new-key',
      model: 'new-model',
    })

    expect(profile).not.toBeNull()
    expect(store.isLoaded).toBe(true)
    expect(store.byKind('chatVision')[0]).toEqual(expect.objectContaining({
      name: '新服务',
      baseUrl: 'https://new.example.com/v1',
      apiKey: 'new-key',
      model: 'new-model',
    }))
  })

  it('applies an edit to the latest authoritative list without losing concurrent additions', async () => {
    mocks.getV2Settings
      .mockResolvedValueOnce({
        settings: [{
          domain: 'custom_ai_profiles',
          revision: 2,
          schemaVersion: 1,
          payload: {
            profiles: [{
              id: PROFILE_ID,
              name: '主翻译服务',
              kind: 'chatVision',
              baseUrl: 'https://gateway.example.com/v1',
              model: 'chat-model',
            }],
          },
        }],
        bookSettings: [],
        providerSettings: [],
        credentials: [credential('stored-key', 1)],
      })
      .mockResolvedValueOnce({
        settings: [{
          domain: 'custom_ai_profiles',
          revision: 5,
          schemaVersion: 1,
          payload: {
            profiles: [
              {
                id: PROFILE_ID,
                name: '主翻译服务',
                kind: 'chatVision',
                baseUrl: 'https://gateway.example.com/v1',
                model: 'chat-model',
              },
              {
                id: SECOND_PROFILE_ID,
                name: '并发新增服务',
                kind: 'embedding',
                baseUrl: 'https://embedding.example.com/v1',
                model: 'embedding-model',
              },
            ],
          },
        }],
        bookSettings: [],
        providerSettings: [],
        credentials: [
          credential('stored-key', 4),
          credential('embedding-key', 1, SECOND_PROFILE_ID),
        ],
      })
    mocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [{ domain: 'custom_ai_profiles', revision: 6 }],
      bookSettings: [],
      providerSettings: [],
      credentials: [credential('replacement-key', 5)],
      prompts: [],
    })
    const store = useCustomAiProfileStore()
    expect(await store.load()).toBe(true)

    expect(await store.update({
      ...store.byKind('chatVision')[0]!,
      apiKey: 'replacement-key',
    })).toBe(true)

    const transaction = mocks.saveV2SettingsTransaction.mock.calls[0]?.[0]
    expect(transaction.settings[0].baseRevision).toBe(5)
    expect(transaction.settings[0].payload.profiles).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: PROFILE_ID }),
      expect.objectContaining({ id: SECOND_PROFILE_ID, name: '并发新增服务' }),
    ]))
    expect(transaction.credentialEdits[0].baseRevision).toBe(4)
  })

  it('allows an unrelated profile to be saved when a browser-local key is unavailable', async () => {
    mocks.getV2Settings.mockResolvedValue({
      settings: [{
        domain: 'custom_ai_profiles',
        revision: 2,
        schemaVersion: 1,
        payload: {
          profiles: [
            {
              id: PROFILE_ID,
              name: '本浏览器已有服务',
              kind: 'chatVision',
              baseUrl: 'https://gateway.example.com/v1',
              model: 'chat-model',
            },
            {
              id: SECOND_PROFILE_ID,
              name: '其他浏览器的服务',
              kind: 'embedding',
              baseUrl: 'https://embedding.example.com/v1',
              model: 'embedding-model',
            },
          ],
        },
      }],
      bookSettings: [],
      providerSettings: [],
      credentials: [credential('stored-key')],
    })
    mocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [{ domain: 'custom_ai_profiles', revision: 3 }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
      prompts: [],
    })
    const store = useCustomAiProfileStore()
    expect(await store.load()).toBe(true)

    expect(await store.update({
      ...store.profiles[0]!,
      model: 'replacement-model',
    })).toBe(true)
    expect(mocks.saveV2SettingsTransaction).toHaveBeenCalledOnce()
  })

  it('deletes the unreferenced credential after removing a profile', async () => {
    mocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [{ domain: 'custom_ai_profiles', revision: 3 }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
      prompts: [],
    })
    const store = useCustomAiProfileStore()
    expect(await store.load()).toBe(true)

    expect(await store.remove(PROFILE_ID)).toBe(true)
    expect(mocks.deleteBrowserCredential).toHaveBeenCalledWith(
      'custom_ai_profile',
      PROFILE_ID,
    )
    expect(mocks.deleteV2Credential).toHaveBeenCalledWith('credential-1')
    expect(store.profiles).toEqual([])
  })

  it('removes browser-owned credentials without calling the backend credential repository', async () => {
    mocks.deleteBrowserCredential.mockResolvedValue(true)
    mocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [{ domain: 'custom_ai_profiles', revision: 3 }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
      prompts: [],
    })
    const store = useCustomAiProfileStore()
    expect(await store.load()).toBe(true)

    expect(await store.remove(PROFILE_ID)).toBe(true)
    expect(mocks.deleteV2Credential).not.toHaveBeenCalled()
  })
})
