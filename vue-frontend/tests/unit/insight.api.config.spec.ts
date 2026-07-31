import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, putMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  putMock: vi.fn(),
}))

vi.mock('@/api/client', () => {
  const apiClient = {
    get: getMock,
    put: putMock,
    post: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn(),
    upload: vi.fn(),
  }
  return { apiClient, default: apiClient }
})

const settingsDocument = {
  settings: [{
    domain: 'insight',
    payload: {
      analysis: {
        batch: {
          pagesPerBatch: 5,
          contextBatchCount: 3,
          architecturePreset: 'standard',
          customLayers: [],
        },
      },
      vlm: { provider: 'gemini' },
      chat: { provider: 'gemini', useSameAsVlm: false },
      embedding: { provider: 'openai' },
      reranker: { provider: 'jina' },
      imageGen: { provider: 'gpt2api' },
    },
    revision: 3,
    schemaVersion: 1,
  }],
  bookSettings: [],
  providerSettings: [{
    domain: 'insight_vlm',
    provider: 'gemini',
    payload: { modelName: 'old-model' },
    revision: 4,
    schemaVersion: 1,
    credentialVersionId: 'credential-version-1',
  }],
  credentials: [{
    credentialId: 'credential-1',
    credentialVersionId: 'credential-version-1',
    currentVersion: 1,
    domain: 'insight_vlm',
    hasKey: true,
    provider: 'gemini',
    revision: 5,
  }],
}

describe('insight v2 settings ownership', () => {
  beforeEach(() => {
    vi.resetModules()
    getMock.mockReset()
    putMock.mockReset()
    getMock.mockImplementation((url: string) => {
      if (url === '/api/v2/settings') return Promise.resolve(settingsDocument)
      if (url === '/api/v2/prompts') return Promise.resolve({ items: [] })
      throw new Error(`Unexpected GET ${url}`)
    })
    putMock.mockResolvedValue({
      settings: [],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })
  })

  it('loads Insight configuration from unified v2 settings and prompts', async () => {
    const { getGlobalConfig } = await import('@/api/insight')

    const response = await getGlobalConfig()

    expect(response.success).toBe(true)
    expect(response.config?.vlm).toMatchObject({
      provider: 'gemini',
      model: 'old-model',
      api_key: '',
    })
    expect(getMock).toHaveBeenCalledWith('/api/v2/settings', {
      params: {
        domains: [
          'insight',
          'insight_vlm',
          'insight_chat',
          'insight_embedding',
          'insight_reranker',
          'insight_image_gen',
        ].join(','),
      },
    })
    expect(getMock).toHaveBeenCalledWith('/api/v2/prompts', { params: {} })
  })

  it('rejects an incomplete backend Insight fact instead of applying browser defaults', async () => {
    getMock.mockImplementation((url: string) => {
      if (url === '/api/v2/settings') {
        return Promise.resolve({
          ...settingsDocument,
          settings: [{
            domain: 'insight',
            payload: {},
            revision: 3,
            schemaVersion: 1,
          }],
        })
      }
      if (url === '/api/v2/prompts') return Promise.resolve({ items: [] })
      throw new Error(`Unexpected GET ${url}`)
    })
    const { getGlobalConfig } = await import('@/api/insight')

    await expect(getGlobalConfig()).rejects.toThrow('Insight 设置字段不完整')
  })

  it('saves configuration atomically with CAS revisions and a credential edit', async () => {
    const { getGlobalConfig, saveGlobalConfig } = await import('@/api/insight')
    await getGlobalConfig()

    await saveGlobalConfig({
      vlm: {
        provider: 'gemini',
        api_key: 'replacement-secret',
        model: 'gemini-2.0-flash',
      },
    })

    expect(putMock).toHaveBeenCalledWith(
      '/api/v2/settings/transactions',
      expect.objectContaining({
        settings: [expect.objectContaining({
          domain: 'insight',
          baseRevision: 3,
        })],
        providerSettings: expect.arrayContaining([
          expect.objectContaining({
            domain: 'insight_vlm',
            provider: 'gemini',
            baseRevision: 4,
            credentialEditRef: 'insight:insight_vlm:gemini',
          }),
        ]),
        credentialEdits: [expect.objectContaining({
          domain: 'insight_vlm',
          provider: 'gemini',
          credentialId: 'credential-1',
          baseRevision: 5,
          secret: { api_key: 'replacement-secret' },
        })],
      }),
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
  })

  it('only sends OpenAI execution options for Insight providers that own them', async () => {
    const { getGlobalConfig, saveGlobalConfig } = await import('@/api/insight')
    await getGlobalConfig()

    await saveGlobalConfig({
      vlm: {
        provider: 'siliconflow',
        model: 'vlm-model',
        openai_options: {
          request: { force_json_output: true },
          execution: {
            use_stream: false,
            rpm_limit: 30,
            transport_retries: 2,
            business_retries: 1,
          },
        },
      },
      chat_llm: {
        provider: 'siliconflow',
        model: 'chat-model',
        openai_options: {
          request: { force_json_output: false },
          execution: {
            use_stream: true,
            rpm_limit: 60,
            transport_retries: 3,
            business_retries: 2,
          },
        },
      },
      embedding: {
        provider: 'siliconflow',
        model: 'embedding-model',
      },
      reranker: {
        provider: 'siliconflow',
        model: 'reranker-model',
      },
    })

    const request = putMock.mock.calls.at(-1)?.[1] as {
      providerSettings: Array<{
        domain: string
        payload: Record<string, unknown>
      }>
    }
    const byDomain = Object.fromEntries(
      request.providerSettings.map(row => [row.domain, row.payload]),
    )

    expect(byDomain.insight_vlm).toHaveProperty('openaiOptions')
    expect(byDomain.insight_chat).toHaveProperty('openaiOptions')
    expect(byDomain.insight_embedding).not.toHaveProperty('openaiOptions')
    expect(byDomain.insight_reranker).not.toHaveProperty('openaiOptions')
  })
})
