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
  settings: [
    {
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
    },
  ],
  bookSettings: [],
  providerSettings: [
    {
      domain: 'insight_vlm',
      provider: 'gemini',
      payload: {
        modelName: 'old-model',
        customBaseUrl: '',
        openaiOptions: {
          request: {
            force_json_output: false,
            temperature: 0.3,
            extra_body: {},
          },
          execution: {
            use_stream: true,
            rpm_limit: 0,
            transport_retries: 1,
            business_retries: 0,
          },
        },
        imageMaxSize: 0,
      },
      revision: 4,
      schemaVersion: 1,
      credentialVersionId: 'credential-version-1',
    },
  ],
  credentials: [
    {
      credentialId: 'credential-1',
      credentialVersionId: 'credential-version-1',
      currentVersion: 1,
      domain: 'insight_vlm',
      hasKey: true,
      provider: 'gemini',
      revision: 5,
      secret: { api_key: 'stored-gemini-key' },
    },
  ],
}

const insightFactoryPrompts = [
  ['batch_analysis', '默认批次分析'],
  ['segment_summary', '默认段落总结'],
  ['chapter_summary', '默认章节总结'],
  ['qa_response', '默认问答响应'],
].map(([type, name]) => ({
  id: `factory-${type}`,
  name,
  content: `${name}内容`,
  type,
  revision: 1,
  isFactoryDefault: true,
}))

describe('insight v2 settings ownership', () => {
  beforeEach(() => {
    vi.resetModules()
    getMock.mockReset()
    putMock.mockReset()
    getMock.mockImplementation((url: string) => {
      if (url === '/api/v2/settings') return Promise.resolve(settingsDocument)
      if (url === '/api/v2/prompts') return Promise.resolve({ items: insightFactoryPrompts })
      throw new Error(`Unexpected GET ${url}`)
    })
    putMock.mockResolvedValue({
      settings: [],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
      prompts: [],
    })
  })

  it('loads Insight configuration from unified v2 settings and prompts', async () => {
    const { getGlobalConfig } = await import('@/api/insight')

    const config = await getGlobalConfig()

    expect(config.config.vlm).toMatchObject({
      provider: 'gemini',
      model: 'old-model',
      apiKey: 'stored-gemini-key',
    })
    expect(config.providerDrafts.vlm.gemini).toMatchObject({
      model: 'old-model',
      imageMaxSize: 0,
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

  it('uses shared current provider defaults before a provider draft is first saved', async () => {
    getMock.mockImplementation((url: string) => {
      if (url === '/api/v2/settings') {
        return Promise.resolve({ ...settingsDocument, providerSettings: [] })
      }
      if (url === '/api/v2/prompts') return Promise.resolve({ items: insightFactoryPrompts })
      throw new Error(`Unexpected GET ${url}`)
    })
    const { getGlobalConfig } = await import('@/api/insight')

    const snapshot = await getGlobalConfig()

    expect(snapshot.config.vlm.model).toBe('gemini-2.0-flash')
    expect(snapshot.config.llm.model).toBe('gemini-2.0-flash')
    expect(snapshot.config.embedding.model).toBe('text-embedding-3-small')
    expect(snapshot.config.reranker.model).toBe('jina-reranker-v2-base-multilingual')
    expect(snapshot.config.imageGen.model).toBe('gpt-image-2')
    expect(snapshot.providerDrafts.vlm.gemini?.model).toBe('gemini-2.0-flash')
  })

  it('reads the prompt catalog from the backend for each independent view', async () => {
    getMock.mockImplementation((url: string) => {
      if (url === '/api/v2/settings') return Promise.resolve(settingsDocument)
      if (url === '/api/v2/prompts') {
        return Promise.resolve({
          items: [
            {
              id: 'factory-batch',
              name: '默认批次分析',
              content: '默认内容',
              type: 'batch_analysis',
              revision: 1,
              isFactoryDefault: true,
            },
            {
              id: 'saved-batch',
              name: '自定义批次分析',
              content: '自定义内容',
              type: 'batch_analysis',
              revision: 1,
              isFactoryDefault: false,
            },
            ...(['segment_summary', 'chapter_summary', 'qa_response'] as const).map(type => ({
              id: `factory-${type}`,
              name: `默认 ${type}`,
              content: `${type} 默认内容`,
              type,
              revision: 1,
              isFactoryDefault: true,
            })),
          ],
        })
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    const { getDefaultPrompts, getGlobalConfig, getPromptsLibrary } = await import('@/api/insight')

    await getGlobalConfig()
    const defaults = await getDefaultPrompts()
    const library = await getPromptsLibrary()

    expect(defaults.batch_analysis).toBe('默认内容')
    expect(library).toEqual([
      expect.objectContaining({ id: 'saved-batch', name: '自定义批次分析' }),
    ])
    expect(getMock.mock.calls.filter(([url]) => url === '/api/v2/prompts')).toHaveLength(3)
  })

  it('rejects an incomplete factory prompt catalog', async () => {
    getMock.mockImplementation((url: string) => {
      if (url === '/api/v2/prompts') {
        return Promise.resolve({
          items: [
            {
              id: 'factory-batch',
              name: '默认批次分析',
              content: '默认内容',
              type: 'batch_analysis',
              revision: 1,
              isFactoryDefault: true,
            },
          ],
        })
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    const { getDefaultPrompts } = await import('@/api/insight')

    await expect(getDefaultPrompts()).rejects.toThrow('后端默认提示词不存在：segment_summary')
  })

  it('rejects an incomplete backend Insight fact instead of applying browser defaults', async () => {
    getMock.mockImplementation((url: string) => {
      if (url === '/api/v2/settings') {
        return Promise.resolve({
          ...settingsDocument,
          settings: [
            {
              domain: 'insight',
              payload: {},
              revision: 3,
              schemaVersion: 1,
            },
          ],
        })
      }
      if (url === '/api/v2/prompts') return Promise.resolve({ items: [] })
      throw new Error(`Unexpected GET ${url}`)
    })
    const { getGlobalConfig } = await import('@/api/insight')

    await expect(getGlobalConfig()).rejects.toThrow('Insight 设置字段不完整')
  })

  it('rejects malformed current Insight fields instead of coercing them', async () => {
    getMock.mockImplementation((url: string) => {
      if (url === '/api/v2/settings') {
        return Promise.resolve({
          ...settingsDocument,
          settings: [
            {
              ...settingsDocument.settings[0],
              payload: {
                ...settingsDocument.settings[0]!.payload,
                analysis: {
                  batch: {
                    ...settingsDocument.settings[0]!.payload.analysis.batch,
                    pagesPerBatch: '5',
                  },
                },
              },
            },
          ],
        })
      }
      if (url === '/api/v2/prompts') return Promise.resolve({ items: [] })
      throw new Error(`Unexpected GET ${url}`)
    })
    const { getGlobalConfig } = await import('@/api/insight')

    await expect(getGlobalConfig()).rejects.toThrow('Insight 批量设置字段类型无效')
  })

  it('saves configuration atomically with CAS revisions and a credential edit', async () => {
    const { getGlobalConfig, saveGlobalConfig } = await import('@/api/insight')
    const current = await getGlobalConfig()

    const saved = await saveGlobalConfig({
      ...current,
      config: {
        ...current.config,
        vlm: {
          ...current.config.vlm,
          provider: 'gemini',
          apiKey: 'replacement-secret',
          model: 'gemini-2.0-flash',
        },
      },
    })

    expect(putMock).toHaveBeenCalledWith(
      '/api/v2/settings/transactions',
      expect.objectContaining({
        settings: [
          expect.objectContaining({
            domain: 'insight',
            baseRevision: 3,
          }),
        ],
        providerSettings: expect.arrayContaining([
          expect.objectContaining({
            domain: 'insight_vlm',
            provider: 'gemini',
            baseRevision: 4,
            credentialEditRef: 'insight:insight_vlm:gemini',
          }),
        ]),
        credentialEdits: [
          expect.objectContaining({
            domain: 'insight_vlm',
            provider: 'gemini',
            credentialId: 'credential-1',
            baseRevision: 5,
            secret: { api_key: 'replacement-secret' },
          }),
        ],
      }),
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(saved.config.vlm.apiKey).toBe('replacement-secret')
    expect(getMock.mock.calls.filter(([url]) => url === '/api/v2/settings')).toHaveLength(2)
    expect(getMock.mock.calls.filter(([url]) => url === '/api/v2/prompts')).toHaveLength(2)
  })

  it('only sends OpenAI execution options for Insight providers that own them', async () => {
    const { getGlobalConfig, saveGlobalConfig } = await import('@/api/insight')
    const current = await getGlobalConfig()

    await saveGlobalConfig({
      ...current,
      config: {
        ...current.config,
        vlm: {
          ...current.config.vlm,
          provider: 'siliconflow',
          model: 'vlm-model',
          openaiOptions: {
            request: { forceJsonOutput: true, temperature: 0.3 },
            execution: {
              useStream: false,
              rpmLimit: 30,
              transportRetries: 2,
              businessRetries: 1,
            },
          },
        },
        llm: {
          ...current.config.llm,
          provider: 'siliconflow',
          model: 'chat-model',
          openaiOptions: {
            request: { forceJsonOutput: false },
            execution: {
              useStream: true,
              rpmLimit: 60,
              transportRetries: 3,
              businessRetries: 2,
            },
          },
        },
        embedding: {
          ...current.config.embedding,
          provider: 'siliconflow',
          model: 'embedding-model',
        },
        reranker: {
          ...current.config.reranker,
          provider: 'siliconflow',
          model: 'reranker-model',
        },
      },
    })

    const request = putMock.mock.calls.at(-1)?.[1] as {
      providerSettings: Array<{
        domain: string
        payload: Record<string, unknown>
      }>
    }
    const byDomain = Object.fromEntries(
      request.providerSettings.map(row => [row.domain, row.payload])
    )

    expect(byDomain.insight_vlm).toHaveProperty('openaiOptions')
    expect(byDomain.insight_chat).toHaveProperty('openaiOptions')
    expect(byDomain.insight_embedding).not.toHaveProperty('openaiOptions')
    expect(byDomain.insight_reranker).not.toHaveProperty('openaiOptions')
  })

  it('persists inactive provider drafts together with the active selections', async () => {
    putMock.mockResolvedValue({
      settings: [],
      bookSettings: [],
      providerSettings: [],
      credentials: [
        {
          credentialId: 'credential-openai',
          credentialVersionId: 'credential-openai-version',
          currentVersion: 1,
          domain: 'insight_vlm',
          hasKey: true,
          provider: 'openai',
          revision: 1,
          secret: { api_key: 'inactive-secret' },
        },
      ],
      prompts: [],
    })
    const { getGlobalConfig, saveGlobalConfig } = await import('@/api/insight')
    const current = await getGlobalConfig()
    current.providerDrafts.vlm.openai = {
      apiKey: 'inactive-secret',
      model: 'gpt-4.1-mini',
      baseUrl: '',
      openaiOptions: {
        request: { forceJsonOutput: true, temperature: 0.3 },
        execution: {
          useStream: false,
          rpmLimit: 10,
          transportRetries: 2,
          businessRetries: 1,
        },
      },
      imageMaxSize: 1536,
    }

    const saved = await saveGlobalConfig(current)

    const request = putMock.mock.calls.at(-1)?.[1] as {
      providerSettings: Array<{
        domain: string
        provider: string
        payload: Record<string, unknown>
      }>
      credentialEdits: Array<{
        domain: string
        provider: string
        secret: Record<string, unknown>
      }>
    }
    expect(request.providerSettings).toContainEqual(
      expect.objectContaining({
        domain: 'insight_vlm',
        provider: 'openai',
        payload: expect.objectContaining({
          modelName: 'gpt-4.1-mini',
          imageMaxSize: 1536,
        }),
      })
    )
    expect(request.credentialEdits).toContainEqual(
      expect.objectContaining({
        domain: 'insight_vlm',
        provider: 'openai',
        secret: { api_key: 'inactive-secret' },
      })
    )
    expect(saved.providerDrafts.vlm.openai?.apiKey).toBe('inactive-secret')
  })

  it('commits changed factory prompts in the same settings transaction', async () => {
    const factoryPrompt = {
      id: 'factory-batch',
      name: '默认批次分析',
      content: '旧提示词',
      type: 'batch_analysis',
      revision: 7,
      isFactoryDefault: true,
    }
    getMock.mockImplementation((url: string) => {
      if (url === '/api/v2/settings') return Promise.resolve(settingsDocument)
      if (url === '/api/v2/prompts') {
        return Promise.resolve({
          items: [
            factoryPrompt,
            ...insightFactoryPrompts.filter(prompt => prompt.type !== 'batch_analysis'),
          ],
        })
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    putMock.mockResolvedValue({
      settings: [],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
      prompts: [{ ...factoryPrompt, content: '新提示词', revision: 8 }],
    })
    const { getGlobalConfig, saveGlobalConfig } = await import('@/api/insight')
    const current = await getGlobalConfig()

    await saveGlobalConfig({
      ...current,
      config: {
        ...current.config,
        prompts: { ...current.config.prompts, batch_analysis: '新提示词' },
      },
    })

    expect(putMock).toHaveBeenCalledTimes(1)
    expect(putMock).toHaveBeenCalledWith(
      '/api/v2/settings/transactions',
      expect.objectContaining({
        promptEdits: [
          {
            id: 'factory-batch',
            name: '默认批次分析',
            content: '新提示词',
            baseRevision: 7,
          },
        ],
      }),
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
  })
})
