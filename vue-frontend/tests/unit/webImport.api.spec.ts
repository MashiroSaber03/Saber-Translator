import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, postMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
  },
}))

describe('backend-owned web import diagnostics', () => {
  beforeEach(() => {
    postMock.mockReset()
    getMock.mockReset()
    postMock.mockResolvedValue({ success: true })
  })

  it('loads one bounded draft candidate page instead of draining every cursor', async () => {
    getMock.mockResolvedValue({ items: [], nextCursor: 200 })
    const { listWebImportDraftPages } = await import('@/api/v2/webImport')

    const result = await listWebImportDraftPages('draft/id', { cursor: 100, limit: 100 })

    expect(getMock).toHaveBeenCalledTimes(1)
    expect(getMock).toHaveBeenCalledWith(
      '/api/v2/web-import/drafts/draft%2Fid/pages?cursor=100&limit=100',
    )
    expect(result.nextCursor).toBe(200)
  })

  it('tests explicit Firecrawl and agent credentials through v2 diagnostics', async () => {
    const {
      testAgentConnection,
      testFirecrawlConnection,
    } = await import('@/api/v2/webImport')

    await testFirecrawlConnection('firecrawl-key')
    await testAgentConnection(
      'deepseek',
      'agent-key',
      'https://agent.example/v1',
      'deepseek-chat',
    )

    expect(postMock).toHaveBeenNthCalledWith(
      1,
      '/api/v2/connection-tests/firecrawl',
      { secret: { api_key: 'firecrawl-key' } },
    )
    expect(postMock).toHaveBeenNthCalledWith(
      2,
      '/api/v2/connection-tests/web_import_agent',
      {
        provider: 'deepseek',
        baseUrl: 'https://agent.example/v1',
        model: 'deepseek-chat',
        secret: { api_key: 'agent-key' },
      },
    )
  })

  it('lets backend diagnostics resolve already stored credentials', async () => {
    const {
      testAgentConnection,
      testFirecrawlConnection,
    } = await import('@/api/v2/webImport')

    await testFirecrawlConnection('')
    await testAgentConnection('deepseek', '', '', 'deepseek-chat')

    expect(postMock).toHaveBeenNthCalledWith(
      1,
      '/api/v2/connection-tests/firecrawl',
      { domain: 'web_import_firecrawl' },
    )
    expect(postMock).toHaveBeenNthCalledWith(
      2,
      '/api/v2/connection-tests/web_import_agent',
      {
        provider: 'deepseek',
        baseUrl: undefined,
        model: 'deepseek-chat',
        domain: 'web_import_agent',
      },
    )
  })
})
