import { beforeEach, describe, expect, it, vi } from 'vitest'

const { postMock } = vi.hoisted(() => ({
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    post: postMock,
  },
}))

describe('backend-owned web import diagnostics', () => {
  beforeEach(() => {
    postMock.mockReset()
    postMock.mockResolvedValue({ success: true })
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
      { secret: { apiKey: 'firecrawl-key' } },
    )
    expect(postMock).toHaveBeenNthCalledWith(
      2,
      '/api/v2/connection-tests/web_import_agent',
      {
        provider: 'deepseek',
        baseUrl: 'https://agent.example/v1',
        model: 'deepseek-chat',
        secret: { apiKey: 'agent-key' },
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
