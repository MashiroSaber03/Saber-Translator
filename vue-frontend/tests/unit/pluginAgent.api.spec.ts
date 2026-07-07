import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, postMock, deleteMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
  deleteMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
    delete: deleteMock,
  },
}))

import {
  cancelPluginAgentExecution,
  createPluginAgentSession,
  deletePluginAgentSession,
  getPluginAgentSession,
  getPluginAgentSettings,
  lockPluginAgentTarget,
  sendPluginAgentMessage,
  startPluginAgentExecution,
  subscribePluginAgentEvents,
} from '@/api/pluginAgent'

function streamFromChunks(chunks: string[]) {
  const encoder = new TextEncoder()
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk))
      }
      controller.close()
    },
  })
}

describe('plugin agent api', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
    deleteMock.mockReset()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('loads plugin agent settings from the system route', async () => {
    getMock.mockResolvedValue({ success: true })

    await getPluginAgentSettings()

    expect(getMock).toHaveBeenCalledWith('/api/plugins/agent/settings')
  })

  it('creates a plugin agent session via the session route', async () => {
    postMock.mockResolvedValue({ success: true })

    await createPluginAgentSession({ mode: 'create' })

    expect(postMock).toHaveBeenCalledWith('/api/plugins/agent/sessions', { mode: 'create' })
  })

  it('locks a pending target through the lock-target route', async () => {
    postMock.mockResolvedValue({ success: true })

    await lockPluginAgentTarget('session-1', {
      plugin_id: 'auto_plugin',
      display_name: 'Auto Plugin',
      supported_steps: ['ocr'],
      supported_modes: ['standard'],
    })

    expect(postMock).toHaveBeenCalledWith('/api/plugins/agent/sessions/session-1/lock-target', {
      proposal: {
        plugin_id: 'auto_plugin',
        display_name: 'Auto Plugin',
        supported_steps: ['ocr'],
        supported_modes: ['standard'],
      },
    })
  })

  it('encodes session ids through every session-scoped endpoint', async () => {
    getMock.mockResolvedValue({ success: true })
    postMock.mockResolvedValue({ success: true })
    deleteMock.mockResolvedValue({ success: true })

    const agentConfig = {
      provider: 'openai-compatible',
      apiKey: 'agent-key',
      modelName: 'agent-model',
      customBaseUrl: 'https://agent.example.test/v1',
      openaiOptions: {
        request: {
          forceJsonOutput: true,
          temperature: 0.4,
          extraBody: { top_p: 0.8 },
        },
        execution: {
          useStream: true,
          rpmLimit: 12,
          transportRetries: 2,
          businessRetries: 3,
        },
      },
    }
    const sessionId = 'session/group one'
    const encodedSessionId = 'session%2Fgroup%20one'

    await getPluginAgentSession(sessionId)
    await deletePluginAgentSession(sessionId)
    await sendPluginAgentMessage(sessionId, {
      content: 'Create a plugin',
      agentConfig,
    })
    await lockPluginAgentTarget(sessionId, {
      plugin_id: 'auto_plugin',
      display_name: 'Auto Plugin',
      supported_steps: ['ocr'],
      supported_modes: ['standard'],
    })
    await startPluginAgentExecution(sessionId, agentConfig)
    await cancelPluginAgentExecution(sessionId)

    expect(getMock).toHaveBeenCalledWith(`/api/plugins/agent/sessions/${encodedSessionId}`)
    expect(deleteMock).toHaveBeenCalledWith(`/api/plugins/agent/sessions/${encodedSessionId}`)
    expect(postMock).toHaveBeenNthCalledWith(1, `/api/plugins/agent/sessions/${encodedSessionId}/messages`, {
      content: 'Create a plugin',
      agent_config: {
        provider: 'openai-compatible',
        api_key: 'agent-key',
        model_name: 'agent-model',
        custom_base_url: 'https://agent.example.test/v1',
        openai_options: {
          request: {
            force_json_output: true,
            temperature: 0.4,
            extra_body: { top_p: 0.8 },
          },
          execution: {
            use_stream: true,
            rpm_limit: 12,
            transport_retries: 2,
            business_retries: 3,
          },
        },
      },
    })
    expect(postMock).toHaveBeenNthCalledWith(2, `/api/plugins/agent/sessions/${encodedSessionId}/lock-target`, {
      proposal: {
        plugin_id: 'auto_plugin',
        display_name: 'Auto Plugin',
        supported_steps: ['ocr'],
        supported_modes: ['standard'],
      },
    })
    expect(postMock).toHaveBeenNthCalledWith(3, `/api/plugins/agent/sessions/${encodedSessionId}/start`, {
      agent_config: {
        provider: 'openai-compatible',
        api_key: 'agent-key',
        model_name: 'agent-model',
        custom_base_url: 'https://agent.example.test/v1',
        openai_options: {
          request: {
            force_json_output: true,
            temperature: 0.4,
            extra_body: { top_p: 0.8 },
          },
          execution: {
            use_stream: true,
            rpm_limit: 12,
            transport_retries: 2,
            business_retries: 3,
          },
        },
      },
    })
    expect(postMock).toHaveBeenNthCalledWith(4, `/api/plugins/agent/sessions/${encodedSessionId}/cancel`)
  })

  it('does not report an error when the event stream fetch is aborted', async () => {
    const controller = new AbortController()
    const onError = vi.fn()
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new DOMException('Aborted', 'AbortError')))

    controller.abort()

    await expect(subscribePluginAgentEvents('session-1', {
      signal: controller.signal,
      onEvent: vi.fn(),
      onError,
    })).resolves.toBeUndefined()

    expect(onError).not.toHaveBeenCalled()
  })

  it('subscribes to plugin agent SSE events through the shared reader', async () => {
    const onEvent = vi.fn()
    const onError = vi.fn()
    const fetchMock = vi.fn().mockResolvedValue(new Response(streamFromChunks([
      'event: assistant_delta\n',
      'data: {"id":1,"type":"state","payload":{"delta":"hel',
      'lo"},"timestamp":"now"}\n\n',
    ]), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await subscribePluginAgentEvents('session-1', {
      afterId: 4,
      onEvent,
      onError,
    })

    expect(fetchMock).toHaveBeenCalledWith('/api/plugins/agent/sessions/session-1/events?after_id=4', {
      method: 'GET',
      signal: undefined,
      headers: {
        Accept: 'text/event-stream',
      },
    })
    expect(onEvent).toHaveBeenCalledWith({
      id: 1,
      type: 'assistant_delta',
      payload: { delta: 'hello' },
      timestamp: 'now',
    })
    expect(onError).not.toHaveBeenCalled()
  })
})
