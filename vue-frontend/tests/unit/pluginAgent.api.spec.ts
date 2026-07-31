import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  getMock,
  postMock,
  deleteMock,
  cancelMock,
} = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
  deleteMock: vi.fn(),
  cancelMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
    delete: deleteMock,
  },
  default: {
    get: getMock,
    post: postMock,
    delete: deleteMock,
  },
}))

vi.mock('@/api/v2/jobs', () => ({
  jobsApi: {
    cancel: cancelMock,
  },
}))

vi.mock('@/api/v2/content', () => ({
  newIdempotencyKey: () => 'agent-idempotency-key',
}))

const session = {
  session_id: 'session-1',
  mode: 'create',
  run_state: 'ready',
  selected_plugin_id: null,
  pending_target: null,
  locked_target: {
    mode: 'create',
    plugin_id: 'auto_plugin',
    display_name: 'Auto Plugin',
    supported_steps: ['ocr'],
    supported_modes: ['standard'],
    plugin_dir: '',
  },
  messages: [],
  events: [],
  touched_files: [],
  file_previews: {},
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
}

const agentConfig = {
  provider: 'openai-compatible',
  apiKey: 'browser-secret-must-not-be-sent',
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

describe('plugin agent v2 api', () => {
  beforeEach(() => {
    vi.resetModules()
    getMock.mockReset()
    postMock.mockReset()
    deleteMock.mockReset()
    cancelMock.mockReset()
  })

  it('builds the workbench settings from backend plugin data', async () => {
    getMock.mockResolvedValue({ items: [] })
    const { getPluginAgentSettings } = await import('@/api/pluginAgent')

    const result = await getPluginAgentSettings()

    expect(getMock).toHaveBeenCalledWith('/api/v2/plugins')
    expect(result.plugins).toEqual([])
    expect(result.overview.join(' ')).toContain('全局后端队列')
  })

  it('creates planning sessions through the v2 route', async () => {
    postMock.mockResolvedValue({ session })
    const { createPluginAgentSession } = await import('@/api/pluginAgent')

    const result = await createPluginAgentSession({ mode: 'create' })

    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/plugin-agent/sessions',
      { mode: 'create' },
    )
    expect(result.session.session_id).toBe('session-1')
  })

  it('uses v2 session routes and never sends browser provider secrets', async () => {
    const encoded = 'session%2Fgroup%20one'
    const scopedSession = {
      ...session,
      session_id: 'session/group one',
    }
    getMock.mockResolvedValue({ session: scopedSession })
    postMock
      .mockResolvedValueOnce({ session: scopedSession })
      .mockResolvedValueOnce({ session: scopedSession })
      .mockResolvedValueOnce({
        session: { ...scopedSession, run_state: 'running' },
        batchId: 'batch-1',
        jobId: 'job-1',
      })
    cancelMock.mockResolvedValue({ status: 'cancelling' })
    deleteMock.mockResolvedValue({ deleted: true })
    const {
      cancelPluginAgentExecution,
      deletePluginAgentSession,
      getPluginAgentSession,
      lockPluginAgentTarget,
      sendPluginAgentMessage,
      startPluginAgentExecution,
    } = await import('@/api/pluginAgent')

    await getPluginAgentSession(scopedSession.session_id)
    await sendPluginAgentMessage(scopedSession.session_id, {
      content: 'Create a plugin',
      agentConfig,
    })
    await lockPluginAgentTarget(scopedSession.session_id, {
      plugin_id: 'auto_plugin',
      display_name: 'Auto Plugin',
      supported_steps: ['ocr'],
      supported_modes: ['standard'],
    })
    await startPluginAgentExecution(scopedSession.session_id, agentConfig)
    await cancelPluginAgentExecution(scopedSession.session_id)
    await deletePluginAgentSession(scopedSession.session_id)

    expect(getMock).toHaveBeenCalledWith(
      `/api/v2/plugin-agent/sessions/${encoded}`,
    )
    expect(postMock).toHaveBeenNthCalledWith(
      1,
      `/api/v2/plugin-agent/sessions/${encoded}/messages`,
      { content: 'Create a plugin' },
    )
    expect(postMock).toHaveBeenNthCalledWith(
      2,
      `/api/v2/plugin-agent/sessions/${encoded}/lock-target`,
      {
        proposal: {
          plugin_id: 'auto_plugin',
          display_name: 'Auto Plugin',
          supported_steps: ['ocr'],
          supported_modes: ['standard'],
        },
      },
    )
    expect(postMock).toHaveBeenNthCalledWith(
      3,
      `/api/v2/plugin-agent/sessions/${encoded}/start`,
      {},
      {
        headers: {
          'Idempotency-Key': 'agent-idempotency-key',
        },
      },
    )
    expect(cancelMock).toHaveBeenCalledWith('job-1')
    expect(deleteMock).toHaveBeenCalledWith(
      `/api/v2/plugin-agent/sessions/${encoded}`,
    )
    expect(JSON.stringify(postMock.mock.calls)).not.toContain(
      'browser-secret-must-not-be-sent',
    )
  })

  it('loads durable Worker events from the global job journal', async () => {
    postMock.mockResolvedValue({
      session: { ...session, run_state: 'running' },
      batchId: 'batch-1',
      jobId: 'job-1',
    })
    getMock.mockResolvedValue({
      items: [
        {
          eventId: 5,
          type: 'plugin_agent_assistant_delta',
          payload: { delta: 'hello' },
          createdAt: '2026-01-01T00:00:01Z',
        },
        {
          eventId: 6,
          type: 'job_progress',
          payload: { ignored: true },
          createdAt: '2026-01-01T00:00:02Z',
        },
      ],
    })
    const {
      startPluginAgentExecution,
      subscribePluginAgentEvents,
    } = await import('@/api/pluginAgent')
    await startPluginAgentExecution('session-1', agentConfig)
    const onEvent = vi.fn()
    const onError = vi.fn()

    await subscribePluginAgentEvents('session-1', {
      afterId: 4,
      onEvent,
      onError,
    })

    expect(getMock).toHaveBeenCalledWith(
      '/api/v2/jobs/job-1/events?after=0&limit=200',
      { signal: undefined },
    )
    expect(onEvent).toHaveBeenCalledTimes(1)
    expect(onEvent).toHaveBeenCalledWith({
      id: 5,
      type: 'assistant_delta',
      payload: { delta: 'hello' },
      timestamp: '2026-01-01T00:00:01Z',
    })
    expect(onError).not.toHaveBeenCalled()
  })

  it('closes the running session when the durable job ends with an error', async () => {
    postMock.mockResolvedValue({
      session: { ...session, run_state: 'running' },
      batchId: 'batch-1',
      jobId: 'job-1',
    })
    getMock.mockResolvedValue({
      items: [
        {
          eventId: 7,
          type: 'job_finished',
          payload: { status: 'completed_with_errors' },
          createdAt: '2026-01-01T00:00:03Z',
        },
      ],
    })
    const {
      startPluginAgentExecution,
      subscribePluginAgentEvents,
    } = await import('@/api/pluginAgent')
    await startPluginAgentExecution('session-1', agentConfig)
    const onEvent = vi.fn()

    await subscribePluginAgentEvents('session-1', {
      onEvent,
      onError: vi.fn(),
    })

    expect(onEvent).toHaveBeenCalledWith({
      id: 7,
      type: 'error',
      payload: {
        run_state: 'failed',
        message: '插件 Agent 执行未成功，请在任务中心查看错误详情',
      },
      timestamp: '2026-01-01T00:00:03Z',
    })
  })

  it('does not report polling errors after an abort', async () => {
    postMock.mockResolvedValue({
      session: { ...session, run_state: 'running' },
      batchId: 'batch-1',
      jobId: 'job-1',
    })
    getMock.mockRejectedValue(new DOMException('Aborted', 'AbortError'))
    const {
      startPluginAgentExecution,
      subscribePluginAgentEvents,
    } = await import('@/api/pluginAgent')
    await startPluginAgentExecution('session-1', agentConfig)
    const controller = new AbortController()
    const onError = vi.fn()
    controller.abort()

    await subscribePluginAgentEvents('session-1', {
      signal: controller.signal,
      onEvent: vi.fn(),
      onError,
    })

    expect(onError).not.toHaveBeenCalled()
  })
})
