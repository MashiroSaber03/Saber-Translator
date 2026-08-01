import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, postMock, deleteMock, cancelMock, eventsMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
  deleteMock: vi.fn(),
  cancelMock: vi.fn(),
  eventsMock: vi.fn(),
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
    events: eventsMock,
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

describe('plugin agent v2 api', () => {
  beforeEach(() => {
    vi.resetModules()
    getMock.mockReset()
    postMock.mockReset()
    deleteMock.mockReset()
    cancelMock.mockReset()
    eventsMock.mockReset()
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

    expect(postMock).toHaveBeenCalledWith('/api/v2/plugin-agent/sessions', { mode: 'create' })
    expect(result.session_id).toBe('session-1')
  })

  it('uses the v2 session and durable job routes', async () => {
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
    })
    await lockPluginAgentTarget(scopedSession.session_id, {
      plugin_id: 'auto_plugin',
      display_name: 'Auto Plugin',
      supported_steps: ['ocr'],
      supported_modes: ['standard'],
    })
    await startPluginAgentExecution(scopedSession.session_id)
    await cancelPluginAgentExecution('job-1')
    await deletePluginAgentSession(scopedSession.session_id)

    expect(getMock).toHaveBeenCalledWith(`/api/v2/plugin-agent/sessions/${encoded}`)
    expect(postMock).toHaveBeenNthCalledWith(
      1,
      `/api/v2/plugin-agent/sessions/${encoded}/messages`,
      { content: 'Create a plugin' }
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
      }
    )
    expect(postMock).toHaveBeenNthCalledWith(
      3,
      `/api/v2/plugin-agent/sessions/${encoded}/start`,
      {},
      {
        headers: {
          'Idempotency-Key': 'agent-idempotency-key',
        },
      }
    )
    expect(cancelMock).toHaveBeenCalledWith('job-1')
    expect(deleteMock).toHaveBeenCalledWith(`/api/v2/plugin-agent/sessions/${encoded}`)
  })

  it('loads durable Worker events from the global job journal', async () => {
    postMock.mockResolvedValue({
      session: { ...session, run_state: 'running' },
      batchId: 'batch-1',
      jobId: 'job-1',
    })
    eventsMock.mockResolvedValue({
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
    const { listPluginAgentJobEvents } = await import('@/api/pluginAgent')
    const result = await listPluginAgentJobEvents('job-1', 4)

    expect(eventsMock).toHaveBeenCalledWith('job-1', { after: 4, limit: 200 })
    expect(result).toEqual({
      cursor: 6,
      events: [
        {
          id: 5,
          eventKey: 'job:5',
          type: 'assistant_delta',
          payload: { delta: 'hello' },
          timestamp: '2026-01-01T00:00:01Z',
        },
      ],
    })
  })

  it('closes the running session when the durable job ends with an error', async () => {
    postMock.mockResolvedValue({
      session: { ...session, run_state: 'running' },
      batchId: 'batch-1',
      jobId: 'job-1',
    })
    eventsMock.mockResolvedValue({
      items: [
        {
          eventId: 7,
          type: 'job_finished',
          payload: { status: 'completed_with_errors' },
          createdAt: '2026-01-01T00:00:03Z',
        },
      ],
    })
    const { listPluginAgentJobEvents } = await import('@/api/pluginAgent')
    const result = await listPluginAgentJobEvents('job-1')

    expect(result.events).toEqual([
      {
        id: 7,
        eventKey: 'job:7',
        type: 'error',
        payload: {
          run_state: 'failed',
          message: '插件 Agent 执行未成功，请在任务中心查看错误详情',
        },
        timestamp: '2026-01-01T00:00:03Z',
      },
    ])
  })
})
