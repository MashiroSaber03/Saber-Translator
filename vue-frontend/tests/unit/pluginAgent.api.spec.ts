import { beforeEach, describe, expect, it, vi } from 'vitest'

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
  createPluginAgentSession,
  getPluginAgentSettings,
  lockPluginAgentTarget,
} from '@/api/pluginAgent'

describe('plugin agent api', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
    deleteMock.mockReset()
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
})
