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

import {
  getTranslateWorkflowPreferences,
  saveTranslateWorkflowPreferences,
} from '@/api/config'

describe('translate workflow preferences api', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
  })

  it('loads workflow preferences from the independent route', async () => {
    getMock.mockResolvedValue({
      success: true,
      preferences: {
        rememberWorkflowModeEnabled: true,
        lastWorkflowMode: 'clear-all',
      },
    })

    await getTranslateWorkflowPreferences()

    expect(getMock).toHaveBeenCalledWith('/api/config/translate-workflow-preferences')
  })

  it('saves workflow preferences to the independent route', async () => {
    const preferences = {
      rememberWorkflowModeEnabled: true,
      lastWorkflowMode: 'delete-current' as const,
    }
    postMock.mockResolvedValue({ success: true, preferences })

    await saveTranslateWorkflowPreferences(preferences)

    expect(postMock).toHaveBeenCalledWith('/api/config/translate-workflow-preferences', preferences)
  })
})
