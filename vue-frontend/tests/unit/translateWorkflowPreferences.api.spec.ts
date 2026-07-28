import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, patchMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  patchMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    patch: patchMock,
  },
}))

import {
  getTranslateWorkflowPreferences,
  saveTranslateWorkflowPreferences,
} from '@/api/config'

describe('translate workflow preferences v2 api', () => {
  beforeEach(() => {
    getMock.mockReset()
    patchMock.mockReset()
    getMock.mockResolvedValue({
      settings: [{
        domain: 'workflow_preferences',
        payload: {
          rememberWorkflowModeEnabled: true,
          lastWorkflowMode: 'clear-all',
        },
        revision: 5,
        schemaVersion: 1,
      }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })
    patchMock.mockResolvedValue({
      domain: 'workflow_preferences',
      payload: {},
      revision: 6,
      schemaVersion: 1,
    })
  })

  it('loads workflow preferences from the settings domain', async () => {
    await expect(getTranslateWorkflowPreferences()).resolves.toEqual({
      success: true,
      preferences: {
        rememberWorkflowModeEnabled: true,
        lastWorkflowMode: 'clear-all',
      },
    })
    expect(getMock).toHaveBeenCalledWith('/api/v2/settings', {
      params: { domains: 'workflow_preferences' },
    })
  })

  it('saves workflow preferences with revision CAS', async () => {
    const preferences = {
      rememberWorkflowModeEnabled: true,
      lastWorkflowMode: 'delete-current' as const,
    }
    await saveTranslateWorkflowPreferences(preferences)

    expect(patchMock).toHaveBeenCalledWith(
      '/api/v2/settings/workflow-preferences',
      { payload: preferences, baseRevision: 5 },
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
  })
})
