import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const { getMock, putMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  putMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    put: putMock,
  },
}))

import {
  getTextStyleDefaults,
  resetTextStyleDefaults,
  saveTextStyleDefaults,
} from '@/api/config'
import { getTextStyleDefaults as createTextStyleDefaultsPayload } from '@/defaults/textStyleDefaults'

describe('config text style defaults v2 api', () => {
  beforeEach(() => {
    getMock.mockReset()
    putMock.mockReset()
    getMock.mockResolvedValue({
      settings: [{
        domain: 'text_style_defaults',
        payload: createTextStyleDefaultsPayload(),
        revision: 7,
        schemaVersion: 1,
      }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })
    putMock.mockResolvedValue({
      settings: [{ domain: 'text_style_defaults', revision: 8 }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })
  })

  it('keeps text style defaults payload fixtures typed to the current schema', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/config.textStyleDefaults.api.spec.ts'), 'utf8')
    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('loads the dedicated backend settings domain', async () => {
    await expect(getTextStyleDefaults()).resolves.toMatchObject({
      success: true,
      defaults: createTextStyleDefaultsPayload(),
    })
    expect(getMock).toHaveBeenCalledWith('/api/v2/settings', {
      params: { domains: 'text_style_defaults' },
    })
  })

  it('saves and resets through one revision-checked settings transaction', async () => {
    const defaults = {
      ...createTextStyleDefaultsPayload(),
      fontSize: 26,
      textColor: '#000000',
    }
    await saveTextStyleDefaults(defaults)
    await resetTextStyleDefaults()

    expect(putMock).toHaveBeenCalledTimes(2)
    expect(putMock.mock.calls[0]![0]).toBe('/api/v2/settings/transactions')
    expect(putMock.mock.calls[0]![1]).toMatchObject({
      settings: [{
        domain: 'text_style_defaults',
        payload: defaults,
        baseRevision: 7,
      }],
    })
    expect(putMock.mock.calls[1]![1]).toMatchObject({
      settings: [{
        domain: 'text_style_defaults',
        baseRevision: 8,
      }],
    })
  })
})
