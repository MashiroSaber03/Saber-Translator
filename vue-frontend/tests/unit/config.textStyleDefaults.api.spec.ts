import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

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
  getUserSettings,
  getTextStyleDefaults,
  resetTextStyleDefaults,
  saveTextStyleDefaults,
} from '@/api/config'
import { getTextStyleDefaults as createTextStyleDefaultsPayload } from '@/defaults/textStyleDefaults'

describe('config text style defaults api', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
  })

  it('keeps text style defaults payload fixtures typed to the current schema', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/config.textStyleDefaults.api.spec.ts'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('getTextStyleDefaults should call the text defaults route', async () => {
    getMock.mockResolvedValue({ success: true, defaults: {} })

    await getTextStyleDefaults()

    expect(getMock).toHaveBeenCalledWith('/api/config/text-style-defaults')
  })

  it('getUserSettings should call the user settings route without routine console logs', async () => {
    const response = { success: true, settings: { theme: 'light' } }
    getMock.mockResolvedValue(response)
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)

    try {
      await expect(getUserSettings()).resolves.toBe(response)
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }

    expect(getMock).toHaveBeenCalledWith('/api/get_settings')
  })

  it('saveTextStyleDefaults should post defaults payload', async () => {
    const defaults = {
      ...createTextStyleDefaultsPayload(),
      fontSize: 26,
      textColor: '#000000',
    }
    postMock.mockResolvedValue({ success: true, defaults })

    await saveTextStyleDefaults(defaults)

    expect(postMock).toHaveBeenCalledWith('/api/config/text-style-defaults', { defaults })
  })

  it('resetTextStyleDefaults should call reset route', async () => {
    postMock.mockResolvedValue({ success: true, defaults: {} })

    await resetTextStyleDefaults()

    expect(postMock).toHaveBeenCalledWith('/api/config/text-style-defaults/reset')
  })
})
