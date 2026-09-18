import { afterEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_SETTINGS, STORAGE_KEY, loadSettings, preferenceFor } from './storage'

afterEach(() => vi.unstubAllGlobals())

it('uses defaults only for settings that have not been saved', async () => {
  const get = vi.fn().mockResolvedValue({})
  vi.stubGlobal('chrome', { storage: { local: { get } } })
  expect(await loadSettings()).toEqual(DEFAULT_SETTINGS)
  get.mockResolvedValue({ [STORAGE_KEY]: {} })
  await expect(loadSettings()).rejects.toThrow('扩展设置格式无效')
})

it('does not fill in an incomplete saved domain preference', () => {
  const settings = structuredClone(DEFAULT_SETTINGS)
  Object.assign(settings.domains, { 'reader.example': { disabled: false } })
  expect(() => preferenceFor(settings, 'reader.example')).toThrow('站点设置格式无效')
})

describe('domain preferences', () => {
  it('discards old pixel positions without changing other preferences', () => {
    const settings = structuredClone(DEFAULT_SETTINGS)
    settings.domains['reader.example'] = {
      disabled: false, method: 'similar', mode: 'standard', glossaryEnabled: true,
      autoTermsEnabled: false,
      ...JSON.parse('{"fabPosition":{"x":730,"y":528}}'),
    }
    expect(preferenceFor(settings, 'reader.example')).toEqual({
      disabled: false, method: 'similar', mode: 'standard', glossaryEnabled: true,
      autoTermsEnabled: false,
    })
  })
  it('keeps defaults isolated and overlays only the current domain', () => {
    const settings = structuredClone(DEFAULT_SETTINGS)
    settings.domains['reader.example'] = {
      disabled: true,
      method: 'similar',
      mode: 'hq',
      glossaryEnabled: true,
      autoTermsEnabled: true,
      fabPosition: { side: 'left', yRatio: 0.6 },
    }

    expect(preferenceFor(settings, 'reader.example')).toMatchObject({
      disabled: true,
      method: 'similar',
      mode: 'hq',
      fabPosition: { side: 'left', yRatio: 0.6 },
    })
    expect(preferenceFor(settings, 'other.example')).toMatchObject({
      disabled: false,
      method: 'adapter',
      mode: 'standard',
    })
  })
})
