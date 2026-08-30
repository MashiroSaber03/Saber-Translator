import { describe, expect, it } from 'vitest'
import { DEFAULT_SETTINGS, preferenceFor } from './storage'

describe('domain preferences', () => {
  it('keeps defaults isolated and overlays only the current domain', () => {
    const settings = structuredClone(DEFAULT_SETTINGS)
    settings.domains['reader.example'] = {
      disabled: true,
      method: 'similar',
      mode: 'hq',
      glossaryEnabled: true,
      autoTermsEnabled: true,
      panelOpen: true,
      panelPosition: { x: 120, y: 80 },
      fabPosition: { x: 24, y: 360 },
    }

    expect(preferenceFor(settings, 'reader.example')).toMatchObject({
      disabled: true,
      method: 'similar',
      mode: 'hq',
      panelOpen: true,
      panelPosition: { x: 120, y: 80 },
      fabPosition: { x: 24, y: 360 },
    })
    expect(preferenceFor(settings, 'other.example')).toMatchObject({
      disabled: false,
      method: 'adapter',
      mode: 'standard',
      panelOpen: false,
    })
  })
})
