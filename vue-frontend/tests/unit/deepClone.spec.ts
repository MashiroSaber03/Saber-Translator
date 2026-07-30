import { describe, expect, it } from 'vitest'

import { deepClone } from '@/utils/deepClone'

describe('deepClone', () => {
  it('preserves undefined values used by optional settings fields', () => {
    const source = {
      request: {
        forceJsonOutput: false,
        extraBody: undefined,
      },
      rounds: [
        {
          name: '校对',
          optional: undefined,
        },
      ],
    }

    const cloned = deepClone(source)

    expect(cloned).toEqual(source)
    expect(Object.prototype.hasOwnProperty.call(cloned.request, 'extraBody')).toBe(true)
    expect(cloned).not.toBe(source)
    expect(cloned.request).not.toBe(source.request)
    expect(cloned.rounds).not.toBe(source.rounds)
  })

  it('accepts undefined as the root value', () => {
    expect(deepClone(undefined)).toBeUndefined()
  })
})
