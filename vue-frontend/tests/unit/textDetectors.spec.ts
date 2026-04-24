import { describe, expect, it } from 'vitest'

import { TEXT_DETECTORS } from '@/constants'

describe('text detectors', () => {
  it('should only expose default, ctd, and yolo detectors', () => {
    expect(TEXT_DETECTORS.map((detector) => detector.value)).toEqual(['ctd', 'yolo', 'default'])
  })
})
