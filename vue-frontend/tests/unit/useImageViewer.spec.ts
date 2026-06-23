import { describe, expect, it } from 'vitest'

import { useImageViewer } from '@/composables/useImageViewer'

describe('useImageViewer', () => {
  it('clamps setTransform scale before later zoom math uses it', () => {
    const viewer = useImageViewer({ minScale: 0.25, maxScale: 4 })

    viewer.setTransform({ scale: 0, translateX: 12, translateY: 24 })
    expect(viewer.scale.value).toBe(0.25)

    viewer.setScale(2, 100, 100)
    const transform = viewer.getTransform()

    expect(transform.scale).toBeGreaterThanOrEqual(0.25)
    expect(transform.scale).toBeLessThanOrEqual(4)
    expect(Number.isFinite(transform.translateX)).toBe(true)
    expect(Number.isFinite(transform.translateY)).toBe(true)
  })
})
