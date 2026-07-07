import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import { useImageViewer } from '@/composables/useImageViewer'
import type { ImageViewerOptions, TransformState } from '@/composables/useImageViewer'

const scaleArb = fc.double({ min: 0.1, max: 5, noNaN: true })
const translateArb = fc.double({ min: -1000, max: 1000, noNaN: true })
const dimensionArb = fc.integer({ min: 100, max: 2000 })
const zoomFactorArb = fc.double({ min: 0.5, max: 2, noNaN: true })

const transformArb: fc.Arbitrary<TransformState> = fc.record({
  scale: scaleArb,
  translateX: translateArb,
  translateY: translateArb,
})

const viewerOptionsArb: fc.Arbitrary<Required<Pick<ImageViewerOptions, 'minScale' | 'maxScale' | 'zoomSpeed'>>> = fc.record({
  minScale: fc.double({ min: 0.01, max: 0.5, noNaN: true }),
  maxScale: fc.double({ min: 2, max: 10, noNaN: true }),
  zoomSpeed: fc.double({ min: 0.05, max: 0.3, noNaN: true }),
})

function createViewer(options: ImageViewerOptions, transform?: TransformState) {
  const viewer = useImageViewer(options)
  if (transform) {
    viewer.setTransform(transform)
  }
  return viewer
}

describe('image viewer properties', () => {
  it('keeps zoomed scale within configured bounds', () => {
    fc.assert(
      fc.property(transformArb, viewerOptionsArb, zoomFactorArb, (transform, options, factor) => {
        const viewer = createViewer(options, transform)

        viewer.zoom(factor, 800, 600)

        expect(viewer.scale.value).toBeGreaterThanOrEqual(options.minScale)
        expect(viewer.scale.value).toBeLessThanOrEqual(options.maxScale)
      }),
      { numRuns: 100 }
    )
  })

  it('keeps repeated zoom-in and zoom-out operations within configured bounds', () => {
    fc.assert(
      fc.property(viewerOptionsArb, fc.integer({ min: 1, max: 50 }), (options, iterations) => {
        const zoomedIn = createViewer(options)
        const zoomedOut = createViewer(options)

        for (let i = 0; i < iterations; i++) {
          zoomedIn.zoom(1.5, 800, 600)
          zoomedOut.zoom(0.5, 800, 600)
        }

        expect(zoomedIn.scale.value).toBeLessThanOrEqual(options.maxScale)
        expect(zoomedOut.scale.value).toBeGreaterThanOrEqual(options.minScale)
      }),
      { numRuns: 100 }
    )
  })

  it('applies pan deltas without changing scale', () => {
    fc.assert(
      fc.property(transformArb, translateArb, translateArb, (transform, deltaX, deltaY) => {
        const viewer = createViewer({}, transform)
        const initialScale = viewer.scale.value

        viewer.pan(deltaX, deltaY)

        expect(viewer.translateX.value).toBeCloseTo(transform.translateX + deltaX, 5)
        expect(viewer.translateY.value).toBeCloseTo(transform.translateY + deltaY, 5)
        expect(viewer.scale.value).toBe(initialScale)
      }),
      { numRuns: 100 }
    )
  })

  it('resets transform state to the default viewport origin', () => {
    fc.assert(
      fc.property(transformArb, (transform) => {
        const viewer = createViewer({}, transform)

        viewer.reset()

        expect(viewer.getTransform()).toEqual({
          scale: 1,
          translateX: 0,
          translateY: 0,
        })
      }),
      { numRuns: 100 }
    )
  })

  it('fits positive image dimensions inside the viewport with a margin', () => {
    fc.assert(
      fc.property(dimensionArb, dimensionArb, dimensionArb, dimensionArb, (viewportWidth, viewportHeight, imageWidth, imageHeight) => {
        const viewer = useImageViewer()

        viewer.fitToScreen(imageWidth, imageHeight, viewportWidth, viewportHeight)

        const scaledWidth = imageWidth * viewer.scale.value
        const scaledHeight = imageHeight * viewer.scale.value
        expect(viewer.scale.value).toBeGreaterThan(0)
        expect(scaledWidth).toBeLessThanOrEqual(viewportWidth)
        expect(scaledHeight).toBeLessThanOrEqual(viewportHeight)
      }),
      { numRuns: 100 }
    )
  })

  it('keeps the zoom focal point stable when zooming at a point', () => {
    fc.assert(
      fc.property(
        viewerOptionsArb,
        fc.double({ min: 100, max: 700, noNaN: true }),
        fc.double({ min: 100, max: 500, noNaN: true }),
        fc.double({ min: 1.1, max: 1.5, noNaN: true }),
        (options, x, y, factor) => {
          const viewer = useImageViewer(options)

          viewer.zoomAt(x, y, factor)

          const transform = viewer.getTransform()
          expect((x - transform.translateX) / transform.scale).toBeCloseTo(x, 5)
          expect((y - transform.translateY) / transform.scale).toBeCloseTo(y, 5)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('keeps zoom factor 1 stable for an in-range transform', () => {
    fc.assert(
      fc.property(viewerOptionsArb, (options) => {
        const scale = (options.minScale + options.maxScale) / 2
        const viewer = createViewer(options, { scale, translateX: 100, translateY: 100 })

        viewer.zoom(1, 800, 600)

        expect(viewer.scale.value).toBeCloseTo(scale, 8)
        expect(viewer.translateX.value).toBeCloseTo(100, 8)
        expect(viewer.translateY.value).toBeCloseTo(100, 8)
      }),
      { numRuns: 100 }
    )
  })

  it('keeps pan reversible', () => {
    fc.assert(
      fc.property(transformArb, translateArb, translateArb, (transform, deltaX, deltaY) => {
        const viewer = createViewer({}, transform)
        const initial = viewer.getTransform()

        viewer.pan(deltaX, deltaY)
        viewer.pan(-deltaX, -deltaY)

        expect(viewer.translateX.value).toBeCloseTo(initial.translateX, 5)
        expect(viewer.translateY.value).toBeCloseTo(initial.translateY, 5)
        expect(viewer.scale.value).toBe(initial.scale)
      }),
      { numRuns: 100 }
    )
  })

  it('leaves transform unchanged for non-positive fit dimensions', () => {
    const viewer = createViewer({}, { scale: 1.4, translateX: 20, translateY: 30 })

    viewer.fitToScreen(0, 100, 800, 600)
    viewer.fitToScreen(100, -100, 800, 600)

    expect(viewer.getTransform()).toEqual({
      scale: 1.4,
      translateX: 20,
      translateY: 30,
    })
  })

  it('clamps extreme zoom factors', () => {
    const options = { minScale: 0.1, maxScale: 5, zoomSpeed: 0.1 }
    const zoomedIn = useImageViewer(options)
    const zoomedOut = useImageViewer(options)

    zoomedIn.zoom(1000, 800, 600)
    zoomedOut.zoom(0.001, 800, 600)

    expect(zoomedIn.scale.value).toBeLessThanOrEqual(options.maxScale)
    expect(zoomedOut.scale.value).toBeGreaterThanOrEqual(options.minScale)
  })
})
