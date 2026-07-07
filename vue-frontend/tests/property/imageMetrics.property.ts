import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import {
  imageToScreenCoords,
  screenToImageCoords,
  bubbleCoordsToScreen,
  screenCoordsToBubble,
  polygonToScreen,
  screenPolygonToImage,
  scaleSize,
  isPointInVisualContent,
  type ImageDisplayMetrics,
} from '@/utils/imageMetrics'

type BubbleCoords = [number, number, number, number]
type PolygonPoint = [number, number]

const displayMetricsArbitrary: fc.Arbitrary<ImageDisplayMetrics> = fc.record({
  visualContentWidth: fc.integer({ min: 100, max: 2000 }),
  visualContentHeight: fc.integer({ min: 100, max: 2000 }),
  visualContentOffsetX: fc.integer({ min: 0, max: 500 }),
  visualContentOffsetY: fc.integer({ min: 0, max: 500 }),
  naturalWidth: fc.integer({ min: 100, max: 4000 }),
  naturalHeight: fc.integer({ min: 100, max: 4000 }),
  elementWidth: fc.integer({ min: 100, max: 2000 }),
  elementHeight: fc.integer({ min: 100, max: 2000 }),
}).map(metrics => ({
  ...metrics,
  scaleX: metrics.visualContentWidth / metrics.naturalWidth,
  scaleY: metrics.visualContentHeight / metrics.naturalHeight,
}))

const imagePointArbitrary = displayMetricsArbitrary.chain(metrics =>
  fc.record({
    metrics: fc.constant(metrics),
    imageX: fc.integer({ min: 0, max: metrics.naturalWidth }),
    imageY: fc.integer({ min: 0, max: metrics.naturalHeight }),
  }),
)

const screenPointArbitrary = displayMetricsArbitrary.chain(metrics =>
  fc.record({
    metrics: fc.constant(metrics),
    screenX: fc.integer({
      min: metrics.visualContentOffsetX,
      max: metrics.visualContentOffsetX + metrics.visualContentWidth,
    }),
    screenY: fc.integer({
      min: metrics.visualContentOffsetY,
      max: metrics.visualContentOffsetY + metrics.visualContentHeight,
    }),
  }),
)

function bubbleCoordsArbitrary(metrics: ImageDisplayMetrics): fc.Arbitrary<BubbleCoords> {
  return fc
    .tuple(
      fc.integer({ min: 0, max: metrics.naturalWidth - 1 }),
      fc.integer({ min: 0, max: metrics.naturalHeight - 1 }),
      fc.integer({ min: 1, max: metrics.naturalWidth }),
      fc.integer({ min: 1, max: metrics.naturalHeight }),
    )
    .filter(([x1, y1, x2, y2]) => x1 < x2 && y1 < y2) as fc.Arbitrary<BubbleCoords>
}

const metricAndBubbleArbitrary = displayMetricsArbitrary.chain(metrics =>
  bubbleCoordsArbitrary(metrics).map(coords => ({ metrics, coords })),
)

function polygonArbitrary(metrics: ImageDisplayMetrics): fc.Arbitrary<PolygonPoint[]> {
  return fc.array(
    fc
      .tuple(
        fc.integer({ min: 0, max: metrics.naturalWidth }),
        fc.integer({ min: 0, max: metrics.naturalHeight }),
      )
      .map(([x, y]) => [x, y] as PolygonPoint),
    { minLength: 3, maxLength: 8 },
  )
}

const metricAndPolygonArbitrary = displayMetricsArbitrary.chain(metrics =>
  polygonArbitrary(metrics).map(polygon => ({ metrics, polygon })),
)

function expectRectCloseTo(actual: BubbleCoords, expected: BubbleCoords, tolerance: number): void {
  for (const [index, expectedValue] of expected.entries()) {
    expect(Math.abs(actual[index] - expectedValue)).toBeLessThanOrEqual(tolerance)
  }
}

function expectPolygonCloseTo(actual: number[][], expected: PolygonPoint[], tolerance: number): void {
  expect(actual).toHaveLength(expected.length)

  for (const [index, expectedPoint] of expected.entries()) {
    const actualPoint = actual[index]
    const actualX = actualPoint?.[0] ?? Number.NaN
    const actualY = actualPoint?.[1] ?? Number.NaN

    expect(Math.abs(actualX - expectedPoint[0])).toBeLessThanOrEqual(tolerance)
    expect(Math.abs(actualY - expectedPoint[1])).toBeLessThanOrEqual(tolerance)
  }
}

describe('image metrics properties', () => {
  it('round-trips image points through screen coordinates', () => {
    fc.assert(
      fc.property(imagePointArbitrary, ({ metrics, imageX, imageY }) => {
        const screenCoords = imageToScreenCoords(imageX, imageY, metrics)
        const imageCoords = screenToImageCoords(screenCoords.x, screenCoords.y, metrics)

        expect(imageCoords.x).toBeCloseTo(imageX, 5)
        expect(imageCoords.y).toBeCloseTo(imageY, 5)
      }),
    )
  })

  it('round-trips screen points through image coordinates', () => {
    fc.assert(
      fc.property(screenPointArbitrary, ({ metrics, screenX, screenY }) => {
        const imageCoords = screenToImageCoords(screenX, screenY, metrics)
        const screenCoords = imageToScreenCoords(imageCoords.x, imageCoords.y, metrics)

        expect(screenCoords.x).toBeCloseTo(screenX, 5)
        expect(screenCoords.y).toBeCloseTo(screenY, 5)
      }),
    )
  })

  it('projects points and sizes with the same scale and offset contract', () => {
    fc.assert(
      fc.property(
        imagePointArbitrary,
        fc.integer({ min: 1, max: 500 }),
        fc.integer({ min: 1, max: 500 }),
        ({ metrics, imageX, imageY }, width, height) => {
          const screenCoords = imageToScreenCoords(imageX, imageY, metrics)
          const scaledSize = scaleSize(width, height, metrics)

          expect(screenCoords.x).toBeCloseTo(imageX * metrics.scaleX + metrics.visualContentOffsetX, 10)
          expect(screenCoords.y).toBeCloseTo(imageY * metrics.scaleY + metrics.visualContentOffsetY, 10)
          expect(scaledSize.width).toBeCloseTo(width * metrics.scaleX, 10)
          expect(scaledSize.height).toBeCloseTo(height * metrics.scaleY, 10)
        },
      ),
    )
  })

  it('keeps bubble rectangles ordered and linearly scaled', () => {
    fc.assert(
      fc.property(metricAndBubbleArbitrary, ({ metrics, coords }) => {
        const screenCoords = bubbleCoordsToScreen(coords, metrics)
        const originalWidth = coords[2] - coords[0]
        const originalHeight = coords[3] - coords[1]
        const screenWidth = screenCoords[2] - screenCoords[0]
        const screenHeight = screenCoords[3] - screenCoords[1]

        expect(screenCoords[0]).toBeLessThan(screenCoords[2])
        expect(screenCoords[1]).toBeLessThan(screenCoords[3])
        expect(screenWidth).toBeCloseTo(originalWidth * metrics.scaleX, 5)
        expect(screenHeight).toBeCloseTo(originalHeight * metrics.scaleY, 5)
      }),
    )
  })

  it('round-trips bubble rectangles within the integer rounding boundary', () => {
    fc.assert(
      fc.property(metricAndBubbleArbitrary, ({ metrics, coords }) => {
        const screenCoords = bubbleCoordsToScreen(coords, metrics)
        const imageCoords = screenCoordsToBubble(screenCoords, metrics)

        expectRectCloseTo(imageCoords, coords, 1)
      }),
    )
  })

  it('keeps polygon vertex counts and coordinates stable across projection', () => {
    fc.assert(
      fc.property(metricAndPolygonArbitrary, ({ metrics, polygon }) => {
        const screenPolygon = polygonToScreen(polygon, metrics)
        const imagePolygon = screenPolygonToImage(screenPolygon, metrics)

        expect(screenPolygon).toHaveLength(polygon.length)
        expectPolygonCloseTo(imagePolygon, polygon, 1)
      }),
    )
  })

  it('classifies points inside and outside the rendered image content', () => {
    fc.assert(
      fc.property(
        displayMetricsArbitrary,
        fc.integer({ min: 0, max: 100 }),
        fc.integer({ min: 0, max: 100 }),
        fc.integer({ min: 1, max: 100 }),
        (metrics, ratioXInt, ratioYInt, offset) => {
          const x = metrics.visualContentOffsetX + (ratioXInt / 100) * metrics.visualContentWidth
          const y = metrics.visualContentOffsetY + (ratioYInt / 100) * metrics.visualContentHeight

          expect(isPointInVisualContent(x, y, metrics)).toBe(true)
          expect(isPointInVisualContent(metrics.visualContentOffsetX - offset, y, metrics)).toBe(false)
          expect(isPointInVisualContent(x, metrics.visualContentOffsetY - offset, metrics)).toBe(false)
          expect(isPointInVisualContent(metrics.visualContentOffsetX + metrics.visualContentWidth + offset, y, metrics)).toBe(false)
          expect(isPointInVisualContent(x, metrics.visualContentOffsetY + metrics.visualContentHeight + offset, metrics)).toBe(false)
        },
      ),
    )
  })

  it('keeps the zero-scale and origin coordinate boundaries explicit', () => {
    fc.assert(
      fc.property(displayMetricsArbitrary, metrics => {
        const zeroScaleMetrics: ImageDisplayMetrics = {
          ...metrics,
          visualContentWidth: 0,
          visualContentHeight: 0,
          scaleX: 0,
          scaleY: 0,
        }
        const zeroScaleCoords = screenToImageCoords(
          metrics.visualContentOffsetX + 100,
          metrics.visualContentOffsetY + 100,
          zeroScaleMetrics,
        )
        const origin = imageToScreenCoords(0, 0, metrics)

        expect(zeroScaleCoords).toEqual({ x: 0, y: 0 })
        expect(origin.x).toBeCloseTo(metrics.visualContentOffsetX, 10)
        expect(origin.y).toBeCloseTo(metrics.visualContentOffsetY, 10)
      }),
    )
  })
})
