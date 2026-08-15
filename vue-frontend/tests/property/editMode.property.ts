import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import type { BubbleCoords } from '@/types/bubble'
import { createBubbleState } from '@/utils/bubbleFactory'
import { calculateDraggedCoords } from '@/utils/bubbleDrag'
import { addTestImage } from '../helpers/imageFixtures'
import {
  calculateResizedCoords as calculateResizeGeometry,
  type ResizeHandle,
} from '@/utils/bubbleResize'

const bubbleCoordsArbitrary = fc.tuple(
  fc.integer({ min: 0, max: 800 }),
  fc.integer({ min: 0, max: 800 }),
  fc.integer({ min: 50, max: 1000 }),
  fc.integer({ min: 50, max: 1000 })
).map(([x1, y1, width, height]): BubbleCoords => [x1, y1, x1 + width, y1 + height])

const dragDeltaArbitrary = fc.record({
  deltaX: fc.integer({ min: -500, max: 500 }),
  deltaY: fc.integer({ min: -500, max: 500 }),
})

const resizeHandleArbitrary = fc.constantFrom<ResizeHandle>(
  'nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'
)

const imageSizeArbitrary = fc.record({
  width: fc.integer({ min: 500, max: 2000 }),
  height: fc.integer({ min: 500, max: 2000 }),
})

function makeBubble(coords: BubbleCoords) {
  const [x1, y1, x2, y2] = coords
  const autoTextDirection = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal'
  return createBubbleState({
    coords,
    polygon: [],
    textDirection: autoTextDirection,
    autoTextDirection,
  })
}

describe('edit mode properties', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('keeps per-image bubble state when switching away and back', () => {
    fc.assert(
      fc.property(
        fc.array(bubbleCoordsArbitrary, { minLength: 1, maxLength: 5 }),
        fc.array(bubbleCoordsArbitrary, { minLength: 1, maxLength: 5 }),
        (firstCoords, secondCoords) => {
          const bubbleStore = useBubbleStore()
          const imageStore = useImageStore()
          const firstBubbles = firstCoords.map(coords => makeBubble(coords))
          const secondBubbles = secondCoords.map(coords => makeBubble(coords))

          addTestImage(imageStore, 'test1.png', '/api/v2/assets/source-1', {
            bubbleStates: firstBubbles,
          })
          addTestImage(imageStore, 'test2.png', '/api/v2/assets/source-2', {
            bubbleStates: secondBubbles,
          })

          bubbleStore.setBubbles([...firstBubbles])
          bubbleStore.updateBubble(0, { translatedText: 'changed text' })
          imageStore.updateCurrentBubbleStates([...bubbleStore.bubbles])

          imageStore.setCurrentImageIndex(1)
          bubbleStore.setBubbles([...secondBubbles])
          expect(bubbleStore.bubbles).toHaveLength(secondCoords.length)

          imageStore.setCurrentImageIndex(0)
          const savedStates = imageStore.currentImage?.bubbleStates
          if (savedStates) {
            bubbleStore.setBubbles([...savedStates])
          }

          expect(bubbleStore.bubbles).toHaveLength(firstCoords.length)
          expect(bubbleStore.bubbles[0]?.translatedText).toBe('changed text')
        }
      ),
      { numRuns: 50 }
    )
  })

  it('keeps dragged bubbles inside image bounds', () => {
    fc.assert(
      fc.property(
        bubbleCoordsArbitrary,
        dragDeltaArbitrary,
        imageSizeArbitrary,
        (coords, delta, imageSize) => {
          const bubbleStore = useBubbleStore()
          bubbleStore.setBubbles([makeBubble(coords)])

          const originalWidth = coords[2] - coords[0]
          const originalHeight = coords[3] - coords[1]
          const nextCoords = calculateDraggedCoords(
            coords,
            delta.deltaX,
            delta.deltaY,
            imageSize.width,
            imageSize.height
          )

          bubbleStore.updateBubble(0, { coords: nextCoords })

          const updatedBubble = bubbleStore.bubbles[0]
          expect(updatedBubble).toBeDefined()
          if (!updatedBubble) {
            return
          }

          const [x1, y1, x2, y2] = updatedBubble.coords
          expect(x2 - x1).toBeLessThanOrEqual(originalWidth)
          expect(y2 - y1).toBeLessThanOrEqual(originalHeight)
          expect(x1).toBeGreaterThanOrEqual(0)
          expect(y1).toBeGreaterThanOrEqual(0)
          expect(x2).toBeLessThanOrEqual(imageSize.width)
          expect(y2).toBeLessThanOrEqual(imageSize.height)
          expect(x2).toBeGreaterThan(x1)
          expect(y2).toBeGreaterThan(y1)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('keeps resized bubbles inside image bounds when resize geometry is valid', () => {
    fc.assert(
      fc.property(
        bubbleCoordsArbitrary,
        resizeHandleArbitrary,
        dragDeltaArbitrary,
        imageSizeArbitrary,
        (coords, handle, delta, imageSize) => {
          const bubbleStore = useBubbleStore()
          bubbleStore.setBubbles([makeBubble(coords)])

          const nextCoords = calculateResizeGeometry(
            coords,
            handle,
            delta.deltaX,
            delta.deltaY,
            {
              rotationAngle: 0,
              minSize: 10,
              imageWidth: imageSize.width,
              imageHeight: imageSize.height,
              clampToImage: true,
              round: true,
            }
          )

          if (!nextCoords) {
            return
          }

          bubbleStore.updateBubble(0, { coords: nextCoords })
          const updatedBubble = bubbleStore.bubbles[0]
          expect(updatedBubble).toBeDefined()
          if (!updatedBubble) {
            return
          }

          const [x1, y1, x2, y2] = updatedBubble.coords
          expect(x2 - x1).toBeGreaterThanOrEqual(10)
          expect(y2 - y1).toBeGreaterThanOrEqual(10)
          expect(x1).toBeGreaterThanOrEqual(0)
          expect(y1).toBeGreaterThanOrEqual(0)
          expect(x2).toBeLessThanOrEqual(imageSize.width)
          expect(y2).toBeLessThanOrEqual(imageSize.height)
          expect(x2).toBeGreaterThan(x1)
          expect(y2).toBeGreaterThan(y1)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('moves the expected edges for each resize handle', () => {
    const cases: Array<{
      handle: ResizeHandle
      expectX1Change: boolean
      expectY1Change: boolean
      expectX2Change: boolean
      expectY2Change: boolean
    }> = [
      { handle: 'nw', expectX1Change: true, expectY1Change: true, expectX2Change: false, expectY2Change: false },
      { handle: 'n', expectX1Change: false, expectY1Change: true, expectX2Change: false, expectY2Change: false },
      { handle: 'ne', expectX1Change: false, expectY1Change: true, expectX2Change: true, expectY2Change: false },
      { handle: 'e', expectX1Change: false, expectY1Change: false, expectX2Change: true, expectY2Change: false },
      { handle: 'se', expectX1Change: false, expectY1Change: false, expectX2Change: true, expectY2Change: true },
      { handle: 's', expectX1Change: false, expectY1Change: false, expectX2Change: false, expectY2Change: true },
      { handle: 'sw', expectX1Change: true, expectY1Change: false, expectX2Change: false, expectY2Change: true },
      { handle: 'w', expectX1Change: true, expectY1Change: false, expectX2Change: false, expectY2Change: false },
    ]

    for (const resizeCase of cases) {
      const coords: BubbleCoords = [100, 100, 300, 300]
      const delta = 50
      const nextCoords = calculateResizeGeometry(
        coords,
        resizeCase.handle,
        resizeCase.handle.includes('w') ? -delta : (resizeCase.handle.includes('e') ? delta : 0),
        resizeCase.handle.includes('n') ? -delta : (resizeCase.handle.includes('s') ? delta : 0),
        {
          rotationAngle: 0,
          minSize: 10,
          imageWidth: 1000,
          imageHeight: 1000,
          clampToImage: true,
          round: true,
        }
      )

      expect(nextCoords).not.toBeNull()
      if (!nextCoords) {
        continue
      }

      const [x1, y1, x2, y2] = nextCoords
      if (resizeCase.expectX1Change) expect(x1).not.toBe(coords[0])
      if (resizeCase.expectY1Change) expect(y1).not.toBe(coords[1])
      if (resizeCase.expectX2Change) expect(x2).not.toBe(coords[2])
      if (resizeCase.expectY2Change) expect(y2).not.toBe(coords[3])
    }
  })

  it('clamps drag movement at image edges', () => {
    const coords: BubbleCoords = [100, 100, 200, 200]

    expect(calculateDraggedCoords(coords, -200, -200, 500, 500).slice(0, 2)).toEqual([0, 0])

    const bottomRight = calculateDraggedCoords(coords, 500, 500, 500, 500)
    expect(bottomRight[2]).toBeLessThanOrEqual(500)
    expect(bottomRight[3]).toBeLessThanOrEqual(500)
  })

  it('rejects resize results below the minimum size', () => {
    const coords: BubbleCoords = [100, 100, 150, 150]

    expect(calculateResizeGeometry(coords, 'e', -45, 0, {
      rotationAngle: 0,
      minSize: 10,
      imageWidth: 1000,
      imageHeight: 1000,
      clampToImage: true,
      round: true,
    })).toBeNull()
  })

  it('does not derive backend-owned automatic direction after coords update', () => {
    const bubbleStore = useBubbleStore()
    bubbleStore.setBubbles([makeBubble([0, 0, 200, 100])])

    expect(bubbleStore.bubbles[0]?.autoTextDirection).toBe('horizontal')

    bubbleStore.updateBubble(0, { coords: [0, 0, 100, 200] })

    expect(bubbleStore.bubbles[0]?.autoTextDirection).toBe('horizontal')
  })
})
