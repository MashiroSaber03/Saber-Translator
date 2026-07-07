import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import type { BubbleApiResponse, BubbleCoords } from '@/types/bubble'
import {
  createBubbleStatesFromResponse,
  detectTextDirection,
  isValidBubbleState,
} from '@/utils/bubbleFactory'

const bubbleCoordsArb = fc
  .tuple(
    fc.integer({ min: 0, max: 1000 }),
    fc.integer({ min: 0, max: 1000 }),
    fc.integer({ min: 0, max: 1000 }),
    fc.integer({ min: 0, max: 1000 }),
  )
  .map(([x1, y1, x2, y2]): BubbleCoords => {
    const minX = Math.min(x1, x2)
    const maxX = Math.max(x1, x2) + 1
    const minY = Math.min(y1, y2)
    const maxY = Math.max(y1, y2) + 1
    return [minX, minY, maxX, maxY]
  })

const bubbleResponseArb = fc.record({
  bubble_coords: fc.array(bubbleCoordsArb, { minLength: 0, maxLength: 20 }),
  bubble_angles: fc.array(fc.integer({ min: -180, max: 180 }), { minLength: 0, maxLength: 20 }),
  auto_directions: fc.array(fc.constantFrom('v', 'h'), { minLength: 0, maxLength: 20 }),
}).map((response): BubbleApiResponse => {
  const length = response.bubble_coords.length
  return {
    bubble_coords: response.bubble_coords,
    bubble_angles: response.bubble_angles.slice(0, length),
    auto_directions: response.auto_directions.slice(0, length),
  }
})

describe('bubble detection property contracts', () => {
  it('uses the shared bubble factory instead of copied detection helpers', () => {
    const source = readFileSync('tests/property/bubbleDetection.property.ts', 'utf8')
    const shadowDirectionHelper = 'detectTextDirection' + 'FromCoords'
    const shadowProgressHelper = 'calculateProgress' + 'Percent'
    const shadowStateHelper = 'createBubbleStates' + 'FromDetection'
    const shadowCoordsValidator = 'isValidBubble' + 'Coords'
    const shadowResponseValidator = 'isValidDetect' + 'Response'

    expect(source).toContain("from '@/utils/bubbleFactory'")
    for (const shadowHelper of [
      shadowDirectionHelper,
      shadowProgressHelper,
      shadowStateHelper,
      shadowCoordsValidator,
      shadowResponseValidator,
    ]) {
      expect(source).not.toContain(shadowHelper)
    }
  })

  it('keeps geometry-based direction detection in the production helper', () => {
    fc.assert(
      fc.property(bubbleCoordsArb, (coords) => {
        const [x1, y1, x2, y2] = coords
        const width = Math.abs(x2 - x1)
        const height = Math.abs(y2 - y1)

        expect(detectTextDirection(coords)).toBe(height > width ? 'vertical' : 'horizontal')
      }),
      { numRuns: 100 },
    )
  })

  it('creates one valid bubble state per detected coordinate', () => {
    fc.assert(
      fc.property(bubbleResponseArb, (response) => {
        const states = createBubbleStatesFromResponse(response)

        expect(states).toHaveLength(response.bubble_coords?.length ?? 0)
        for (const state of states) {
          expect(isValidBubbleState(state)).toBe(true)
        }
      }),
      { numRuns: 100 },
    )
  })

  it('uses backend auto direction before geometry fallback', () => {
    fc.assert(
      fc.property(bubbleCoordsArb, fc.constantFrom('v', 'h'), (coords, backendDirection) => {
        const states = createBubbleStatesFromResponse({
          bubble_coords: [coords],
          auto_directions: [backendDirection],
        })

        expect(states).toHaveLength(1)
        expect(states[0]?.autoTextDirection).toBe(
          backendDirection === 'v' ? 'vertical' : 'horizontal',
        )
      }),
      { numRuns: 100 },
    )
  })

  it('preserves detection angles, textlines, and current text fields', () => {
    fc.assert(
      fc.property(
        bubbleCoordsArb,
        fc.integer({ min: -180, max: 180 }),
        fc.string({ maxLength: 20 }),
        fc.string({ maxLength: 20 }),
        (coords, angle, originalText, translatedText) => {
          const states = createBubbleStatesFromResponse({
            bubble_coords: [coords],
            bubble_angles: [angle],
            original_texts: [originalText],
            bubble_texts: [translatedText],
            textbox_texts: [translatedText],
            textlines_per_bubble: [[{
              polygon: [[0, 0], [10, 0], [10, 10], [0, 10]],
              direction: 'h',
              confidence: 0.9,
            }]],
          })

          expect(states[0]?.rotationAngle).toBe(angle)
          expect(states[0]?.originalText).toBe(originalText)
          expect(states[0]?.translatedText).toBe(translatedText)
          expect(states[0]?.textboxText).toBe(translatedText)
          expect(states[0]?.textlines).toHaveLength(1)
        },
      ),
      { numRuns: 100 },
    )
  })

  it('applies current text-style defaults through the shared factory options', () => {
    fc.assert(
      fc.property(
        fc.array(bubbleCoordsArb, { minLength: 1, maxLength: 10 }),
        fc.integer({ min: 8, max: 72 }),
        fc.hexaString({ minLength: 6, maxLength: 6 }).map(hex => `#${hex}`),
        (coords, fontSize, textColor) => {
          const states = createBubbleStatesFromResponse(
            { bubble_coords: coords },
            {
              fontSize,
              fontFamily: 'Arial',
              textDirection: 'horizontal',
              textColor,
              fillColor: '#ffffff',
              strokeEnabled: true,
              strokeColor: '#ffffff',
              strokeWidth: 3,
              inpaintMethod: 'solid',
            },
          )

          for (const state of states) {
            expect(state.fontSize).toBe(fontSize)
            expect(state.fontFamily).toBe('Arial')
            expect(state.textDirection).toBe('horizontal')
            expect(state.textColor).toBe(textColor)
            expect(state.fillColor).toBe('#ffffff')
            expect(state.strokeEnabled).toBe(true)
            expect(state.strokeColor).toBe('#ffffff')
            expect(state.strokeWidth).toBe(3)
            expect(state.inpaintMethod).toBe('solid')
          }
        },
      ),
      { numRuns: 100 },
    )
  })

  it('returns an empty state list for an empty detection response', () => {
    expect(createBubbleStatesFromResponse({ bubble_coords: [] })).toEqual([])
    expect(createBubbleStatesFromResponse({})).toEqual([])
  })
})
