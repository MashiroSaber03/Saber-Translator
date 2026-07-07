import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import {
  useBubbleStore,
  createBubbleState,
  cloneBubbleStates,
  isValidBubbleState,
} from '@/stores/bubbleStore'
import type { BubbleState, BubbleCoords, TextDirection, InpaintMethod } from '@/types/bubble'

const bubbleCoordsArbitrary = fc.tuple(
  fc.integer({ min: 0, max: 1000 }),
  fc.integer({ min: 0, max: 1000 }),
  fc.integer({ min: 0, max: 1000 }),
  fc.integer({ min: 0, max: 1000 })
).map(([x1, y1, x2, y2]): BubbleCoords => {
  const minX = Math.min(x1, x2)
  const maxX = Math.max(x1, x2) + 1
  const minY = Math.min(y1, y2)
  const maxY = Math.max(y1, y2) + 1
  return [minX, minY, maxX, maxY]
})

const textDirectionArbitrary: fc.Arbitrary<TextDirection> = fc.constantFrom(
  'vertical',
  'horizontal',
  'auto'
)

const inpaintMethodArbitrary: fc.Arbitrary<InpaintMethod> = fc.constantFrom(
  'solid',
  'lama_mpe',
  'litelama'
)

const colorArbitrary = fc.hexaString({ minLength: 6, maxLength: 6 }).map(hex => `#${hex}`)

const bubbleStateArbitrary: fc.Arbitrary<BubbleState> = fc.record({
  originalText: fc.string({ maxLength: 500 }),
  translatedText: fc.string({ maxLength: 500 }),
  textboxText: fc.string({ maxLength: 500 }),
  coords: bubbleCoordsArbitrary,
  polygon: fc.array(fc.tuple(fc.integer(), fc.integer()).map(([x, y]) => [x, y]), { maxLength: 10 }),
  fontSize: fc.integer({ min: 8, max: 72 }),
  fontFamily: fc.constantFrom('fonts/STSONG.TTF', 'fonts/msyh.ttc', 'Arial'),
  textDirection: textDirectionArbitrary,
  autoTextDirection: fc.constantFrom('vertical', 'horizontal') as fc.Arbitrary<TextDirection>,
  textColor: colorArbitrary,
  fillColor: colorArbitrary,
  rotationAngle: fc.integer({ min: -180, max: 180 }),
  position: fc.record({
    x: fc.integer({ min: -100, max: 100 }),
    y: fc.integer({ min: -100, max: 100 }),
  }),
  strokeEnabled: fc.boolean(),
  strokeColor: colorArbitrary,
  strokeWidth: fc.integer({ min: 1, max: 10 }),
  lineSpacing: fc.double({ min: 0.5, max: 3.0, noNaN: true }),
  textAlign: fc.constantFrom('start', 'center', 'end') as fc.Arbitrary<'start' | 'center' | 'end'>,
  inpaintMethod: inpaintMethodArbitrary,
  textlines: fc.constant([]),
})

const bubbleStatesArbitrary = fc.array(bubbleStateArbitrary, { maxLength: 20 })

describe('bubble store properties', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('round-trips serialized bubble state', () => {
    fc.assert(
      fc.property(bubbleStatesArbitrary, (states) => {
        const store = useBubbleStore()
        store.setBubbles(states)

        const serialized = store.serialize()
        store.clearBubbles()
        expect(store.bubbles).toHaveLength(0)

        expect(store.deserialize(serialized)).toBe(true)
        expect(store.bubbles).toHaveLength(states.length)

        for (let index = 0; index < states.length; index += 1) {
          const original = states[index]
          const restored = store.bubbles[index]
          if (!original || !restored) {
            continue
          }

          expect(restored.originalText).toBe(original.originalText)
          expect(restored.translatedText).toBe(original.translatedText)
          expect(restored.textboxText).toBe(original.textboxText)
          expect(restored.coords).toEqual(original.coords)
          expect(restored.fontSize).toBe(original.fontSize)
          expect(restored.fontFamily).toBe(original.fontFamily)
          expect(restored.textDirection).toBe(original.textDirection)
          expect(restored.textColor).toBe(original.textColor)
          expect(restored.fillColor).toBe(original.fillColor)
          expect(restored.rotationAngle).toBe(original.rotationAngle)
          expect(restored.strokeEnabled).toBe(original.strokeEnabled)
          expect(restored.strokeColor).toBe(original.strokeColor)
          expect(restored.strokeWidth).toBe(original.strokeWidth)
          expect(restored.inpaintMethod).toBe(original.inpaintMethod)
        }
      }),
      { numRuns: 100 }
    )
  })

  it('keeps multi-selection as a duplicate-free set of valid indices', () => {
    fc.assert(
      fc.property(
        fc.tuple(
          fc.array(bubbleCoordsArbitrary, { minLength: 1, maxLength: 10 }),
          fc.array(fc.integer({ min: 0, max: 9 }), { minLength: 1, maxLength: 20 })
        ),
        ([coordsList, selectOperations]) => {
          const store = useBubbleStore()
          store.setBubbles(coordsList.map(coords => createBubbleState(coords)))

          const expectedSelected = new Set<number>()
          for (const index of selectOperations) {
            if (index >= store.bubbles.length) {
              continue
            }

            store.toggleMultiSelect(index)
            if (expectedSelected.has(index)) {
              expectedSelected.delete(index)
            } else {
              expectedSelected.add(index)
            }
          }

          expect(new Set(store.selectedIndices).size).toBe(store.selectedIndices.length)
          expect(new Set(store.selectedIndices)).toEqual(expectedSelected)
          for (const index of store.selectedIndices) {
            expect(index).toBeGreaterThanOrEqual(0)
            expect(index).toBeLessThan(store.bubbles.length)
          }
        }
      ),
      { numRuns: 100 }
    )
  })

  it('clones bubble states without sharing nested references', () => {
    fc.assert(
      fc.property(bubbleStatesArbitrary, (states) => {
        const cloned = cloneBubbleStates(states)
        expect(cloned).toHaveLength(states.length)

        for (let index = 0; index < states.length; index += 1) {
          const original = states[index]
          const copy = cloned[index]
          if (!original || !copy) {
            continue
          }

          expect(copy.originalText).toBe(original.originalText)
          expect(copy.coords).toEqual(original.coords)
          expect(copy).not.toBe(original)
          expect(copy.coords).not.toBe(original.coords)
          expect(copy.position).not.toBe(original.position)
        }

        if (cloned[0] && states[0]) {
          cloned[0].originalText = 'changed clone text'
          expect(states[0].originalText).not.toBe('changed clone text')
        }
      }),
      { numRuns: 100 }
    )
  })

  it('accepts generated bubble states as valid state objects', () => {
    fc.assert(
      fc.property(bubbleStateArbitrary, (state) => {
        expect(isValidBubbleState(state)).toBe(true)
      }),
      { numRuns: 100 }
    )
  })

  it('rejects non-bubble values', () => {
    expect(isValidBubbleState(null)).toBe(false)
    expect(isValidBubbleState(undefined)).toBe(false)
    expect(isValidBubbleState('string')).toBe(false)
    expect(isValidBubbleState(123)).toBe(false)
    expect(isValidBubbleState([])).toBe(false)
    expect(isValidBubbleState({})).toBe(false)
    expect(isValidBubbleState({ coords: [0, 0, 100, 100] })).toBe(false)
    expect(isValidBubbleState({ originalText: 'test' })).toBe(false)
    expect(isValidBubbleState({
      coords: [0, 0, 100],
      originalText: 'test',
      translatedText: 'test',
    })).toBe(false)
  })

  it('keeps selection indices valid after deleting a bubble', () => {
    fc.assert(
      fc.property(
        fc.tuple(
          fc.array(bubbleCoordsArbitrary, { minLength: 3, maxLength: 10 }),
          fc.integer({ min: 0, max: 9 })
        ),
        ([coordsList, deleteIndex]) => {
          const store = useBubbleStore()
          store.setBubbles(coordsList.map(coords => createBubbleState(coords)))

          const originalLength = store.bubbles.length
          const selectIndex = Math.min(deleteIndex + 1, originalLength - 1)
          store.selectBubble(selectIndex)

          const actualDeleteIndex = Math.min(deleteIndex, originalLength - 1)
          store.deleteBubble(actualDeleteIndex)

          expect(store.bubbles).toHaveLength(originalLength - 1)
          if (store.selectedIndex >= 0) {
            expect(store.selectedIndex).toBeLessThan(store.bubbles.length)
          }
        }
      ),
      { numRuns: 100 }
    )
  })
})
