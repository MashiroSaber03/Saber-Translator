import { describe, it, expect } from 'vitest'
import * as fc from 'fast-check'
import {
  createBubbleState,
  detectTextDirection,
  isValidBubbleState,
  createBubbleStatesFromResponse,
  bubbleStatesToApiRequest,
  updateBubbleState,
  updateAllBubbleStates,
  cloneBubbleStates,
  cloneBubbleState,
  getBubbleCenter,
  getBubbleSize,
  isPointInBubble,
  isPointInPolygon,
  isPointInBubbleArea,
  getDefaultBubbleSettings,
  initBubbleStates,
} from '@/utils/bubbleFactory'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import type { BubbleCoords, BubbleState, BubbleTextline, InpaintMethod, TextAlign, TextDirection } from '@/types/bubble'

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

const textDirectionArbitrary: fc.Arbitrary<TextDirection> = fc.constantFrom('vertical', 'horizontal', 'auto')
const inpaintMethodArbitrary: fc.Arbitrary<InpaintMethod> = fc.constantFrom('solid', 'lama_mpe', 'litelama')
const textAlignArbitrary: fc.Arbitrary<TextAlign> = fc.constantFrom('start', 'center', 'end')
const colorArbitrary = fc.hexaString({ minLength: 6, maxLength: 6 }).map(hex => `#${hex}`)

const polygonPointArbitrary = fc.tuple(fc.integer({ min: -500, max: 1500 }), fc.integer({ min: -500, max: 1500 }))
  .map(([x, y]) => [x, y])

const textlineArbitrary: fc.Arbitrary<BubbleTextline> = fc.record({
  polygon: fc.array(polygonPointArbitrary, { minLength: 4, maxLength: 4 }),
  direction: fc.constantFrom('h', 'v'),
  confidence: fc.double({ min: 0, max: 1, noNaN: true }),
})

const bubbleStateArbitrary: fc.Arbitrary<BubbleState> = fc.record({
  originalText: fc.string({ maxLength: 500 }),
  translatedText: fc.string({ maxLength: 500 }),
  textboxText: fc.string({ maxLength: 500 }),
  coords: bubbleCoordsArbitrary,
  polygon: fc.array(polygonPointArbitrary, { maxLength: 10 }),
  fontSize: fc.integer({ min: 8, max: 72 }),
  fontFamily: fc.constantFrom('fonts/STSONG.TTF', 'fonts/msyh.ttc', 'Arial'),
  textDirection: textDirectionArbitrary,
  autoTextDirection: textDirectionArbitrary,
  textColor: colorArbitrary,
  fillColor: colorArbitrary,
  rotationAngle: fc.integer({ min: -180, max: 180 }),
  position: fc.record({
    x: fc.integer({ min: -100, max: 100 }),
    y: fc.integer({ min: -100, max: 100 }),
  }),
  strokeEnabled: fc.boolean(),
  strokeColor: colorArbitrary,
  strokeWidth: fc.integer({ min: 0, max: 10 }),
  lineSpacing: fc.double({ min: 0.5, max: 3, noNaN: true }),
  textAlign: textAlignArbitrary,
  inpaintMethod: inpaintMethodArbitrary,
  autoFgColor: fc.option(fc.tuple(fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 })), { nil: null }),
  autoBgColor: fc.option(fc.tuple(fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 })), { nil: null }),
  colorConfidence: fc.double({ min: 0, max: 1, noNaN: true }),
  textlines: fc.array(textlineArbitrary, { maxLength: 5 }),
  ocrResult: fc.constant(null),
})

const bubbleStatesArbitrary = fc.array(bubbleStateArbitrary, { maxLength: 20 })

describe('bubble factory properties', () => {
  describe('state creation', () => {
    it('creates the current default bubble state', () => {
      const state = createBubbleState()

      expect(state.originalText).toBe('')
      expect(state.translatedText).toBe('')
      expect(state.textboxText).toBe('')
      expect(state.coords).toEqual([0, 0, 100, 100])
      expect(state.polygon).toEqual([])
      expect(state.fontSize).toBe(TEXT_STYLE_DEFAULTS.fontSize)
      expect(state.fontFamily).toBe(TEXT_STYLE_DEFAULTS.fontFamily)
      expect(state.textDirection).toBe('vertical')
      expect(state.autoTextDirection).toBe('vertical')
      expect(state.textColor).toBe(TEXT_STYLE_DEFAULTS.textColor)
      expect(state.fillColor).toBe(TEXT_STYLE_DEFAULTS.fillColor)
      expect(state.strokeEnabled).toBe(TEXT_STYLE_DEFAULTS.strokeEnabled)
      expect(state.strokeColor).toBe(TEXT_STYLE_DEFAULTS.strokeColor)
      expect(state.strokeWidth).toBe(TEXT_STYLE_DEFAULTS.strokeWidth)
      expect(state.lineSpacing).toBe(TEXT_STYLE_DEFAULTS.lineSpacing)
      expect(state.textAlign).toBe(TEXT_STYLE_DEFAULTS.textAlign)
      expect(state.inpaintMethod).toBe(TEXT_STYLE_DEFAULTS.inpaintMethod)
    })

    it('applies overrides while preserving independent nested values', () => {
      fc.assert(
        fc.property(
          fc.record({
            originalText: fc.string(),
            translatedText: fc.string(),
            coords: bubbleCoordsArbitrary,
            polygon: fc.array(polygonPointArbitrary, { maxLength: 6 }),
            position: fc.record({ x: fc.integer(), y: fc.integer() }),
            textlines: fc.array(textlineArbitrary, { maxLength: 3 }),
            fontSize: fc.integer({ min: 8, max: 72 }),
            textColor: colorArbitrary,
            strokeEnabled: fc.boolean(),
          }),
          (overrides) => {
            const state = createBubbleState(overrides)

            expect(state.originalText).toBe(overrides.originalText)
            expect(state.translatedText).toBe(overrides.translatedText)
            expect(state.coords).toEqual(overrides.coords)
            expect(state.polygon).toEqual(overrides.polygon)
            expect(state.position).toEqual(overrides.position)
            expect(state.textlines).toEqual(overrides.textlines)
            expect(state.fontSize).toBe(overrides.fontSize)
            expect(state.textColor).toBe(overrides.textColor)
            expect(state.strokeEnabled).toBe(overrides.strokeEnabled)
            expect(state.coords).not.toBe(overrides.coords)
            expect(state.polygon).not.toBe(overrides.polygon)
            expect(state.position).not.toBe(overrides.position)
            expect(state.textlines).not.toBe(overrides.textlines)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('returns independent objects on repeated creation', () => {
      const first = createBubbleState()
      const second = createBubbleState()

      first.originalText = 'changed'
      first.coords[0] = 99
      first.position.x = 99

      expect(second.originalText).toBe('')
      expect(second.coords).toEqual([0, 0, 100, 100])
      expect(second.position).toEqual({ x: 0, y: 0 })
    })
  })

  describe('direction and validation', () => {
    it('detects text direction from normalized or reversed coords', () => {
      fc.assert(
        fc.property(bubbleCoordsArbitrary, (coords) => {
          const [x1, y1, x2, y2] = coords
          const expected = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal'

          expect(detectTextDirection(coords)).toBe(expected)
          expect(detectTextDirection([x2, y2, x1, y1])).toBe(expected)
        }),
        { numRuns: 100 }
      )
    })

    it('accepts generated and factory-created bubble states', () => {
      fc.assert(
        fc.property(bubbleStateArbitrary, fc.record({
          coords: bubbleCoordsArbitrary,
          fontSize: fc.integer({ min: 1, max: 100 }),
          textDirection: textDirectionArbitrary,
          inpaintMethod: inpaintMethodArbitrary,
        }), (state, overrides) => {
          expect(isValidBubbleState(state)).toBe(true)
          expect(isValidBubbleState(createBubbleState(overrides))).toBe(true)
        }),
        { numRuns: 100 }
      )
    })

    it('rejects malformed bubble state values', () => {
      const requiredFields = [
        'originalText',
        'translatedText',
        'textboxText',
        'coords',
        'polygon',
        'fontSize',
        'fontFamily',
        'textDirection',
        'autoTextDirection',
        'textColor',
        'fillColor',
        'rotationAngle',
        'position',
        'strokeEnabled',
        'strokeColor',
        'strokeWidth',
        'lineSpacing',
        'textAlign',
        'inpaintMethod',
        'textlines',
      ]

      expect(isValidBubbleState(null)).toBe(false)
      expect(isValidBubbleState(undefined)).toBe(false)
      expect(isValidBubbleState('string')).toBe(false)
      expect(isValidBubbleState(123)).toBe(false)
      expect(isValidBubbleState([])).toBe(false)
      expect(isValidBubbleState({})).toBe(false)

      for (const field of requiredFields) {
        const state = { ...createBubbleState() } as Record<string, unknown>
        delete state[field]
        expect(isValidBubbleState(state)).toBe(false)
      }

      for (const state of [
        { ...createBubbleState(), coords: [0, 0, 100] },
        { ...createBubbleState(), coords: [0, 0, 'bad', 100] },
        { ...createBubbleState(), fontSize: 0 },
        { ...createBubbleState(), lineSpacing: 0 },
        { ...createBubbleState(), strokeWidth: -1 },
        { ...createBubbleState(), textDirection: 'bad' },
        { ...createBubbleState(), autoTextDirection: 'bad' },
        { ...createBubbleState(), textAlign: 'bad' },
        { ...createBubbleState(), inpaintMethod: 'bad' },
        { ...createBubbleState(), textlines: [{ polygon: [[0, 0]], direction: 'h', confidence: 1 }] },
      ]) {
        expect(isValidBubbleState(state)).toBe(false)
      }
    })
  })

  describe('updates and cloning', () => {
    it('updates one state immutably', () => {
      fc.assert(
        fc.property(
          bubbleStateArbitrary,
          fc.record({
            originalText: fc.string({ minLength: 1 }).map(value => `updated-${value}`),
            fontSize: fc.integer({ min: 8, max: 72 }),
            position: fc.record({ x: fc.integer(), y: fc.integer() }),
          }),
          (state, updates) => {
            const updated = updateBubbleState(state, updates)

            expect(updated.originalText).toBe(updates.originalText)
            expect(updated.fontSize).toBe(updates.fontSize)
            expect(updated.position).toEqual({ ...state.position, ...updates.position })
            expect(updated.translatedText).toBe(state.translatedText)
            expect(updated.coords).toEqual(state.coords)
            expect(updated).not.toBe(state)
            expect(state.originalText).not.toBe(updates.originalText)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('updates every state in a collection', () => {
      fc.assert(
        fc.property(
          fc.array(bubbleStateArbitrary, { minLength: 1, maxLength: 10 }),
          fc.record({
            fontSize: fc.integer({ min: 8, max: 72 }),
            textColor: colorArbitrary,
          }),
          (states, updates) => {
            const updatedStates = updateAllBubbleStates(states, updates)

            expect(updatedStates).toHaveLength(states.length)
            for (const updated of updatedStates) {
              expect(updated.fontSize).toBe(updates.fontSize)
              expect(updated.textColor).toBe(updates.textColor)
            }
          }
        ),
        { numRuns: 100 }
      )
    })

    it('clones state objects and nested mutable fields', () => {
      fc.assert(
        fc.property(bubbleStateArbitrary, (state) => {
          const cloned = cloneBubbleState(state)

          expect(cloned).toEqual(state)
          expect(cloned).not.toBe(state)
          expect(cloned.coords).not.toBe(state.coords)
          expect(cloned.polygon).not.toBe(state.polygon)
          expect(cloned.position).not.toBe(state.position)
          expect(cloned.textlines).not.toBe(state.textlines)

          cloned.originalText = 'changed clone'
          cloned.coords[0] += 1
          cloned.position.x += 1

          expect(state.originalText).not.toBe('changed clone')
          expect(cloned.coords).not.toEqual(state.coords)
          expect(cloned.position).not.toEqual(state.position)
        }),
        { numRuns: 100 }
      )
    })

    it('clones arrays of state objects', () => {
      fc.assert(
        fc.property(bubbleStatesArbitrary, (states) => {
          const cloned = cloneBubbleStates(states)

          expect(cloned).toHaveLength(states.length)
          for (let index = 0; index < states.length; index += 1) {
            expect(cloned[index]).toEqual(states[index])
            if (states[index]) {
              expect(cloned[index]).not.toBe(states[index])
            }
          }
        }),
        { numRuns: 100 }
      )
    })
  })

  describe('geometry and hit testing', () => {
    it('computes center, size, and rectangle containment', () => {
      fc.assert(
        fc.property(bubbleCoordsArbitrary, (coords) => {
          const state = createBubbleState({ coords })
          const [x1, y1, x2, y2] = coords

          expect(getBubbleCenter(state)).toEqual({ x: (x1 + x2) / 2, y: (y1 + y2) / 2 })
          expect(getBubbleSize(state)).toEqual({ width: Math.abs(x2 - x1), height: Math.abs(y2 - y1) })
          expect(isPointInBubble(state, (x1 + x2) / 2, (y1 + y2) / 2)).toBe(true)
          expect(isPointInBubble(state, Math.max(x1, x2) + 1000, Math.max(y1, y2) + 1000)).toBe(false)
        }),
        { numRuns: 100 }
      )
    })

    it('handles polygon containment for simple and concave shapes', () => {
      expect(isPointInPolygon([], 50, 50)).toBe(false)
      expect(isPointInPolygon([[0, 0], [100, 0]], 50, 50)).toBe(false)

      const rectangle = [[0, 0], [100, 0], [100, 100], [0, 100]]
      expect(isPointInPolygon(rectangle, 50, 50)).toBe(true)
      expect(isPointInPolygon(rectangle, 150, 50)).toBe(false)

      const lShape = [[0, 0], [50, 0], [50, 50], [100, 50], [100, 100], [0, 100]]
      expect(isPointInPolygon(lShape, 25, 25)).toBe(true)
      expect(isPointInPolygon(lShape, 75, 25)).toBe(false)

      const star = [[50, 0], [60, 35], [100, 35], [70, 60], [80, 100], [50, 75], [20, 100], [30, 60], [0, 35], [40, 35]]
      expect(isPointInPolygon(star, 50, 50)).toBe(true)
      expect(isPointInPolygon(star, 75, 20)).toBe(false)
    })

    it('classifies generated rectangle polygon points', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 500 }),
          fc.integer({ min: 0, max: 500 }),
          fc.integer({ min: 10, max: 200 }),
          fc.integer({ min: 10, max: 200 }),
          fc.integer({ min: 10, max: 90 }),
          fc.integer({ min: 10, max: 90 }),
          fc.integer({ min: 10, max: 100 }),
          (x, y, width, height, relXPercent, relYPercent, offset) => {
            const polygon = [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
            const pointX = x + width * (relXPercent / 100)
            const pointY = y + height * (relYPercent / 100)

            expect(isPointInPolygon(polygon, pointX, pointY)).toBe(true)
            expect(isPointInPolygon(polygon, x - offset, y + height / 2)).toBe(false)
            expect(isPointInPolygon(polygon, x + width + offset, y + height / 2)).toBe(false)
            expect(isPointInPolygon(polygon, x + width / 2, y - offset)).toBe(false)
            expect(isPointInPolygon(polygon, x + width / 2, y + height + offset)).toBe(false)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('prefers polygon hit testing over rectangular coords', () => {
      const state = createBubbleState({
        coords: [0, 0, 100, 100],
        polygon: [[25, 25], [75, 25], [75, 75], [25, 75]],
      })

      expect(isPointInBubbleArea(state, 50, 50)).toBe(true)
      expect(isPointInBubbleArea(state, 10, 10)).toBe(false)
      expect(isPointInBubbleArea(createBubbleState({ coords: [0, 0, 100, 100], polygon: [] }), 10, 10)).toBe(true)
    })

    it('keeps generated inner polygons stricter than the rectangle', () => {
      fc.assert(
        fc.property(bubbleCoordsArbitrary, fc.integer({ min: 20, max: 40 }), (coords, scalePercent) => {
          const [x1, y1, x2, y2] = coords
          const width = x2 - x1
          const height = y2 - y1
          const centerX = (x1 + x2) / 2
          const centerY = (y1 + y2) / 2
          const scale = scalePercent / 100
          const innerWidth = width * scale
          const innerHeight = height * scale
          const polygon = [
            [centerX - innerWidth / 2, centerY - innerHeight / 2],
            [centerX + innerWidth / 2, centerY - innerHeight / 2],
            [centerX + innerWidth / 2, centerY + innerHeight / 2],
            [centerX - innerWidth / 2, centerY + innerHeight / 2],
          ]
          const state = createBubbleState({ coords, polygon })

          expect(isPointInBubbleArea(state, centerX, centerY)).toBe(true)
          expect(isPointInBubbleArea(state, x1 + width * 0.1, y1 + height * 0.1)).toBe(false)
        }),
        { numRuns: 100 }
      )
    })
  })

  describe('API projection and response hydration', () => {
    it('creates bubble states from coordinate response arrays', () => {
      fc.assert(
        fc.property(
          fc.array(bubbleCoordsArbitrary, { minLength: 1, maxLength: 5 }),
          fc.array(fc.string(), { minLength: 1, maxLength: 5 }),
          (coords, texts) => {
            const states = createBubbleStatesFromResponse({
              bubble_coords: coords,
              original_texts: texts.slice(0, coords.length),
              bubble_texts: texts.slice(0, coords.length),
              textbox_texts: texts.slice(0, coords.length),
              bubble_angles: coords.map((_, index) => index),
              auto_directions: coords.map((_, index) => index % 2 === 0 ? 'v' : 'h'),
            })

            expect(states).toHaveLength(coords.length)
            for (let index = 0; index < coords.length; index += 1) {
              expect(states[index]?.coords).toEqual(coords[index])
              expect(states[index]?.originalText).toBe(texts[index] ?? '')
              expect(states[index]?.translatedText).toBe(texts[index] ?? '')
              expect(states[index]?.textboxText).toBe(texts[index] ?? '')
              expect(states[index]?.rotationAngle).toBe(index)
              expect(states[index]?.autoTextDirection).toBe(index % 2 === 0 ? 'vertical' : 'horizontal')
            }
          }
        ),
        { numRuns: 100 }
      )
    })

    it('keeps saved response states while cloning textline data from the response shape', () => {
      fc.assert(
        fc.property(bubbleStateArbitrary, textlineArbitrary, (state, textline) => {
          const states = createBubbleStatesFromResponse({
            bubble_coords: [[1, 2, 3, 4]],
            bubble_states: [{ ...state, textlines: [] }],
            textlines_per_bubble: [[textline]],
          })

          expect(states).toHaveLength(1)
          expect(states[0]?.coords).toEqual(state.coords)
          expect(states[0]?.textlines).toEqual([textline])
          expect(states[0]?.textlines).not.toBe([textline])
        }),
        { numRuns: 100 }
      )
    })

    it('projects states to the current API request arrays', () => {
      fc.assert(
        fc.property(fc.array(bubbleStateArbitrary, { minLength: 1, maxLength: 5 }), (states) => {
          const request = bubbleStatesToApiRequest(states)

          expect(request.bubble_coords).toEqual(states.map(state => state.coords))
          expect(request.bubble_states).toBe(states)
          expect(request.original_texts).toEqual(states.map(state => state.originalText))
          expect(request.translated_texts).toEqual(states.map(state => state.translatedText))
          expect(request.textbox_texts).toEqual(states.map(state => state.textboxText))
        }),
        { numRuns: 100 }
      )
    })
  })

  describe('default settings and initialization', () => {
    it('returns current default bubble settings without global input', () => {
      expect(getDefaultBubbleSettings()).toEqual({
        fontSize: TEXT_STYLE_DEFAULTS.fontSize,
        fontFamily: TEXT_STYLE_DEFAULTS.fontFamily,
        textDirection: 'vertical',
        textColor: TEXT_STYLE_DEFAULTS.textColor,
        fillColor: TEXT_STYLE_DEFAULTS.fillColor,
        strokeEnabled: TEXT_STYLE_DEFAULTS.strokeEnabled,
        strokeColor: TEXT_STYLE_DEFAULTS.strokeColor,
        strokeWidth: TEXT_STYLE_DEFAULTS.strokeWidth,
        inpaintMethod: TEXT_STYLE_DEFAULTS.inpaintMethod,
        lineSpacing: TEXT_STYLE_DEFAULTS.lineSpacing,
        textAlign: TEXT_STYLE_DEFAULTS.textAlign,
      })
    })

    it('maps global text-style settings into bubble defaults', () => {
      fc.assert(
        fc.property(
          fc.record({
            fontSize: fc.integer({ min: 8, max: 72 }),
            fontFamily: fc.constantFrom('fonts/STSONG.TTF', 'fonts/msyh.ttc', 'Arial'),
            layoutDirection: fc.constantFrom('auto', 'vertical', 'horizontal') as fc.Arbitrary<'auto' | 'vertical' | 'horizontal'>,
            textColor: colorArbitrary,
            fillColor: colorArbitrary,
            strokeEnabled: fc.boolean(),
            strokeColor: colorArbitrary,
            strokeWidth: fc.integer({ min: 1, max: 10 }),
            lineSpacing: fc.double({ min: 0.5, max: 3, noNaN: true }),
            textAlign: textAlignArbitrary,
            inpaintMethod: inpaintMethodArbitrary,
          }),
          (globalSettings) => {
            const defaults = getDefaultBubbleSettings(globalSettings)

            expect(defaults.fontSize).toBe(globalSettings.fontSize)
            expect(defaults.fontFamily).toBe(globalSettings.fontFamily)
            expect(defaults.textDirection).toBe(globalSettings.layoutDirection)
            expect(defaults.textColor).toBe(globalSettings.textColor)
            expect(defaults.fillColor).toBe(globalSettings.fillColor)
            expect(defaults.strokeEnabled).toBe(globalSettings.strokeEnabled)
            expect(defaults.strokeColor).toBe(globalSettings.strokeColor)
            expect(defaults.strokeWidth).toBe(globalSettings.strokeWidth)
            expect(defaults.lineSpacing).toBe(globalSettings.lineSpacing)
            expect(defaults.textAlign).toBe(globalSettings.textAlign)
            expect(defaults.inpaintMethod).toBe(globalSettings.inpaintMethod)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('initializes from saved states when they match the page shape', () => {
      fc.assert(
        fc.property(fc.array(bubbleStateArbitrary, { minLength: 1, maxLength: 5 }), (savedStates) => {
          const result = initBubbleStates(savedStates, savedStates.map(state => state.coords), undefined)

          expect(result).toEqual(savedStates)
          expect(result).not.toBe(savedStates)
          for (let index = 0; index < savedStates.length; index += 1) {
            expect(result[index]).not.toBe(savedStates[index])
            expect(result[index]?.coords).not.toBe(savedStates[index]?.coords)
          }
        }),
        { numRuns: 100 }
      )
    })

    it('initializes from coords when saved state is unavailable or mismatched', () => {
      fc.assert(
        fc.property(
          fc.array(bubbleCoordsArbitrary, { minLength: 1, maxLength: 5 }),
          fc.record({
            fontSize: fc.integer({ min: 8, max: 72 }),
            textColor: colorArbitrary,
            textDirection: textDirectionArbitrary,
          }),
          (coords, globalDefaults) => {
            const result = initBubbleStates([], coords, globalDefaults)

            expect(result).toHaveLength(coords.length)
            for (let index = 0; index < coords.length; index += 1) {
              const expectedAutoDirection = detectTextDirection(coords[index] as BubbleCoords)
              expect(result[index]?.coords).toEqual(coords[index])
              expect(result[index]?.autoTextDirection).toBe(expectedAutoDirection)
              expect(result[index]?.textDirection).toBe(
                globalDefaults.textDirection === 'vertical' || globalDefaults.textDirection === 'horizontal'
                  ? globalDefaults.textDirection
                  : expectedAutoDirection
              )
              expect(result[index]?.fontSize).toBe(globalDefaults.fontSize)
              expect(result[index]?.textColor).toBe(globalDefaults.textColor)
            }
          }
        ),
        { numRuns: 100 }
      )
    })

    it('returns an empty collection without saved states or coords', () => {
      expect(initBubbleStates(undefined, undefined, undefined)).toEqual([])
      expect(initBubbleStates(undefined, [], undefined)).toEqual([])
    })
  })
})
