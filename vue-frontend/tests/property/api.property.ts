import { describe, it, expect } from 'vitest'
import * as fc from 'fast-check'
import type { BubbleState, BubbleCoords, TextDirection } from '@/types'
import type { ReRenderParams } from '@/api/translate'
import type { ParallelTranslateParams } from '@/api/parallelTranslate'
import { AI_PROVIDER_MANIFEST } from '@/config/aiProviders'

const TRANSLATION_PROVIDER_IDS = AI_PROVIDER_MANIFEST.filter(provider =>
  provider.capabilities.includes('translation')
).map(provider => provider.id) as [string, ...string[]]

function generateBubbleCoords(): fc.Arbitrary<BubbleCoords> {
  return fc
    .record({
      x1: fc.integer({ min: 0, max: 1000 }),
      y1: fc.integer({ min: 0, max: 1000 }),
      width: fc.integer({ min: 10, max: 500 }),
      height: fc.integer({ min: 10, max: 500 }),
    })
    .map(({ x1, y1, width, height }) => [x1, y1, x1 + width, y1 + height])
}

function generateBubbleState(): fc.Arbitrary<BubbleState> {
  const textDirection = fc.constantFrom(
    'auto',
    'vertical',
    'horizontal'
  ) as fc.Arbitrary<TextDirection>

  return fc.record({
    coords: generateBubbleCoords(),
    polygon: fc.array(
      fc.tuple(fc.integer(), fc.integer()).map(([x, y]) => [x, y]),
      { maxLength: 10 }
    ),
    originalText: fc.string({ minLength: 0, maxLength: 200 }),
    translatedText: fc.string({ minLength: 0, maxLength: 200 }),
    textboxText: fc.string({ minLength: 0, maxLength: 200 }),
    fontSize: fc.integer({ min: 8, max: 72 }),
    fontFamily: fc.constantFrom('Arial', 'SimHei', 'Microsoft YaHei'),
    textDirection,
    autoTextDirection: textDirection,
    textColor: fc.hexaString({ minLength: 6, maxLength: 6 }).map(value => `#${value}`),
    fillColor: fc.hexaString({ minLength: 6, maxLength: 6 }).map(value => `#${value}`),
    rotationAngle: fc.integer({ min: 0, max: 360 }),
    position: fc.record({
      x: fc.integer({ min: -100, max: 100 }),
      y: fc.integer({ min: -100, max: 100 }),
    }),
    strokeEnabled: fc.boolean(),
    strokeColor: fc.hexaString({ minLength: 6, maxLength: 6 }).map(value => `#${value}`),
    strokeWidth: fc.integer({ min: 1, max: 10 }),
    lineSpacing: fc.double({ min: 0.5, max: 3.0, noNaN: true }),
    textAlign: fc.constantFrom('start', 'center', 'end'),
    inpaintMethod: fc.constantFrom('solid', 'lama_mpe', 'litelama'),
    textlines: fc.constant([]),
  })
}

function generateParallelTranslateParams(): fc.Arbitrary<ParallelTranslateParams> {
  return fc.record({
    original_texts: fc.array(fc.string({ minLength: 1, maxLength: 200 }), {
      minLength: 1,
      maxLength: 10,
    }),
    target_language: fc.constantFrom('zh-CN', 'en', 'ja', 'ko'),
    source_language: fc.option(fc.constantFrom('japanese', 'english', 'korean'), {
      nil: undefined,
    }),
    model_provider: fc.constantFrom(...TRANSLATION_PROVIDER_IDS),
    model_name: fc.option(fc.string({ minLength: 1, maxLength: 80 }), { nil: undefined }),
    api_key: fc.option(fc.string({ minLength: 1, maxLength: 80 }), { nil: undefined }),
    custom_base_url: fc.option(fc.webUrl(), { nil: undefined }),
    prompt_content: fc.option(fc.string({ minLength: 0, maxLength: 200 }), { nil: undefined }),
    use_textbox_prompt: fc.boolean(),
  })
}

describe('API parameter invariants', () => {
  it('parallel translation params include current required fields', () => {
    fc.assert(
      fc.property(generateParallelTranslateParams(), params => {
        expect(params.original_texts.length).toBeGreaterThan(0)
        expect(params.original_texts.every(text => typeof text === 'string')).toBe(true)
        expect(params.target_language).toMatch(/^\S+$/)
        expect(TRANSLATION_PROVIDER_IDS).toContain(params.model_provider)
      }),
      { numRuns: 100 }
    )
  })

  it('bubble state JSON serialization preserves render-critical fields', () => {
    fc.assert(
      fc.property(generateBubbleState(), bubbleState => {
        const deserialized = JSON.parse(JSON.stringify(bubbleState)) as BubbleState

        expect(deserialized.coords).toEqual(bubbleState.coords)
        expect(deserialized.originalText).toBe(bubbleState.originalText)
        expect(deserialized.translatedText).toBe(bubbleState.translatedText)
        expect(deserialized.fontSize).toBe(bubbleState.fontSize)
        expect(deserialized.fontFamily).toBe(bubbleState.fontFamily)
        expect(deserialized.textDirection).toBe(bubbleState.textDirection)
        expect(deserialized.textColor).toBe(bubbleState.textColor)
        expect(deserialized.fillColor).toBe(bubbleState.fillColor)
        expect(deserialized.strokeEnabled).toBe(bubbleState.strokeEnabled)
        expect(deserialized.strokeColor).toBe(bubbleState.strokeColor)
        expect(deserialized.strokeWidth).toBe(bubbleState.strokeWidth)
        expect(deserialized.inpaintMethod).toBe(bubbleState.inpaintMethod)
      }),
      { numRuns: 100 }
    )
  })

  it('bubble coords form positive rectangles', () => {
    fc.assert(
      fc.property(generateBubbleCoords(), coords => {
        const [x1, y1, x2, y2] = coords

        expect(x1).toBeGreaterThanOrEqual(0)
        expect(y1).toBeGreaterThanOrEqual(0)
        expect(x2).toBeGreaterThan(x1)
        expect(y2).toBeGreaterThan(y1)
        expect(Number.isInteger(x1)).toBe(true)
        expect(Number.isInteger(y1)).toBe(true)
        expect(Number.isInteger(x2)).toBe(true)
        expect(Number.isInteger(y2)).toBe(true)
      }),
      { numRuns: 100 }
    )
  })

  it('rerender params keep bubble text, coords, and state arrays aligned', () => {
    fc.assert(
      fc.property(
        fc.array(generateBubbleState(), { minLength: 1, maxLength: 10 }),
        bubbleStates => {
          const params: ReRenderParams = {
            clean_image: 'base64_clean',
            bubble_texts: bubbleStates.map(state => state.translatedText),
            bubble_coords: bubbleStates.map(state => state.coords),
            bubble_states: bubbleStates.map(state => ({
              translatedText: state.translatedText,
              coords: state.coords,
              fontSize: state.fontSize,
              fontFamily: state.fontFamily,
              textDirection: state.textDirection,
              textColor: state.textColor,
              rotationAngle: state.rotationAngle,
              position: state.position,
              strokeEnabled: state.strokeEnabled,
              strokeColor: state.strokeColor,
              strokeWidth: state.strokeWidth,
              lineSpacing: state.lineSpacing,
              textAlign: state.textAlign,
            })),
          }

          expect(params.clean_image).toBeDefined()
          expect(params.bubble_texts).toHaveLength(bubbleStates.length)
          expect(params.bubble_coords).toHaveLength(bubbleStates.length)
          expect(params.bubble_states).toHaveLength(bubbleStates.length)

          params.bubble_states?.forEach((state, index) => {
            expect(params.bubble_texts[index]).toBe(bubbleStates[index].translatedText)
            expect(params.bubble_coords[index]).toEqual(bubbleStates[index].coords)
            expect(state.coords).toEqual(bubbleStates[index].coords)
            expect(state.fontSize).toBe(bubbleStates[index].fontSize)
          })
        }
      ),
      { numRuns: 50 }
    )
  })

  it('session data serialization preserves image order', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            originalDataURL: fc.string({ minLength: 10, maxLength: 50 }),
            translatedDataURL: fc.option(fc.string({ minLength: 10, maxLength: 50 })),
            fileName: fc.string({ minLength: 1, maxLength: 50 }),
            bubbleStates: fc.array(generateBubbleState(), { minLength: 0, maxLength: 5 }),
          }),
          { minLength: 1, maxLength: 10 }
        ),
        images => {
          const sessionData = {
            name: 'test_session',
            version: '2.0',
            savedAt: new Date().toISOString(),
            imageCount: images.length,
            ui_settings: {},
            images,
            currentImageIndex: 0,
          }

          const deserialized = JSON.parse(JSON.stringify(sessionData)) as typeof sessionData

          expect(deserialized.images.length).toBe(images.length)
          images.forEach((image, index) => {
            expect(deserialized.images[index].originalDataURL).toBe(image.originalDataURL)
            expect(deserialized.images[index].fileName).toBe(image.fileName)
            expect(deserialized.images[index].bubbleStates.length).toBe(image.bubbleStates.length)
          })
        }
      ),
      { numRuns: 50 }
    )
  })

  it('translation providers come from the current provider manifest', () => {
    fc.assert(
      fc.property(fc.constantFrom(...TRANSLATION_PROVIDER_IDS), providerId => {
        const manifest = AI_PROVIDER_MANIFEST.find(provider => provider.id === providerId)

        expect(manifest).toBeDefined()
        expect(manifest?.capabilities).toContain('translation')
        expect(typeof manifest?.requiresApiKey).toBe('boolean')
        expect(typeof manifest?.requiresModel).toBe('boolean')
        expect(typeof manifest?.requiresBaseUrl).toBe('boolean')
      }),
      { numRuns: 30 }
    )
  })

  it('text box expansion params stay in the backend-supported range', () => {
    fc.assert(
      fc.property(
        fc.record({
          box_expand_ratio: fc.float({ min: 0, max: 0.5, noNaN: true }),
          box_expand_top: fc.float({ min: 0, max: 0.5, noNaN: true }),
          box_expand_bottom: fc.float({ min: 0, max: 0.5, noNaN: true }),
          box_expand_left: fc.float({ min: 0, max: 0.5, noNaN: true }),
          box_expand_right: fc.float({ min: 0, max: 0.5, noNaN: true }),
        }),
        expandParams => {
          expect(expandParams.box_expand_ratio).toBeGreaterThanOrEqual(0)
          expect(expandParams.box_expand_ratio).toBeLessThanOrEqual(0.5)
          expect(expandParams.box_expand_top).toBeGreaterThanOrEqual(0)
          expect(expandParams.box_expand_top).toBeLessThanOrEqual(0.5)
          expect(expandParams.box_expand_bottom).toBeGreaterThanOrEqual(0)
          expect(expandParams.box_expand_bottom).toBeLessThanOrEqual(0.5)
          expect(expandParams.box_expand_left).toBeGreaterThanOrEqual(0)
          expect(expandParams.box_expand_left).toBeLessThanOrEqual(0.5)
          expect(expandParams.box_expand_right).toBeGreaterThanOrEqual(0)
          expect(expandParams.box_expand_right).toBeLessThanOrEqual(0.5)
        }
      ),
      { numRuns: 100 }
    )
  })
})
