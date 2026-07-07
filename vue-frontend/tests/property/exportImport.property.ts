import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import * as fc from 'fast-check'
import { useImageStore } from '@/stores/imageStore'
import { useExportImport, type ExportTextData } from '@/composables/useExportImport'
import type { BubbleState } from '@/types/bubble'

describe('text export/import properties', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  const validTextArb = fc.string({ minLength: 0, maxLength: 100 })
  const concreteDirectionArb = fc.constantFrom('vertical', 'horizontal') as fc.Arbitrary<
    'vertical' | 'horizontal'
  >
  const bubbleCoordsArb = fc
    .record({
      x1: fc.nat(1000),
      y1: fc.nat(1000),
      width: fc.integer({ min: 1, max: 1000 }),
      height: fc.integer({ min: 1, max: 1000 }),
    })
    .map(
      ({ x1, y1, width, height }) =>
        [x1, y1, x1 + width, y1 + height] as [number, number, number, number],
    )

  const bubbleStateArb: fc.Arbitrary<BubbleState> = fc.record({
    coords: bubbleCoordsArb,
    polygon: fc.constant([]) as fc.Arbitrary<number[][]>,
    originalText: validTextArb,
    translatedText: validTextArb,
    textboxText: validTextArb,
    fontSize: fc.integer({ min: 10, max: 100 }),
    fontFamily: fc.constant('fonts/STSONG.TTF'),
    textDirection: fc.constantFrom('vertical', 'horizontal', 'auto') as fc.Arbitrary<
      'vertical' | 'horizontal' | 'auto'
    >,
    autoTextDirection: concreteDirectionArb,
    textColor: fc.constant('#000000'),
    fillColor: fc.constant('#FFFFFF'),
    rotationAngle: fc.integer({ min: 0, max: 360 }),
    position: fc.constant({ x: 0, y: 0 }),
    strokeEnabled: fc.boolean(),
    strokeColor: fc.constant('#FFFFFF'),
    strokeWidth: fc.integer({ min: 1, max: 10 }),
    lineSpacing: fc.double({ min: 0.5, max: 3, noNaN: true }),
    textAlign: fc.constantFrom('start', 'center', 'end') as fc.Arbitrary<
      'start' | 'center' | 'end'
    >,
    inpaintMethod: fc.constantFrom('solid', 'lama_mpe', 'litelama') as fc.Arbitrary<
      'solid' | 'lama_mpe' | 'litelama'
    >,
    textlines: fc.constant([]),
  })
  const mockDataUrlArb = fc.constant(
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
  )

  function addImageWithBubbles(fileName: string, dataUrl: string, bubbleStates: BubbleState[]): void {
    const store = useImageStore()
    store.addImage(fileName, dataUrl)
    const image = store.images.at(-1)
    if (image) {
      image.bubbleStates = bubbleStates
    }
  }

  function exportCurrentStoreText(): ExportTextData[] | null {
    const { exportTextToJson } = useExportImport()
    return exportTextToJson()
  }

  it('exports one image record for each image in the store', () => {
    fc.assert(
      fc.property(
        fc.array(fc.array(bubbleStateArb, { minLength: 1, maxLength: 5 }), {
          minLength: 1,
          maxLength: 5,
        }),
        mockDataUrlArb,
        (bubbleStatesPerImage, dataUrl) => {
          setActivePinia(createPinia())

          for (const [index, bubbleStates] of bubbleStatesPerImage.entries()) {
            addImageWithBubbles(`image_${index}.png`, dataUrl, bubbleStates)
          }

          const exportData = exportCurrentStoreText()

          expect(exportData).not.toBeNull()
          expect(exportData).toHaveLength(bubbleStatesPerImage.length)
          expect(exportData?.map(imageData => imageData.imageIndex)).toEqual(
            bubbleStatesPerImage.map((_, index) => index),
          )
          for (const [imageIndex, imageData] of (exportData ?? []).entries()) {
            const expectedBubbles = bubbleStatesPerImage[imageIndex] ?? []
            expect(imageData.bubbles).toHaveLength(expectedBubbles.length)
            expect(imageData.bubbles.map(bubble => bubble.bubbleIndex)).toEqual(
              expectedBubbles.map((_, bubbleIndex) => bubbleIndex),
            )
          }
        },
      ),
      { numRuns: 100 },
    )
  })

  it('exports original and translated text from production bubble states', () => {
    fc.assert(
      fc.property(bubbleStateArb, mockDataUrlArb, (bubbleState, dataUrl) => {
        setActivePinia(createPinia())
        addImageWithBubbles('test.png', dataUrl, [bubbleState])

        const exportData = exportCurrentStoreText()
        const exportedBubble = exportData?.[0]?.bubbles[0]

        expect(exportedBubble?.original).toBe(bubbleState.originalText || '')
        expect(exportedBubble?.translated).toBe(
          bubbleState.translatedText || bubbleState.textboxText || '',
        )
      }),
      { numRuns: 100 },
    )
  })

  it('exports a concrete text direction instead of auto', () => {
    fc.assert(
      fc.property(bubbleStateArb, mockDataUrlArb, (bubbleState, dataUrl) => {
        setActivePinia(createPinia())
        addImageWithBubbles('test.png', dataUrl, [bubbleState])

        const exportedDirection = exportCurrentStoreText()?.[0]?.bubbles[0]?.textDirection

        expect(exportedDirection).not.toBe('auto')
        expect(['vertical', 'horizontal']).toContain(exportedDirection)
        expect(exportedDirection).toBe(
          bubbleState.textDirection === 'auto'
            ? bubbleState.autoTextDirection
            : bubbleState.textDirection,
        )
      }),
      { numRuns: 100 },
    )
  })

  it('returns null when there are no images to export', () => {
    expect(exportCurrentStoreText()).toBeNull()
  })

  it('serializes exported text as valid JSON', () => {
    fc.assert(
      fc.property(fc.array(bubbleStateArb, { minLength: 1, maxLength: 5 }), mockDataUrlArb, (
        bubbleStates,
        dataUrl,
      ) => {
        setActivePinia(createPinia())
        addImageWithBubbles('test.png', dataUrl, bubbleStates)

        const exportData = exportCurrentStoreText()
        expect(exportData).not.toBeNull()

        const parsed = JSON.parse(JSON.stringify(exportData)) as ExportTextData[]
        expect(Array.isArray(parsed)).toBe(true)
        expect(parsed).toHaveLength(exportData?.length ?? 0)
      }),
      { numRuns: 100 },
    )
  })

  it('round-trips exported text through JSON without changing the schema', () => {
    fc.assert(
      fc.property(
        validTextArb,
        validTextArb,
        concreteDirectionArb,
        mockDataUrlArb,
        (originalText, translatedText, textDirection, dataUrl) => {
          setActivePinia(createPinia())
          const bubbleState: BubbleState = {
            coords: [0, 0, 100, 100],
            polygon: [],
            originalText,
            translatedText,
            textboxText: translatedText,
            fontSize: 25,
            fontFamily: 'fonts/STSONG.TTF',
            textDirection,
            autoTextDirection: textDirection,
            textColor: '#000000',
            fillColor: '#FFFFFF',
            rotationAngle: 0,
            position: { x: 0, y: 0 },
            strokeEnabled: false,
            strokeColor: '#FFFFFF',
            strokeWidth: 3,
            lineSpacing: 1,
            textAlign: 'center',
            inpaintMethod: 'solid',
            textlines: [],
          }
          addImageWithBubbles('test.png', dataUrl, [bubbleState])

          const importedData = JSON.parse(JSON.stringify(exportCurrentStoreText())) as ExportTextData[]
          const importedBubble = importedData[0]?.bubbles[0]

          expect(importedData).toHaveLength(1)
          expect(importedData[0]?.bubbles).toHaveLength(1)
          expect(importedBubble?.original).toBe(originalText)
          expect(importedBubble?.translated).toBe(translatedText)
          expect(importedBubble?.textDirection).toBe(textDirection)
        },
      ),
      { numRuns: 100 },
    )
  })
})
