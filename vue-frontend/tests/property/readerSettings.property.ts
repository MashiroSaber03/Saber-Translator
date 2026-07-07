import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as fc from 'fast-check'
import {
  READER_BG_COLOR_PRESETS,
  READER_SETTINGS_KEY,
  loadReaderSettings,
  parseReaderSettingsPayload,
  saveReaderSettings,
  toStoredReaderSettings,
  type ReaderSettings,
} from '@/components/reader/readerSettings'

const readerBgColorValues = READER_BG_COLOR_PRESETS.map((preset) => preset.value)
const validImageWidthArb = fc.integer({ min: 50, max: 100 })
const validImageGapArb = fc.integer({ min: 0, max: 50 })
const validBgColorArb = fc.constantFrom(readerBgColorValues[0]!, ...readerBgColorValues.slice(1))
const validReaderSettingsArb: fc.Arbitrary<ReaderSettings> = fc.record({
  imageWidth: validImageWidthArb,
  imageGap: validImageGapArb,
  bgColor: validBgColorArb,
})

describe('reader settings persistence properties', () => {
  let localStorageMock: Record<string, string> = {}

  beforeEach(() => {
    localStorageMock = {}
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation((key: string) => localStorageMock[key] ?? null)
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation((key: string, value: string) => {
      localStorageMock[key] = value
    })
    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation((key: string) => {
      delete localStorageMock[key]
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('loads the same current-schema settings after saving', () => {
    fc.assert(
      fc.property(validReaderSettingsArb, (settings) => {
        localStorageMock = {}

        expect(saveReaderSettings(settings)).toBe(true)
        expect(localStorageMock[READER_SETTINGS_KEY]).toBe(JSON.stringify(toStoredReaderSettings(settings)))
        expect(loadReaderSettings()).toEqual(settings)
      }),
      { numRuns: 100 }
    )
  })

  it('loads the last saved reader settings payload', () => {
    fc.assert(
      fc.property(fc.array(validReaderSettingsArb, { minLength: 2, maxLength: 10 }), (settingsArray) => {
        localStorageMock = {}

        for (const settings of settingsArray) {
          saveReaderSettings(settings)
        }

        expect(loadReaderSettings()).toEqual(settingsArray.at(-1))
      }),
      { numRuns: 100 }
    )
  })

  it('rejects incomplete settings payloads', () => {
    fc.assert(
      fc.property(validImageWidthArb, (imageWidth) => {
        localStorageMock = {
          [READER_SETTINGS_KEY]: JSON.stringify({ imageWidth }),
        }

        expect(loadReaderSettings()).toBeNull()
      }),
      { numRuns: 100 }
    )
  })

  it('rejects out-of-range current-schema payloads', () => {
    fc.assert(
      fc.property(
        fc.oneof(fc.integer({ max: 49 }), fc.integer({ min: 101 })),
        fc.oneof(fc.integer({ max: -1 }), fc.integer({ min: 51 })),
        fc.string().filter((value) => !readerBgColorValues.includes(value)),
        (imageWidth, imageGap, bgColor) => {
          expect(parseReaderSettingsPayload(JSON.stringify({
            readerSettingsSchemaVersion: 1,
            imageWidth,
            imageGap,
            bgColor,
          }))).toBeNull()
        }
      ),
      { numRuns: 100 }
    )
  })

  it('round-trips stored current-schema payloads through JSON', () => {
    fc.assert(
      fc.property(validReaderSettingsArb, (settings) => {
        const stored = toStoredReaderSettings(settings)
        const parsed = parseReaderSettingsPayload(JSON.stringify(stored))

        expect(parsed).toEqual(settings)
      }),
      { numRuns: 100 }
    )
  })
})
