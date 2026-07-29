import { readFileSync } from 'node:fs'
import { describe, expect, it, beforeEach } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import * as fc from 'fast-check'
import { useImageStore } from '@/stores/imageStore'
import {
  collectDownloadImageEntries,
  DOWNLOAD_FORMATS,
  resolveDownloadFileName,
} from '@/composables/useExportImport'

describe('download export property contracts', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  const validFileNameArb = fc
    .stringOf(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'), {
      minLength: 1,
      maxLength: 20,
    })
    .map(name => `${name}.png`)

  const mockDataURLArb = fc.constant(
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
  )

  it('uses the production helpers instead of copied download logic', () => {
    const source = readFileSync('tests/property/download.property.ts', 'utf8')
    const shadowFilenameHelper = 'function generateDownload' + 'FileName'
    const shadowImageList = 'const imageInfo' + 'List'
    const shadowStoreLoop = 'for (let i = 0; i < store.images.' + 'length; i++)'

    expect(source).toContain("from '@/composables/useExportImport'")
    expect(source).not.toContain(shadowFilenameHelper)
    expect(source).not.toContain(shadowImageList)
    expect(source).not.toContain(shadowStoreLoop)
  })

  it('resolves current-image filenames through the production helper', () => {
    fc.assert(
      fc.property(validFileNameArb, fc.nat(100), fc.constantFrom('translated', 'original'), (
        fileName,
        imageIndex,
        type,
      ) => {
        const downloadFileName = resolveDownloadFileName(fileName, imageIndex, type)

        expect(downloadFileName).toMatch(/\.png$/)
        expect(downloadFileName.startsWith(`${type}_`)).toBe(true)
        expect(downloadFileName).toContain(fileName.replace(/\.[^/.]+$/, ''))
      }),
      { numRuns: 100 },
    )
  })

  it('uses the current fallback filename when the image has no name', () => {
    fc.assert(
      fc.property(fc.nat(100), fc.constantFrom('translated', 'original'), (imageIndex, type) => {
        expect(resolveDownloadFileName('', imageIndex, type)).toBe(
          `${type}_image_${imageIndex}.png`,
        )
      }),
      { numRuns: 100 },
    )
  })

  it('collects every downloadable image through the production helper', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            fileName: validFileNameArb,
            hasOriginal: fc.boolean(),
            hasTranslated: fc.boolean(),
          }),
          { minLength: 1, maxLength: 10 },
        ),
        mockDataURLArb,
        (imageConfigs, dataURL) => {
          setActivePinia(createPinia())
          const store = useImageStore()

          for (const config of imageConfigs) {
            store.addImage(config.fileName, config.hasOriginal ? dataURL : '')
            const image = store.images.at(-1)
            if (image && config.hasTranslated) {
              image.translatedAssetUrl = dataURL
            }
          }

          const entries = collectDownloadImageEntries(store.images)
          const validImageCount = imageConfigs.filter(
            config => config.hasOriginal || config.hasTranslated,
          ).length

          expect(entries).toHaveLength(validImageCount)
          expect(entries.map(entry => entry.index)).toEqual(
            entries.map(entry => entry.index).toSorted((left, right) => left - right),
          )
          for (const entry of entries) {
            expect(entry.index).toBeGreaterThanOrEqual(0)
            expect(entry.index).toBeLessThan(store.images.length)
          }
        },
      ),
      { numRuns: 100 },
    )
  })

  it('prefers translated data when both image variants exist', () => {
    fc.assert(
      fc.property(validFileNameArb, mockDataURLArb, (fileName, dataURL) => {
        setActivePinia(createPinia())
        const store = useImageStore()
        store.addImage(fileName, dataURL)
        const image = store.images[0]
        if (image) {
          image.translatedAssetUrl = dataURL
        }

        expect(collectDownloadImageEntries(store.images)).toEqual([
          { index: 0, type: 'translated' },
        ])
      }),
      { numRuns: 100 },
    )
  })

  it('keeps the product download formats as the source of truth', () => {
    expect(DOWNLOAD_FORMATS).toEqual(['zip', 'pdf', 'cbz'])
  })

  it('returns no batch entries for an empty image list', () => {
    const store = useImageStore()

    expect(collectDownloadImageEntries(store.images)).toEqual([])
  })
})
