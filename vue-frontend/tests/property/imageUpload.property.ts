import { describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import * as fc from 'fast-check'
import { useImageStore } from '@/stores/imageStore'
import { naturalSort, naturalSortCompare } from '@/utils'

const validFileNameArb = fc.stringOf(
  fc.constantFrom(...'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'),
  { minLength: 1, maxLength: 20 }
).map(name => `${name}.png`)

const mockDataURLArb = fc.constant('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')

function freshImageStore() {
  setActivePinia(createPinia())
  return useImageStore()
}

describe('image upload store properties', () => {
  it('adds one image record per uploaded file', () => {
    fc.assert(
      fc.property(
        fc.array(validFileNameArb, { minLength: 1, maxLength: 10 }),
        mockDataURLArb,
        (fileNames, dataURL) => {
          const store = freshImageStore()
          const initialCount = store.imageCount

          for (const fileName of fileNames) {
            store.addImage(fileName, dataURL)
          }

          expect(store.imageCount).toBe(initialCount + fileNames.length)

          const ids = store.images.map(img => img.id)
          expect(new Set(ids).size).toBe(ids.length)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('sets the initial image state for uploaded files', () => {
    fc.assert(
      fc.property(
        validFileNameArb,
        mockDataURLArb,
        (fileName, dataURL) => {
          const store = freshImageStore()

          store.addImage(fileName, dataURL)

          const addedImage = store.images.at(-1)

          expect(addedImage).toBeDefined()
          expect(addedImage?.translationStatus).toBe('pending')
          expect(addedImage?.translationFailed).toBe(false)
          expect(addedImage?.sourceAssetUrl).toBe(dataURL)
          expect(addedImage?.fileName).toBe(fileName)
          expect(addedImage?.translatedAssetUrl).toBeNull()
          expect(addedImage?.hasUnsavedChanges).toBe(false)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('orders generated file names through the shared natural-sort helper', () => {
    fc.assert(
      fc.property(
        fc.array(fc.string({ minLength: 1, maxLength: 20 }), { minLength: 2, maxLength: 20 }),
        (names) => {
          const sorted = naturalSort(names)

          for (let index = 0; index < sorted.length - 1; index += 1) {
            const current = sorted[index]
            const next = sorted[index + 1]
            if (current && next) {
              expect(naturalSortCompare(current, next)).toBeLessThanOrEqual(0)
            }
          }

          const remainingNames = [...names]
          for (const name of sorted) {
            const originalIndex = remainingNames.indexOf(name)
            expect(originalIndex).toBeGreaterThanOrEqual(0)
            remainingNames.splice(originalIndex, 1)
          }
          expect(remainingNames).toHaveLength(0)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('orders numeric file names by numeric segments', () => {
    const testCases = [
      { input: ['page_2.png', 'page_10.png', 'page_1.png'], expected: ['page_1.png', 'page_2.png', 'page_10.png'] },
      { input: ['img10.jpg', 'img2.jpg', 'img1.jpg'], expected: ['img1.jpg', 'img2.jpg', 'img10.jpg'] },
      { input: ['001.png', '010.png', '002.png'], expected: ['001.png', '002.png', '010.png'] },
    ]

    for (const { input, expected } of testCases) {
      expect(naturalSort(input)).toEqual(expected)
    }
  })

  it('keeps the current image index in range after uploads', () => {
    fc.assert(
      fc.property(
        fc.array(validFileNameArb, { minLength: 1, maxLength: 10 }),
        mockDataURLArb,
        (fileNames, dataURL) => {
          const store = freshImageStore()

          for (const fileName of fileNames) {
            store.addImage(fileName, dataURL)
          }

          expect(store.currentImageIndex).toBeGreaterThanOrEqual(0)
          expect(store.currentImageIndex).toBeLessThan(store.imageCount)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('keeps image list state valid after deleting the current image', () => {
    fc.assert(
      fc.property(
        fc.array(validFileNameArb, { minLength: 2, maxLength: 10 }),
        mockDataURLArb,
        fc.nat(),
        (fileNames, dataURL, deleteIndexSeed) => {
          const store = freshImageStore()

          for (const fileName of fileNames) {
            store.addImage(fileName, dataURL)
          }

          const countBeforeDelete = store.imageCount
          const deleteIndex = deleteIndexSeed % countBeforeDelete

          store.setCurrentImageIndex(deleteIndex)
          store.deleteCurrentImage()

          expect(store.imageCount).toBe(countBeforeDelete - 1)
          if (store.imageCount > 0) {
            expect(store.currentImageIndex).toBeGreaterThanOrEqual(0)
            expect(store.currentImageIndex).toBeLessThan(store.imageCount)
          } else {
            expect(store.currentImageIndex).toBe(-1)
          }
        }
      ),
      { numRuns: 100 }
    )
  })

  it('resets image state after clearing all images', () => {
    fc.assert(
      fc.property(
        fc.array(validFileNameArb, { minLength: 1, maxLength: 10 }),
        mockDataURLArb,
        (fileNames, dataURL) => {
          const store = freshImageStore()

          for (const fileName of fileNames) {
            store.addImage(fileName, dataURL)
          }

          store.clearImages()

          expect(store.imageCount).toBe(0)
          expect(store.currentImageIndex).toBe(-1)
          expect(store.hasImages).toBe(false)
          expect(store.currentImage).toBeNull()
        }
      ),
      { numRuns: 100 }
    )
  })
})
