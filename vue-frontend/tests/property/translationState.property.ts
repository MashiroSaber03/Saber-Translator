import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import type { TranslationStatus } from '@/types/image'

type ImageInput = {
  fileName: string
  originalDataURL: string
}

type ImageStore = ReturnType<typeof useImageStore>

function createStore(): ImageStore {
  setActivePinia(createPinia())
  return useImageStore()
}

function addImage(store: ImageStore, imageInput: ImageInput) {
  return store.addImage(imageInput.fileName, imageInput.originalDataURL)
}

const fileNameArbitrary = fc
  .stringOf(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789_-'.split('')), {
    minLength: 1,
    maxLength: 48,
  })
  .map(name => `${name}.png`)

const base64DataUrlArbitrary = fc
  .stringOf(fc.constantFrom(...'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'.split('')), {
    minLength: 10,
    maxLength: 100,
  })
  .map(data => `data:image/png;base64,${data}`)

const imageInputArbitrary: fc.Arbitrary<ImageInput> = fc.record({
  fileName: fileNameArbitrary,
  originalDataURL: base64DataUrlArbitrary,
})

const translationStatusArbitrary = fc.constantFrom<TranslationStatus>(
  'pending',
  'processing',
  'completed',
  'failed',
)

const activeStatusArbitrary = fc.constantFrom<TranslationStatus>(
  'pending',
  'processing',
  'completed',
)

describe('translation state properties', () => {
  it('adds images in the pending state and can move them into processing', () => {
    fc.assert(
      fc.property(imageInputArbitrary, imageInput => {
        const store = createStore()
        const image = addImage(store, imageInput)

        expect(image.fileName).toBe(imageInput.fileName)
        expect(image.originalDataURL).toBe(imageInput.originalDataURL)
        expect(image.translationStatus).toBe('pending')
        expect(image.translationFailed).toBe(false)
        expect(store.pendingImageCount).toBe(1)

        store.setTranslationStatus(0, 'processing')

        expect(store.images[0]?.translationStatus).toBe('processing')
        expect(store.images[0]?.translationFailed).toBe(false)
        expect(store.pendingImageCount).toBe(0)
      }),
    )
  })

  it('records failure messages and clears stale errors when work resumes', () => {
    fc.assert(
      fc.property(
        imageInputArbitrary,
        fc.string({ minLength: 1, maxLength: 100 }),
        activeStatusArbitrary,
        (imageInput, errorMessage, recoveryStatus) => {
          const store = createStore()
          addImage(store, imageInput)

          store.setTranslationStatus(0, 'failed', errorMessage)

          expect(store.images[0]?.translationStatus).toBe('failed')
          expect(store.images[0]?.translationFailed).toBe(true)
          expect(store.images[0]?.errorMessage).toBe(errorMessage)
          expect(store.failedImageCount).toBe(1)

          store.setTranslationStatus(0, recoveryStatus)

          expect(store.images[0]?.translationStatus).toBe(recoveryStatus)
          expect(store.images[0]?.translationFailed).toBe(false)
          expect(store.images[0]?.errorMessage).toBeUndefined()
          expect(store.failedImageCount).toBe(0)
        },
      ),
    )
  })

  it('returns failed image indexes in list order', () => {
    fc.assert(
      fc.property(fc.array(fc.boolean(), { minLength: 1, maxLength: 10 }), failedFlags => {
        const store = createStore()

        failedFlags.forEach((shouldFail, index) => {
          store.addImage(`image${index}.png`, `data:image/png;base64,test${index}`)
          store.setTranslationStatus(index, shouldFail ? 'failed' : 'completed', `error-${index}`)
        })

        const expectedFailedIndices = failedFlags
          .map((shouldFail, index) => (shouldFail ? index : -1))
          .filter(index => index >= 0)

        expect(store.getFailedImageIndices()).toEqual(expectedFailedIndices)
        expect(store.failedImageCount).toBe(expectedFailedIndices.length)
        expect(store.completedImageCount).toBe(failedFlags.length - expectedFailedIndices.length)
      }),
    )
  })

  it('marks the selected image as failed without changing the selection', () => {
    fc.assert(
      fc.property(
        fc.array(imageInputArbitrary, { minLength: 1, maxLength: 6 }),
        fc.nat(),
        fc.string({ minLength: 1, maxLength: 100 }),
        (imageInputs, targetSeed, errorMessage) => {
          const store = createStore()
          imageInputs.forEach(imageInput => addImage(store, imageInput))
          const targetIndex = targetSeed % store.imageCount

          store.setCurrentImageIndex(targetIndex)
          store.markCurrentAsFailed(errorMessage)

          expect(store.currentImageIndex).toBe(targetIndex)
          expect(store.currentImage?.translationStatus).toBe('failed')
          expect(store.currentImage?.translationFailed).toBe(true)
          expect(store.currentImage?.errorMessage).toBe(errorMessage)
          expect(store.getFailedImageIndices()).toEqual([targetIndex])
        },
      ),
    )
  })

  it('updates one image status without mutating neighboring image data', () => {
    fc.assert(
      fc.property(
        fc.array(imageInputArbitrary, { minLength: 2, maxLength: 6 }),
        translationStatusArbitrary,
        fc.nat(),
        fc.string({ minLength: 1, maxLength: 100 }),
        (imageInputs, nextStatus, targetSeed, errorMessage) => {
          const store = createStore()
          imageInputs.forEach(imageInput => addImage(store, imageInput))
          const targetIndex = targetSeed % store.imageCount
          const snapshots = store.images.map(image => ({
            fileName: image.fileName,
            originalDataURL: image.originalDataURL,
            translationStatus: image.translationStatus,
            translationFailed: image.translationFailed,
          }))

          store.setTranslationStatus(targetIndex, nextStatus, errorMessage)

          store.images.forEach((image, index) => {
            const snapshot = snapshots[index]
            expect(image.fileName).toBe(snapshot?.fileName)
            expect(image.originalDataURL).toBe(snapshot?.originalDataURL)

            if (index === targetIndex) {
              expect(image.translationStatus).toBe(nextStatus)
              expect(image.translationFailed).toBe(nextStatus === 'failed')
              if (nextStatus === 'failed') {
                expect(image.errorMessage).toBe(errorMessage)
              } else {
                expect(image.errorMessage).toBeUndefined()
              }
            } else {
              expect(image.translationStatus).toBe(snapshot?.translationStatus)
              expect(image.translationFailed).toBe(snapshot?.translationFailed)
              expect(image.errorMessage).toBeUndefined()
            }
          })
        },
      ),
    )
  })

  it('resets every translation state back to pending', () => {
    fc.assert(
      fc.property(fc.array(fc.boolean(), { minLength: 1, maxLength: 10 }), failedFlags => {
        const store = createStore()

        failedFlags.forEach((shouldFail, index) => {
          store.addImage(`image${index}.png`, `data:image/png;base64,test${index}`)
          store.setTranslationStatus(index, shouldFail ? 'failed' : 'completed', `error-${index}`)
        })

        store.resetAllTranslationStatus()

        expect(store.pendingImageCount).toBe(failedFlags.length)
        expect(store.failedImageCount).toBe(0)
        expect(store.completedImageCount).toBe(0)
        store.images.forEach(image => {
          expect(image.translationStatus).toBe('pending')
          expect(image.translationFailed).toBe(false)
          expect(image.errorMessage).toBeUndefined()
        })
      }),
    )
  })
})
