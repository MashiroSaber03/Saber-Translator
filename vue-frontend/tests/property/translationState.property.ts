import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import type { TranslationStatus } from '@/types/image'
import { addTestImage, setTestImages } from '../helpers/imageFixtures'

type ImageInput = {
  fileName: string
  sourceAssetUrl: string
}

type ImageStore = ReturnType<typeof useImageStore>

function createStore(): ImageStore {
  setActivePinia(createPinia())
  return useImageStore()
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
  sourceAssetUrl: base64DataUrlArbitrary,
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
        const image = addTestImage(store, imageInput.fileName, imageInput.sourceAssetUrl)

        expect(image.fileName).toBe(imageInput.fileName)
        expect(image.sourceAssetUrl).toBe(imageInput.sourceAssetUrl)
        expect(image.translationStatus).toBe('pending')
        expect(image.translationFailed).toBe(false)
        expect(store.images.filter(item => item.translationStatus === 'pending')).toHaveLength(1)

        store.setTranslationStatus(0, 'processing')

        expect(store.images[0]?.translationStatus).toBe('processing')
        expect(store.images[0]?.translationFailed).toBe(false)
        expect(store.images.filter(item => item.translationStatus === 'pending')).toHaveLength(0)
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
          addTestImage(store, imageInput.fileName, imageInput.sourceAssetUrl)

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

        setTestImages(store, failedFlags.map((_, index) => ({
          fileName: `image${index}.png`,
          sourceAssetUrl: `data:image/png;base64,test${index}`,
        })))
        failedFlags.forEach((shouldFail, index) => {
          store.setTranslationStatus(index, shouldFail ? 'failed' : 'completed', `error-${index}`)
        })

        const expectedFailedIndices = failedFlags
          .map((shouldFail, index) => (shouldFail ? index : -1))
          .filter(index => index >= 0)

        expect(store.images.flatMap((image, index) => image.translationFailed ? [index] : [])).toEqual(
          expectedFailedIndices,
        )
        expect(store.failedImageCount).toBe(expectedFailedIndices.length)
        expect(store.images.filter(item => item.translationStatus === 'completed')).toHaveLength(
          failedFlags.length - expectedFailedIndices.length,
        )
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
          setTestImages(store, imageInputs)
          const targetIndex = targetSeed % store.imageCount

          store.setCurrentImageIndex(targetIndex)
          store.setTranslationStatus(targetIndex, 'failed', errorMessage)

          expect(store.currentImageIndex).toBe(targetIndex)
          expect(store.currentImage?.translationStatus).toBe('failed')
          expect(store.currentImage?.translationFailed).toBe(true)
          expect(store.currentImage?.errorMessage).toBe(errorMessage)
          expect(store.images.flatMap((image, index) => image.translationFailed ? [index] : [])).toEqual(
            [targetIndex],
          )
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
          setTestImages(store, imageInputs)
          const targetIndex = targetSeed % store.imageCount
          const snapshots = store.images.map(image => ({
            fileName: image.fileName,
            sourceAssetUrl: image.sourceAssetUrl,
            translationStatus: image.translationStatus,
            translationFailed: image.translationFailed,
          }))

          store.setTranslationStatus(targetIndex, nextStatus, errorMessage)

          store.images.forEach((image, index) => {
            const snapshot = snapshots[index]
            expect(image.fileName).toBe(snapshot?.fileName)
            expect(image.sourceAssetUrl).toBe(snapshot?.sourceAssetUrl)

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

})
