import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'

type ImageInput = {
  fileName: string
  originalDataURL: string
}

function createStore(): ReturnType<typeof useImageStore> {
  setActivePinia(createPinia())
  return useImageStore()
}

const fileNameArbitrary = fc
  .stringOf(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789_-'.split('')), {
    minLength: 1,
    maxLength: 50,
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

describe('image store properties', () => {
  it('adds every image in a batch and selects the first image for a fresh store', () => {
    fc.assert(
      fc.property(fc.array(imageInputArbitrary, { minLength: 1, maxLength: 10 }), imageList => {
        const store = createStore()
        const addedImages = store.addImages(imageList)

        expect(addedImages).toHaveLength(imageList.length)
        expect(store.imageCount).toBe(imageList.length)
        expect(store.hasImages).toBe(true)
        expect(store.currentImageIndex).toBe(0)
        expect(store.currentImage?.fileName).toBe(imageList[0]?.fileName)
      }),
    )
  })

  it('creates images with generated ids and the current default translation state', () => {
    fc.assert(
      fc.property(imageInputArbitrary, imageInput => {
        const store = createStore()
        const image = store.addImage(imageInput.fileName, imageInput.originalDataURL)

        expect(image.id).toEqual(expect.any(String))
        expect(image.id.length).toBeGreaterThan(0)
        expect(image.fileName).toBe(imageInput.fileName)
        expect(image.originalDataURL).toBe(imageInput.originalDataURL)
        expect(image.translatedDataURL).toBeNull()
        expect(image.cleanImageData).toBeNull()
        expect(image.bubbleStates).toBeNull()
        expect(image.translationStatus).toBe('pending')
        expect(image.translationFailed).toBe(false)
        expect(image.hasUnsavedChanges).toBe(false)
      }),
    )
  })

  it('assigns unique ids to batch-created images', () => {
    fc.assert(
      fc.property(fc.array(imageInputArbitrary, { minLength: 2, maxLength: 20 }), imageList => {
        const store = createStore()
        const addedImages = store.addImages(imageList)
        const ids = addedImages.map(image => image.id)

        expect(new Set(ids).size).toBe(ids.length)
      }),
    )
  })

  it('deletes a valid image and keeps the current selection inside the list', () => {
    fc.assert(
      fc.property(
        fc.array(imageInputArbitrary, { minLength: 2, maxLength: 10 }),
        fc.nat(),
        (imageList, deleteIndexSeed) => {
          const store = createStore()
          store.addImages(imageList)
          const countAfterAdd = store.imageCount
          const deleteIndex = deleteIndexSeed % countAfterAdd

          expect(store.deleteImage(deleteIndex)).toBe(true)
          expect(store.imageCount).toBe(countAfterAdd - 1)
          expect(store.currentImageIndex).toBeGreaterThanOrEqual(store.imageCount > 0 ? 0 : -1)
          expect(store.currentImageIndex).toBeLessThan(store.imageCount)
        },
      ),
    )
  })

  it('clears images, selection, and batch progress state together', () => {
    fc.assert(
      fc.property(fc.array(imageInputArbitrary, { minLength: 1, maxLength: 10 }), imageList => {
        const store = createStore()
        store.addImages(imageList)
        store.setBatchTranslationInProgress(true)

        store.clearImages()

        expect(store.imageCount).toBe(0)
        expect(store.currentImageIndex).toBe(-1)
        expect(store.currentImage).toBeNull()
        expect(store.hasImages).toBe(false)
        expect(store.isBatchTranslationInProgress).toBe(false)
      }),
    )
  })

  it('accepts valid image indexes and ignores indexes outside the current list', () => {
    fc.assert(
      fc.property(
        fc.array(imageInputArbitrary, { minLength: 3, maxLength: 10 }),
        fc.nat(),
        (imageList, targetIndexSeed) => {
          const store = createStore()
          store.addImages(imageList)
          const targetIndex = targetIndexSeed % store.imageCount

          store.setCurrentImageIndex(targetIndex)
          expect(store.currentImageIndex).toBe(targetIndex)
          expect(store.currentImage?.fileName).toBe(imageList[targetIndex]?.fileName)

          store.setCurrentImageIndex(store.imageCount)
          expect(store.currentImageIndex).toBe(targetIndex)

          store.setCurrentImageIndex(-2)
          expect(store.currentImageIndex).toBe(targetIndex)
        },
      ),
    )
  })

  it('moves previous and next only when the boundary allows it', () => {
    fc.assert(
      fc.property(fc.array(imageInputArbitrary, { minLength: 3, maxLength: 10 }), imageList => {
        const store = createStore()
        store.addImages(imageList)

        expect(store.canGoPrevious).toBe(false)
        expect(store.canGoNext).toBe(true)
        expect(store.goToNext()).toBe(true)
        expect(store.currentImageIndex).toBe(1)
        expect(store.canGoPrevious).toBe(true)
        expect(store.goToPrevious()).toBe(true)
        expect(store.currentImageIndex).toBe(0)
        expect(store.goToPrevious()).toBe(false)
        expect(store.currentImageIndex).toBe(0)
      }),
    )
  })

  it('updates current image fields without changing the selected image', () => {
    fc.assert(
      fc.property(
        imageInputArbitrary,
        fc.integer({ min: 10, max: 100 }),
        fc.hexaString({ minLength: 6, maxLength: 6 }),
        (imageInput, fontSize, colorHex) => {
          const store = createStore()
          store.addImage(imageInput.fileName, imageInput.originalDataURL)
          const selectedId = store.currentImage?.id
          const textColor = `#${colorHex}`

          store.updateCurrentImage({ fontSize, textColor })

          expect(store.currentImage?.id).toBe(selectedId)
          expect(store.currentImage?.fontSize).toBe(fontSize)
          expect(store.currentImage?.textColor).toBe(textColor)
        },
      ),
    )
  })

  it('tracks failed images and clears stale errors when status returns to active work', () => {
    fc.assert(
      fc.property(
        fc.array(imageInputArbitrary, { minLength: 1, maxLength: 10 }),
        fc.nat(),
        fc.string({ minLength: 1, maxLength: 80 }),
        (imageList, failedIndexSeed, errorMessage) => {
          const store = createStore()
          store.addImages(imageList)
          const failedIndex = failedIndexSeed % store.imageCount

          store.setTranslationStatus(failedIndex, 'failed', errorMessage)
          expect(store.failedImageCount).toBe(1)
          expect(store.getFailedImageIndices()).toEqual([failedIndex])

          store.setTranslationStatus(failedIndex, 'processing')
          expect(store.failedImageCount).toBe(0)
          expect(store.images[failedIndex]?.translationFailed).toBe(false)
          expect(store.images[failedIndex]?.errorMessage).toBeUndefined()
        },
      ),
    )
  })
})
