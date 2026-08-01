import type { ImageData, ImageDataLoadInput } from '@/types/image'
import type { useImageStore } from '@/stores/imageStore'

type ImageStore = ReturnType<typeof useImageStore>

let nextPageId = 0

export interface TestImageInput {
  fileName: string
  sourceAssetUrl: string
  overrides?: Partial<ImageDataLoadInput>
}

export function createTestImage(
  fileName: string,
  sourceAssetUrl: string,
  overrides: Partial<ImageDataLoadInput> = {},
): ImageDataLoadInput {
  nextPageId += 1
  return {
    id: `test-page-${nextPageId}`,
    fileName,
    sourceAssetUrl,
    translatedAssetUrl: null,
    cleanAssetUrl: null,
    bubbleStates: null,
    translationStatus: 'pending',
    translationFailed: false,
    hasUnsavedChanges: false,
    ...overrides,
  }
}

export function setTestImages(
  store: ImageStore,
  imageInputs: TestImageInput[],
): ImageData[] {
  store.setImages(imageInputs.map(({ fileName, sourceAssetUrl, overrides }) =>
    createTestImage(fileName, sourceAssetUrl, overrides),
  ))
  return store.images
}

export function addTestImage(
  store: ImageStore,
  fileName: string,
  sourceAssetUrl: string,
  overrides: Partial<ImageDataLoadInput> = {},
): ImageData {
  store.setImages([
    ...store.images,
    createTestImage(fileName, sourceAssetUrl, overrides),
  ])
  const image = store.images.at(-1)
  if (!image) throw new Error('测试图片载入失败')
  return image
}
