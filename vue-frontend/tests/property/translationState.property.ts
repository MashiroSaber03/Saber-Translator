/**
 * 翻译状态管理属性测试
 * 使用 fast-check 进行属性基测试，验证翻译状态流转的一致性
 *
 * Feature: vue-frontend-migration, Property 20: 翻译状态流转一致性
 * Validates: Requirements 4.4, 4.5
 */
import { describe, it, beforeEach, afterEach, vi } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import type { TranslationStatus } from '@/types/image'

describe('翻译状态管理属性测试', () => {
  // 模拟 localStorage
  let localStorageMock: Record<string, string> = {}

  beforeEach(() => {
    // 重置 localStorage 模拟
    localStorageMock = {}

    // 模拟 localStorage
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation((key: string) => {
      return localStorageMock[key] || null
    })

    vi.spyOn(Storage.prototype, 'setItem').mockImplementation((key: string, value: string) => {
      localStorageMock[key] = value
    })

    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation((key: string) => {
      delete localStorageMock[key]
    })

    // 重置 Pinia
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  /**
   * 生成有效的翻译状态
   */
  const validTranslationStatusArb = fc.constantFrom<TranslationStatus>(
    'pending',
    'processing',
    'completed',
    'failed'
  )

  /**
   * 生成有效的图片数据
   */
  const validImageDataArb = fc.record({
    originalDataURL: fc.string({ minLength: 10, maxLength: 100 }).map((s) => `data:image/png;base64,${s}`),
    fileName: fc.string({ minLength: 1, maxLength: 50 }).map((s) => `${s}.png`)
  })

  /**
   * Feature: vue-frontend-migration, Property 20: 翻译状态流转一致性
   * Validates: Requirements 4.4, 4.5
   *
   * 对于任意图片，状态从 pending → processing → completed/failed 流转应正确
   */
  it('翻译状态流转一致性 - pending → processing', () => {
    fc.assert(
      fc.property(validImageDataArb, (imageData) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())

        const store = useImageStore()

        // 添加图片
        store.addImage(imageData.originalDataURL, imageData.fileName)

        // 验证初始状态为 pending
        const image = store.images[0]
        if (!image || image.translationStatus !== 'pending') return false

        // 设置状态为 processing
        store.setTranslationStatus(0, 'processing')

        // 验证状态已更新
        return store.images[0]?.translationStatus === 'processing'
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 20: 翻译状态流转一致性
   * Validates: Requirements 4.4, 4.5
   *
   * 对于任意图片，状态从 processing → completed 流转应正确
   */
  it('翻译状态流转一致性 - processing → completed', () => {
    fc.assert(
      fc.property(validImageDataArb, (imageData) => {
        // 每次迭代重新创建 Pinia 实例
        setActivePinia(createPinia())

        const store = useImageStore()

        // 添加图片并设置为 processing
        store.addImage(imageData.originalDataURL, imageData.fileName)
        store.setTranslationStatus(0, 'processing')

        // 设置状态为 completed
        store.setTranslationStatus(0, 'completed')

        // 验证状态已更新
        const image = store.images[0]
        return image?.translationStatus === 'completed' && image?.translationFailed === false
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 20: 翻译状态流转一致性
   * Validates: Requirements 4.4, 4.5
   *
   * 对于任意图片，状态从 processing → failed 流转应正确，并记录错误信息
   */
  it('翻译状态流转一致性 - processing → failed', () => {
    fc.assert(
      fc.property(
        validImageDataArb,
        fc.string({ minLength: 1, maxLength: 100 }),
        (imageData, errorMessage) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())

          const store = useImageStore()

          // 添加图片并设置为 processing
          store.addImage(imageData.originalDataURL, imageData.fileName)
          store.setTranslationStatus(0, 'processing')

          // 设置状态为 failed，带错误信息
          store.setTranslationStatus(0, 'failed', errorMessage)

          // 验证状态已更新
          const image = store.images[0]
          return (
            image?.translationStatus === 'failed' &&
            image?.translationFailed === true &&
            image?.errorMessage === errorMessage
          )
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 20: 翻译状态流转一致性
   * Validates: Requirements 4.4, 4.5
   *
   * 获取失败图片索引应正确返回所有失败的图片
   */
  it('获取失败图片索引一致性', () => {
    fc.assert(
      fc.property(
        fc.array(fc.boolean(), { minLength: 1, maxLength: 10 }),
        (failedFlags) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())

          const store = useImageStore()

          // 添加图片并设置状态
          failedFlags.forEach((shouldFail, index) => {
            store.addImage(`data:image/png;base64,test${index}`, `image${index}.png`)
            if (shouldFail) {
              store.setTranslationStatus(index, 'failed', '测试错误')
            } else {
              store.setTranslationStatus(index, 'completed')
            }
          })

          // 获取失败图片索引
          const failedIndices = store.getFailedImageIndices()

          // 验证失败图片索引正确
          const expectedFailedIndices = failedFlags
            .map((failed, index) => (failed ? index : -1))
            .filter((index) => index !== -1)

          // 验证数量一致
          if (failedIndices.length !== expectedFailedIndices.length) return false

          // 验证每个索引都正确
          return expectedFailedIndices.every((index) => failedIndices.includes(index))
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 20: 翻译状态流转一致性
   * Validates: Requirements 4.4, 4.5
   *
   * 标记当前图片为失败应正确更新状态
   */
  it('标记当前图片为失败一致性', () => {
    fc.assert(
      fc.property(
        validImageDataArb,
        fc.string({ minLength: 1, maxLength: 100 }),
        fc.integer({ min: 0, max: 5 }),
        (imageData, errorMessage, imageCount) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())

          const store = useImageStore()

          // 添加多张图片
          for (let i = 0; i <= imageCount; i++) {
            store.addImage(`${imageData.originalDataURL}${i}`, `${imageData.fileName}${i}`)
          }

          // 设置当前图片索引
          const targetIndex = Math.min(imageCount, store.images.length - 1)
          store.setCurrentImageIndex(targetIndex)

          // 标记当前图片为失败
          store.markCurrentAsFailed(errorMessage)

          // 验证状态已更新
          const currentImage = store.images[targetIndex]
          return (
            currentImage?.translationStatus === 'failed' &&
            currentImage?.translationFailed === true &&
            currentImage?.errorMessage === errorMessage
          )
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 20: 翻译状态流转一致性
   * Validates: Requirements 4.4, 4.5
   *
   * 翻译状态更新应保持其他图片数据不变
   */
  it('翻译状态更新不影响其他图片数据', () => {
    fc.assert(
      fc.property(
        fc.array(validImageDataArb, { minLength: 2, maxLength: 5 }),
        validTranslationStatusArb,
        fc.integer({ min: 0, max: 4 }),
        (imagesData, newStatus, targetIndexRaw) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())

          const store = useImageStore()

          // 添加多张图片
          imagesData.forEach((img) => {
            store.addImage(img.originalDataURL, img.fileName)
          })

          // 确保目标索引有效
          const targetIndex = targetIndexRaw % store.images.length

          // 记录其他图片的原始数据
          const otherImagesData = store.images
            .filter((_, index) => index !== targetIndex)
            .map((img) => ({
              originalDataURL: img.originalDataURL,
              fileName: img.fileName,
              translationStatus: img.translationStatus
            }))

          // 更新目标图片状态
          store.setTranslationStatus(targetIndex, newStatus)

          // 验证其他图片数据未变
          const currentOtherImages = store.images.filter((_, index) => index !== targetIndex)

          return otherImagesData.every((original, index) => {
            const current = currentOtherImages[index]
            return (
              current?.originalDataURL === original.originalDataURL &&
              current?.fileName === original.fileName &&
              current?.translationStatus === original.translationStatus
            )
          })
        }
      ),
      { numRuns: 100 }
    )
  })
})
