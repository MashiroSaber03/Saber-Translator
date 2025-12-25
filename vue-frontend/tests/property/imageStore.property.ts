/**
 * 图片状态管理属性测试
 * 使用 fast-check 进行属性基测试，验证图片上传状态的一致性
 *
 * Feature: vue-frontend-migration, Property 3: 图片上传状态一致性
 * Validates: Requirements 4.1
 */
import { describe, it, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'

describe('图片状态管理属性测试', () => {
  // 每个测试前重置 Pinia
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * 生成有效的文件名
   */
  const validFileNameArb = fc
    .stringOf(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789_-'.split('')), {
      minLength: 1,
      maxLength: 50
    })
    .map((name) => `${name}.png`)

  /**
   * 生成有效的 Base64 图片数据（简化版，用于测试）
   */
  const validBase64Arb = fc
    .stringOf(fc.constantFrom(...'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'.split('')), {
      minLength: 10,
      maxLength: 100
    })
    .map((data) => `data:image/png;base64,${data}`)

  /**
   * Feature: vue-frontend-migration, Property 3: 图片上传状态一致性
   * Validates: Requirements 4.1
   *
   * 对于任意有效图片文件，上传后 imageStore 中的图片数量应当增加相应数量
   */
  it('添加图片后图片数量应该正确增加', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            fileName: validFileNameArb,
            originalDataURL: validBase64Arb
          }),
          { minLength: 1, maxLength: 10 }
        ),
        (imageList) => {
          // 每次迭代重新创建 Pinia 实例以确保隔离
          setActivePinia(createPinia())
          const store = useImageStore()
          const initialCount = store.imageCount

          // 批量添加图片
          store.addImages(imageList)

          // 验证图片数量正确增加
          return store.imageCount === initialCount + imageList.length
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 3: 图片上传状态一致性
   * Validates: Requirements 4.1
   *
   * 对于任意有效图片，每张图片都应该有唯一的 ID
   */
  it('每张添加的图片都应该有唯一的ID', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            fileName: validFileNameArb,
            originalDataURL: validBase64Arb
          }),
          { minLength: 2, maxLength: 20 }
        ),
        (imageList) => {
          // 每次迭代重新创建 Pinia 实例以确保隔离
          setActivePinia(createPinia())
          const store = useImageStore()

          // 批量添加图片
          const addedImages = store.addImages(imageList)

          // 收集所有 ID
          const ids = addedImages.map((img) => img.id)

          // 验证所有 ID 都是唯一的
          const uniqueIds = new Set(ids)
          return uniqueIds.size === ids.length
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 3: 图片上传状态一致性
   * Validates: Requirements 4.1
   *
   * 对于任意有效图片，添加后应该有正确的初始状态
   */
  it('添加的图片应该有正确的初始状态', () => {
    fc.assert(
      fc.property(validFileNameArb, validBase64Arb, (fileName, originalDataURL) => {
        // 每次迭代重新创建 Pinia 实例以确保隔离
        setActivePinia(createPinia())
        const store = useImageStore()

        // 添加单张图片
        const addedImage = store.addImage(fileName, originalDataURL)

        // 验证初始状态
        return (
          addedImage.fileName === fileName &&
          addedImage.originalDataURL === originalDataURL &&
          addedImage.translatedDataURL === null &&
          addedImage.cleanImageData === null &&
          addedImage.bubbleStates === null &&
          addedImage.translationStatus === 'pending' &&
          addedImage.translationFailed === false &&
          addedImage.hasUnsavedChanges === false &&
          typeof addedImage.id === 'string' &&
          addedImage.id.length > 0
        )
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 3: 图片上传状态一致性
   * Validates: Requirements 4.1
   *
   * 添加第一张图片时，应该自动设置为当前图片
   */
  it('添加第一张图片时应该自动设置为当前图片', () => {
    fc.assert(
      fc.property(validFileNameArb, validBase64Arb, (fileName, originalDataURL) => {
        // 每次迭代重新创建 Pinia 实例以确保隔离
        setActivePinia(createPinia())
        const store = useImageStore()

        // 确保初始状态没有图片
        const initialIndex = store.currentImageIndex as number
        if (initialIndex !== -1) return false
        if (store.currentImage !== null) return false

        // 添加第一张图片
        store.addImage(fileName, originalDataURL)

        // 验证当前索引被设置为 0
        const newIndex = store.currentImageIndex as number
        return newIndex === 0 && store.currentImage !== null
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 3: 图片上传状态一致性
   * Validates: Requirements 4.1
   *
   * 删除图片后，图片数量应该正确减少
   */
  it('删除图片后图片数量应该正确减少', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            fileName: validFileNameArb,
            originalDataURL: validBase64Arb
          }),
          { minLength: 2, maxLength: 10 }
        ),
        fc.nat(),
        (imageList, deleteIndexSeed) => {
          // 每次迭代重新创建 Pinia 实例以确保隔离
          setActivePinia(createPinia())
          const store = useImageStore()

          // 批量添加图片
          store.addImages(imageList)
          const countAfterAdd = store.imageCount

          // 计算有效的删除索引
          const deleteIndex = deleteIndexSeed % countAfterAdd

          // 删除图片
          const result = store.deleteImage(deleteIndex)

          // 验证删除成功且数量正确减少
          return result === true && store.imageCount === countAfterAdd - 1
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 3: 图片上传状态一致性
   * Validates: Requirements 4.1
   *
   * 清除所有图片后，状态应该重置
   */
  it('清除所有图片后状态应该重置', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            fileName: validFileNameArb,
            originalDataURL: validBase64Arb
          }),
          { minLength: 1, maxLength: 10 }
        ),
        (imageList) => {
          // 每次迭代重新创建 Pinia 实例以确保隔离
          setActivePinia(createPinia())
          const store = useImageStore()

          // 批量添加图片
          store.addImages(imageList)

          // 清除所有图片
          store.clearImages()

          // 验证状态已重置
          return (
            store.imageCount === 0 &&
            store.currentImageIndex === -1 &&
            store.currentImage === null &&
            store.hasImages === false
          )
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 3: 图片上传状态一致性
   * Validates: Requirements 4.1
   *
   * 图片索引切换应该在有效范围内
   */
  it('图片索引切换应该在有效范围内', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            fileName: validFileNameArb,
            originalDataURL: validBase64Arb
          }),
          { minLength: 3, maxLength: 10 }
        ),
        fc.nat(),
        (imageList, targetIndexSeed) => {
          // 每次迭代重新创建 Pinia 实例以确保隔离
          setActivePinia(createPinia())
          const store = useImageStore()

          // 批量添加图片
          store.addImages(imageList)

          // 计算有效的目标索引
          const targetIndex = targetIndexSeed % store.imageCount

          // 设置当前索引
          store.setCurrentImageIndex(targetIndex)

          // 验证索引被正确设置
          // 注意：由于 addImages 会自动设置第一张为当前图片，
          // 我们需要验证 setCurrentImageIndex 能正确切换到目标索引
          return (
            store.currentImageIndex === targetIndex &&
            store.currentImage !== null
          )
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 3: 图片上传状态一致性
   * Validates: Requirements 4.1
   *
   * 上一张/下一张导航应该正确工作
   */
  it('上一张/下一张导航应该正确工作', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            fileName: validFileNameArb,
            originalDataURL: validBase64Arb
          }),
          { minLength: 3, maxLength: 10 }
        ),
        (imageList) => {
          // 每次迭代重新创建 Pinia 实例以确保隔离
          setActivePinia(createPinia())
          const store = useImageStore()

          // 批量添加图片
          store.addImages(imageList)

          // 初始在第一张，不能往前
          if (store.canGoPrevious !== false) return false
          if (store.canGoNext !== true) return false

          // 切换到下一张
          const nextResult = store.goToNext()
          if (!nextResult) return false
          const indexAfterNext = store.currentImageIndex as number
          if (indexAfterNext !== 1) return false

          // 现在可以往前了
          if (!store.canGoPrevious) return false

          // 切换到上一张
          const prevResult = store.goToPrevious()
          if (!prevResult) return false
          const indexAfterPrev = store.currentImageIndex as number
          if (indexAfterPrev !== 0) return false

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 3: 图片上传状态一致性
   * Validates: Requirements 4.1
   *
   * 更新图片属性应该正确反映
   */
  it('更新图片属性应该正确反映', () => {
    fc.assert(
      fc.property(
        validFileNameArb,
        validBase64Arb,
        fc.integer({ min: 10, max: 100 }),
        fc.hexaString({ minLength: 6, maxLength: 6 }),
        (fileName, originalDataURL, fontSize, colorHex) => {
          // 每次迭代重新创建 Pinia 实例以确保隔离
          setActivePinia(createPinia())
          const store = useImageStore()

          // 添加图片
          store.addImage(fileName, originalDataURL)

          // 更新属性
          const textColor = `#${colorHex}`
          store.updateCurrentImage({
            fontSize,
            textColor
          })

          // 验证属性已更新
          return (
            store.currentImage !== null &&
            store.currentImage.fontSize === fontSize &&
            store.currentImage.textColor === textColor
          )
        }
      ),
      { numRuns: 100 }
    )
  })
})
