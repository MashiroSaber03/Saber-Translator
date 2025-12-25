/**
 * 图片上传处理属性测试
 * 
 * **Feature: vue-frontend-migration, Property 26: 图片上传处理一致性**
 * **Validates: Requirements 4.1**
 * 
 * 测试内容：
 * - 多图片上传后列表正确更新
 * - 文件名自然排序正确性
 * - 重复文件名处理
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useImageStore } from '@/stores/imageStore'

describe('图片上传处理属性测试', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * 生成有效的文件名
   */
  const validFileNameArb = fc.stringOf(
    fc.constantFrom(...'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'),
    { minLength: 1, maxLength: 20 }
  ).map(name => `${name}.png`)

  /**
   * 生成模拟的 Base64 图片数据
   */
  const mockDataURLArb = fc.constant('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')

  /**
   * Property 26.1: 多图片上传后列表正确更新
   * 对于任意数量的图片上传，上传后 imageStore 中的图片数量应当增加相应数量
   */
  it('多图片上传后列表长度正确增加', () => {
    fc.assert(
      fc.property(
        fc.array(validFileNameArb, { minLength: 1, maxLength: 10 }),
        mockDataURLArb,
        (fileNames, dataURL) => {
          const store = useImageStore()
          const initialCount = store.imageCount
          
          // 添加图片
          for (const fileName of fileNames) {
            store.addImage(fileName, dataURL)
          }
          
          // 验证图片数量正确增加
          expect(store.imageCount).toBe(initialCount + fileNames.length)
          
          // 验证每张图片都有唯一 ID
          const ids = store.images.map(img => img.id)
          const uniqueIds = new Set(ids)
          expect(uniqueIds.size).toBe(ids.length)
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 26.2: 图片初始状态正确
   * 对于任意上传的图片，初始状态应当为 pending，且 translationFailed 为 false
   */
  it('上传图片的初始状态正确', () => {
    fc.assert(
      fc.property(
        validFileNameArb,
        mockDataURLArb,
        (fileName, dataURL) => {
          const store = useImageStore()
          
          store.addImage(fileName, dataURL)
          
          const addedImage = store.images[store.images.length - 1]
          
          // 验证初始状态
          expect(addedImage).toBeDefined()
          expect(addedImage?.translationStatus).toBe('pending')
          expect(addedImage?.translationFailed).toBe(false)
          expect(addedImage?.originalDataURL).toBe(dataURL)
          expect(addedImage?.fileName).toBe(fileName)
          expect(addedImage?.translatedDataURL).toBeNull()
          expect(addedImage?.hasUnsavedChanges).toBe(false)
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 26.3: 文件名自然排序正确性
   * 测试自然排序算法的正确性
   */
  it('文件名自然排序正确', () => {
    /**
     * 自然排序函数（与 ImageUpload 组件中的实现一致）
     */
    function naturalSort(a: string, b: string): number {
      return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })
    }

    fc.assert(
      fc.property(
        fc.array(fc.string({ minLength: 1, maxLength: 20 }), { minLength: 2, maxLength: 20 }),
        (names) => {
          const sorted = [...names].sort(naturalSort)
          
          // 验证排序后的数组是有序的
          for (let i = 0; i < sorted.length - 1; i++) {
            const current = sorted[i]
            const next = sorted[i + 1]
            if (current && next) {
              expect(naturalSort(current, next)).toBeLessThanOrEqual(0)
            }
          }
          
          // 验证排序后的数组包含所有原始元素
          expect(sorted.length).toBe(names.length)
          for (const name of names) {
            expect(sorted).toContain(name)
          }
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 26.4: 数字文件名自然排序
   * 测试包含数字的文件名按数字大小排序
   */
  it('数字文件名按数字大小排序', () => {
    function naturalSort(a: string, b: string): number {
      return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })
    }

    // 测试特定的数字排序场景
    const testCases = [
      { input: ['page_2.png', 'page_10.png', 'page_1.png'], expected: ['page_1.png', 'page_2.png', 'page_10.png'] },
      { input: ['img10.jpg', 'img2.jpg', 'img1.jpg'], expected: ['img1.jpg', 'img2.jpg', 'img10.jpg'] },
      { input: ['001.png', '010.png', '002.png'], expected: ['001.png', '002.png', '010.png'] },
    ]

    for (const { input, expected } of testCases) {
      const sorted = [...input].sort(naturalSort)
      expect(sorted).toEqual(expected)
    }
  })

  /**
   * Property 26.5: 当前图片索引在有效范围内
   * 添加图片后，当前图片索引应当在有效范围内
   */
  it('当前图片索引在有效范围内', () => {
    fc.assert(
      fc.property(
        fc.array(validFileNameArb, { minLength: 1, maxLength: 10 }),
        mockDataURLArb,
        (fileNames, dataURL) => {
          const store = useImageStore()
          
          // 添加图片
          for (const fileName of fileNames) {
            store.addImage(fileName, dataURL)
          }
          
          // 验证当前索引在有效范围内
          if (store.imageCount > 0) {
            expect(store.currentImageIndex).toBeGreaterThanOrEqual(0)
            expect(store.currentImageIndex).toBeLessThan(store.imageCount)
          }
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 26.6: 删除图片后列表正确更新
   * 删除图片后，列表长度应当减少，且当前索引保持有效
   */
  it('删除图片后列表正确更新', () => {
    fc.assert(
      fc.property(
        fc.array(validFileNameArb, { minLength: 2, maxLength: 10 }),
        mockDataURLArb,
        fc.nat(),
        (fileNames, dataURL, deleteIndexSeed) => {
          const store = useImageStore()
          
          // 添加图片
          for (const fileName of fileNames) {
            store.addImage(fileName, dataURL)
          }
          
          const countBeforeDelete = store.imageCount
          const deleteIndex = deleteIndexSeed % countBeforeDelete
          
          // 设置当前索引并删除
          store.setCurrentImageIndex(deleteIndex)
          store.deleteCurrentImage()
          
          // 验证删除后的状态
          expect(store.imageCount).toBe(countBeforeDelete - 1)
          
          // 验证当前索引仍然有效
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

  /**
   * Property 26.7: 清除所有图片后状态重置
   * 清除所有图片后，列表应当为空，当前索引应当为 -1
   */
  it('清除所有图片后状态重置', () => {
    fc.assert(
      fc.property(
        fc.array(validFileNameArb, { minLength: 1, maxLength: 10 }),
        mockDataURLArb,
        (fileNames, dataURL) => {
          const store = useImageStore()
          
          // 添加图片
          for (const fileName of fileNames) {
            store.addImage(fileName, dataURL)
          }
          
          // 清除所有图片
          store.clearImages()
          
          // 验证状态重置
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
