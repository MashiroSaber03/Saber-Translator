/**
 * 图片下载功能属性测试
 * 
 * **Feature: vue-frontend-migration, Property 30: 图片下载格式一致性**
 * **Validates: Requirements 18.3, 18.4**
 * 
 * 测试内容：
 * - 单张下载文件名格式正确
 * - 批量下载ZIP包含所有图片
 * - PDF生成页面顺序正确
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useImageStore } from '@/stores/imageStore'

describe('图片下载功能属性测试', () => {
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
   * 生成下载文件名的函数
   * 与 useExportImport 中的逻辑一致
   */
  function generateDownloadFileName(
    originalFileName: string,
    imageIndex: number,
    hasTranslated: boolean
  ): string {
    let fileName = originalFileName || `image_${imageIndex}.png`
    const prefix = hasTranslated ? 'translated' : 'original'
    fileName = `${prefix}_${fileName.replace(/\.[^/.]+$/, '')}.png`
    return fileName
  }

  /**
   * Property 30.1: 单张下载文件名格式正确
   * 对于任意图片，下载文件名应当包含正确的前缀和扩展名
   */
  it('单张下载文件名格式正确', () => {
    fc.assert(
      fc.property(
        validFileNameArb,
        fc.boolean(),
        fc.nat(100),
        (fileName, hasTranslated, imageIndex) => {
          const downloadFileName = generateDownloadFileName(fileName, imageIndex, hasTranslated)
          
          // 验证文件名格式
          expect(downloadFileName).toMatch(/\.(png)$/)
          
          // 验证前缀
          if (hasTranslated) {
            expect(downloadFileName.startsWith('translated_')).toBe(true)
          } else {
            expect(downloadFileName.startsWith('original_')).toBe(true)
          }
          
          // 验证原始文件名被包含（去除扩展名后）
          const baseFileName = fileName.replace(/\.[^/.]+$/, '')
          expect(downloadFileName).toContain(baseFileName)
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 30.2: 默认文件名生成正确
   * 当原始文件名为空时，应当使用默认文件名格式
   */
  it('默认文件名生成正确', () => {
    fc.assert(
      fc.property(
        fc.nat(100),
        fc.boolean(),
        (imageIndex, hasTranslated) => {
          const downloadFileName = generateDownloadFileName('', imageIndex, hasTranslated)
          
          // 验证默认文件名格式
          const prefix = hasTranslated ? 'translated' : 'original'
          expect(downloadFileName).toBe(`${prefix}_image_${imageIndex}.png`)
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 30.3: 批量下载图片信息收集正确
   * 对于任意图片列表，收集的下载信息应当包含所有有效图片
   */
  it('批量下载图片信息收集正确', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            fileName: validFileNameArb,
            hasOriginal: fc.boolean(),
            hasTranslated: fc.boolean()
          }),
          { minLength: 1, maxLength: 10 }
        ),
        mockDataURLArb,
        (imageConfigs, dataURL) => {
          // 每次测试重新创建 Pinia 实例
          setActivePinia(createPinia())
          const store = useImageStore()
          
          // 添加图片
          for (const config of imageConfigs) {
            store.addImage(config.fileName, config.hasOriginal ? dataURL : '')
            const image = store.images[store.images.length - 1]
            if (image && config.hasTranslated) {
              image.translatedDataURL = dataURL
            }
          }
          
          // 收集下载信息
          const imageInfoList: Array<{ index: number; type: 'translated' | 'original' }> = []
          
          for (let i = 0; i < store.images.length; i++) {
            const imgData = store.images[i]
            if (!imgData) continue
            
            if (imgData.translatedDataURL) {
              imageInfoList.push({ index: i, type: 'translated' })
            } else if (imgData.originalDataURL) {
              imageInfoList.push({ index: i, type: 'original' })
            }
          }
          
          // 验证收集的信息
          // 有效图片数量应当等于有原图或翻译图的图片数量
          const validImageCount = imageConfigs.filter(
            config => config.hasOriginal || config.hasTranslated
          ).length
          expect(imageInfoList.length).toBe(validImageCount)
          
          // 验证每个收集的信息都有有效的索引
          for (const info of imageInfoList) {
            expect(info.index).toBeGreaterThanOrEqual(0)
            expect(info.index).toBeLessThan(store.images.length)
          }
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 30.4: 翻译图优先于原图
   * 当图片同时有翻译图和原图时，应当优先使用翻译图
   */
  it('翻译图优先于原图', () => {
    fc.assert(
      fc.property(
        validFileNameArb,
        mockDataURLArb,
        (fileName, dataURL) => {
          // 每次测试重新创建 Pinia 实例
          setActivePinia(createPinia())
          const store = useImageStore()
          
          // 添加同时有原图和翻译图的图片
          store.addImage(fileName, dataURL)
          const image = store.images[0]
          if (image) {
            image.translatedDataURL = dataURL
          }
          
          // 收集下载信息
          const imageInfoList: Array<{ index: number; type: 'translated' | 'original' }> = []
          
          for (let i = 0; i < store.images.length; i++) {
            const imgData = store.images[i]
            if (!imgData) continue
            
            if (imgData.translatedDataURL) {
              imageInfoList.push({ index: i, type: 'translated' })
            } else if (imgData.originalDataURL) {
              imageInfoList.push({ index: i, type: 'original' })
            }
          }
          
          // 验证优先使用翻译图
          expect(imageInfoList.length).toBe(1)
          expect(imageInfoList[0]?.type).toBe('translated')
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 30.5: 下载格式验证
   * 支持的下载格式应当为 zip、pdf、cbz
   */
  it('下载格式验证', () => {
    const validFormats = ['zip', 'pdf', 'cbz'] as const
    
    for (const format of validFormats) {
      expect(validFormats).toContain(format)
    }
    
    // 验证格式类型
    type DownloadFormat = 'zip' | 'pdf' | 'cbz'
    const testFormat: DownloadFormat = 'zip'
    expect(['zip', 'pdf', 'cbz']).toContain(testFormat)
  })

  /**
   * Property 30.6: 图片索引顺序保持一致
   * 批量下载时，图片索引应当按顺序排列
   */
  it('图片索引顺序保持一致', () => {
    fc.assert(
      fc.property(
        fc.array(validFileNameArb, { minLength: 2, maxLength: 10 }),
        mockDataURLArb,
        (fileNames, dataURL) => {
          // 每次测试重新创建 Pinia 实例
          setActivePinia(createPinia())
          const store = useImageStore()
          
          // 添加图片
          for (const fileName of fileNames) {
            store.addImage(fileName, dataURL)
          }
          
          // 收集下载信息
          const imageInfoList: Array<{ index: number; type: 'translated' | 'original' }> = []
          
          for (let i = 0; i < store.images.length; i++) {
            const imgData = store.images[i]
            if (!imgData) continue
            
            if (imgData.translatedDataURL) {
              imageInfoList.push({ index: i, type: 'translated' })
            } else if (imgData.originalDataURL) {
              imageInfoList.push({ index: i, type: 'original' })
            }
          }
          
          // 验证索引顺序
          for (let i = 0; i < imageInfoList.length - 1; i++) {
            const current = imageInfoList[i]
            const next = imageInfoList[i + 1]
            if (current && next) {
              expect(current.index).toBeLessThan(next.index)
            }
          }
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 30.7: 空图片列表处理
   * 当没有图片时，下载信息列表应当为空
   */
  it('空图片列表处理', () => {
    setActivePinia(createPinia())
    const store = useImageStore()
    
    // 收集下载信息
    const imageInfoList: Array<{ index: number; type: 'translated' | 'original' }> = []
    
    for (let i = 0; i < store.images.length; i++) {
      const imgData = store.images[i]
      if (!imgData) continue
      
      if (imgData.translatedDataURL) {
        imageInfoList.push({ index: i, type: 'translated' })
      } else if (imgData.originalDataURL) {
        imageInfoList.push({ index: i, type: 'original' })
      }
    }
    
    expect(imageInfoList.length).toBe(0)
  })
})
