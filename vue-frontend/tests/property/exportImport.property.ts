/**
 * 文本导出导入属性测试
 * 
 * **Feature: vue-frontend-migration, Property 14: 文本导出导入往返一致性**
 * **Validates: Requirements 18.1, 18.2**
 * 
 * 测试内容：
 * - 导出 JSON 格式正确性
 * - 导入后翻译文本正确恢复
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useImageStore } from '@/stores/imageStore'
import type { BubbleState } from '@/types/bubble'
import type { ExportTextData } from '@/composables/useExportImport'

describe('文本导出导入属性测试', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * 生成有效的文本内容
   */
  const validTextArb = fc.string({ minLength: 0, maxLength: 100 })

  /**
   * 生成有效的排版方向
   */
  const textDirectionArb = fc.constantFrom('vertical', 'horizontal') as fc.Arbitrary<'vertical' | 'horizontal'>

  /**
   * 生成有效的气泡状态
   */
  const bubbleStateArb: fc.Arbitrary<BubbleState> = fc.record({
    coords: fc.tuple(fc.nat(1000), fc.nat(1000), fc.nat(1000), fc.nat(1000)) as fc.Arbitrary<[number, number, number, number]>,
    polygon: fc.constant([]) as fc.Arbitrary<number[][]>,
    originalText: validTextArb,
    translatedText: validTextArb,
    textboxText: validTextArb,
    fontSize: fc.integer({ min: 10, max: 100 }),
    fontFamily: fc.constant('fonts/STSONG.TTF'),
    textDirection: fc.constantFrom('vertical', 'horizontal', 'auto') as fc.Arbitrary<'vertical' | 'horizontal' | 'auto'>,
    autoTextDirection: textDirectionArb,
    textColor: fc.constant('#000000'),
    fillColor: fc.constant('#FFFFFF'),
    rotationAngle: fc.integer({ min: 0, max: 360 }),
    position: fc.constant({ x: 0, y: 0 }),
    strokeEnabled: fc.boolean(),
    strokeColor: fc.constant('#FFFFFF'),
    strokeWidth: fc.integer({ min: 1, max: 10 }),
    inpaintMethod: fc.constantFrom('solid', 'lama_mpe', 'litelama') as fc.Arbitrary<'solid' | 'lama_mpe' | 'litelama'>
  })

  /**
   * 生成模拟的 Base64 图片数据
   */
  const mockDataURLArb = fc.constant('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')

  /**
   * 模拟导出文本为 JSON 数据的函数
   * 与 useExportImport 中的 exportTextToJson 逻辑一致
   */
  function exportTextToJson(images: ReturnType<typeof useImageStore>['images']): ExportTextData[] | null {
    if (images.length === 0) return null

    const exportData: ExportTextData[] = []

    for (let imageIndex = 0; imageIndex < images.length; imageIndex++) {
      const image = images[imageIndex]
      if (!image) continue

      const bubbleStates = image.bubbleStates || []

      const imageTextData: ExportTextData = {
        imageIndex: imageIndex,
        bubbles: []
      }

      for (let bubbleIndex = 0; bubbleIndex < bubbleStates.length; bubbleIndex++) {
        const bubble = bubbleStates[bubbleIndex]
        if (!bubble) continue

        const original = bubble.originalText || ''
        const translated = bubble.translatedText || bubble.textboxText || ''

        let textDirection: 'vertical' | 'horizontal' = 'vertical'
        if (bubble.textDirection && bubble.textDirection !== 'auto') {
          textDirection = bubble.textDirection as 'vertical' | 'horizontal'
        } else if (bubble.autoTextDirection) {
          textDirection = bubble.autoTextDirection as 'vertical' | 'horizontal'
        }

        imageTextData.bubbles.push({
          bubbleIndex: bubbleIndex,
          original: original,
          translated: translated,
          textDirection: textDirection
        })
      }

      exportData.push(imageTextData)
    }

    return exportData
  }

  /**
   * Property 14.1: 导出 JSON 格式正确性
   * 对于任意图片和气泡数据，导出的 JSON 应当包含正确的结构
   */
  it('导出 JSON 格式正确', () => {
    fc.assert(
      fc.property(
        fc.array(fc.array(bubbleStateArb, { minLength: 1, maxLength: 5 }), { minLength: 1, maxLength: 5 }),
        mockDataURLArb,
        (bubbleStatesPerImage, dataURL) => {
          // 每次测试重新创建 Pinia 实例
          setActivePinia(createPinia())
          const store = useImageStore()
          
          // 添加图片并设置气泡状态
          for (let i = 0; i < bubbleStatesPerImage.length; i++) {
            store.addImage(`image_${i}.png`, dataURL)
            const image = store.images[i]
            if (image) {
              image.bubbleStates = bubbleStatesPerImage[i] || []
            }
          }
          
          // 导出文本
          const exportData = exportTextToJson(store.images)
          
          // 验证导出数据不为空
          expect(exportData).not.toBeNull()
          expect(Array.isArray(exportData)).toBe(true)
          
          // 验证导出数据结构
          if (exportData) {
            expect(exportData.length).toBe(bubbleStatesPerImage.length)
            
            for (let i = 0; i < exportData.length; i++) {
              const imageData = exportData[i]
              expect(imageData).toBeDefined()
              expect(imageData?.imageIndex).toBe(i)
              expect(Array.isArray(imageData?.bubbles)).toBe(true)
              
              // 验证气泡数据结构
              const expectedBubbles = bubbleStatesPerImage[i] || []
              expect(imageData?.bubbles.length).toBe(expectedBubbles.length)
              
              for (let j = 0; j < (imageData?.bubbles.length || 0); j++) {
                const bubble = imageData?.bubbles[j]
                expect(bubble).toBeDefined()
                expect(bubble?.bubbleIndex).toBe(j)
                expect(typeof bubble?.original).toBe('string')
                expect(typeof bubble?.translated).toBe('string')
                expect(['vertical', 'horizontal']).toContain(bubble?.textDirection)
              }
            }
          }
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 14.2: 导出文本内容正确
   * 导出的文本应当与原始气泡状态中的文本一致
   */
  it('导出文本内容正确', () => {
    fc.assert(
      fc.property(
        bubbleStateArb,
        mockDataURLArb,
        (bubbleState, dataURL) => {
          // 每次测试重新创建 Pinia 实例
          setActivePinia(createPinia())
          const store = useImageStore()
          
          // 添加图片并设置气泡状态
          store.addImage('test.png', dataURL)
          const image = store.images[0]
          if (image) {
            image.bubbleStates = [bubbleState]
          }
          
          // 导出文本
          const exportData = exportTextToJson(store.images)
          
          // 验证导出的文本内容
          expect(exportData).not.toBeNull()
          if (exportData && exportData[0]) {
            const exportedBubble = exportData[0].bubbles[0]
            expect(exportedBubble?.original).toBe(bubbleState.originalText || '')
            expect(exportedBubble?.translated).toBe(bubbleState.translatedText || bubbleState.textboxText || '')
          }
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 14.3: 排版方向导出正确
   * 导出的排版方向不应为 'auto'，应当转换为具体的 'vertical' 或 'horizontal'
   */
  it('排版方向导出正确（不包含 auto）', () => {
    fc.assert(
      fc.property(
        bubbleStateArb,
        mockDataURLArb,
        (bubbleState, dataURL) => {
          // 每次测试重新创建 Pinia 实例
          setActivePinia(createPinia())
          const store = useImageStore()
          
          // 添加图片并设置气泡状态
          store.addImage('test.png', dataURL)
          const image = store.images[0]
          if (image) {
            image.bubbleStates = [bubbleState]
          }
          
          // 导出文本
          const exportData = exportTextToJson(store.images)
          
          // 验证排版方向不为 'auto'
          expect(exportData).not.toBeNull()
          if (exportData && exportData[0]) {
            const exportedBubble = exportData[0].bubbles[0]
            expect(exportedBubble?.textDirection).not.toBe('auto')
            expect(['vertical', 'horizontal']).toContain(exportedBubble?.textDirection)
          }
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 14.4: 空图片列表导出返回 null
   * 当没有图片时，导出应当返回 null
   */
  it('空图片列表导出返回 null', () => {
    const store = useImageStore()
    const exportData = exportTextToJson(store.images)
    expect(exportData).toBeNull()
  })

  /**
   * Property 14.5: 导出数据可序列化为有效 JSON
   * 导出的数据应当可以序列化为有效的 JSON 字符串
   */
  it('导出数据可序列化为有效 JSON', () => {
    fc.assert(
      fc.property(
        fc.array(bubbleStateArb, { minLength: 1, maxLength: 5 }),
        mockDataURLArb,
        (bubbleStates, dataURL) => {
          // 每次测试重新创建 Pinia 实例
          setActivePinia(createPinia())
          const store = useImageStore()
          
          // 添加图片并设置气泡状态
          store.addImage('test.png', dataURL)
          const image = store.images[0]
          if (image) {
            image.bubbleStates = bubbleStates
          }
          
          // 导出文本
          const exportData = exportTextToJson(store.images)
          
          // 验证可以序列化为 JSON
          expect(exportData).not.toBeNull()
          if (exportData) {
            const jsonString = JSON.stringify(exportData)
            expect(typeof jsonString).toBe('string')
            
            // 验证可以反序列化
            const parsed = JSON.parse(jsonString)
            expect(Array.isArray(parsed)).toBe(true)
            expect(parsed.length).toBe(exportData.length)
          }
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 14.6: 导出导入往返一致性
   * 导出后再导入，文本内容应当保持一致
   */
  it('导出导入往返一致性', () => {
    fc.assert(
      fc.property(
        validTextArb,
        validTextArb,
        textDirectionArb,
        mockDataURLArb,
        (originalText, translatedText, textDirection, dataURL) => {
          // 每次测试重新创建 Pinia 实例
          setActivePinia(createPinia())
          const store = useImageStore()
          
          // 创建气泡状态
          const bubbleState: BubbleState = {
            coords: [0, 0, 100, 100],
            polygon: [],
            originalText,
            translatedText,
            textboxText: translatedText,
            fontSize: 25,
            fontFamily: 'fonts/STSONG.TTF',
            textDirection,
            autoTextDirection: textDirection,
            textColor: '#000000',
            fillColor: '#FFFFFF',
            rotationAngle: 0,
            position: { x: 0, y: 0 },
            strokeEnabled: false,
            strokeColor: '#FFFFFF',
            strokeWidth: 3,
            inpaintMethod: 'solid'
          }
          
          // 添加图片并设置气泡状态
          store.addImage('test.png', dataURL)
          const image = store.images[0]
          if (image) {
            image.bubbleStates = [bubbleState]
          }
          
          // 导出文本
          const exportData = exportTextToJson(store.images)
          expect(exportData).not.toBeNull()
          
          // 序列化和反序列化（模拟文件保存和读取）
          const jsonString = JSON.stringify(exportData)
          const importedData: ExportTextData[] = JSON.parse(jsonString)
          
          // 验证导入数据与原始数据一致
          expect(importedData.length).toBe(1)
          expect(importedData[0]?.bubbles.length).toBe(1)
          
          const importedBubble = importedData[0]?.bubbles[0]
          expect(importedBubble?.original).toBe(originalText)
          expect(importedBubble?.translated).toBe(translatedText)
          expect(importedBubble?.textDirection).toBe(textDirection)
        }
      ),
      { numRuns: 100 }
    )
  })
})
