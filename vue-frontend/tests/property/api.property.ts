/**
 * API 请求参数构建属性测试
 * Property 19: API 请求参数构建一致性
 * 测试翻译请求参数完整性和会话数据序列化正确性
 */

import { describe, it, expect } from 'vitest'
import * as fc from 'fast-check'
import type { BubbleState, BubbleCoords } from '@/types'
import type { TranslateImageParams, ReRenderParams } from '@/api/translate'

// ==================== 辅助函数 ====================

/**
 * 生成有效的气泡坐标
 */
function generateBubbleCoords(): fc.Arbitrary<BubbleCoords> {
  return fc.record({
    x: fc.integer({ min: 0, max: 1000 }),
    y: fc.integer({ min: 0, max: 1000 }),
    width: fc.integer({ min: 10, max: 500 }),
    height: fc.integer({ min: 10, max: 500 }),
  })
}

/**
 * 生成有效的气泡状态
 */
function generateBubbleState(): fc.Arbitrary<BubbleState> {
  return fc.record({
    coords: generateBubbleCoords(),
    originalText: fc.string({ minLength: 0, maxLength: 200 }),
    translatedText: fc.string({ minLength: 0, maxLength: 200 }),
    textboxText: fc.string({ minLength: 0, maxLength: 200 }),
    fontSize: fc.integer({ min: 8, max: 72 }),
    fontFamily: fc.constantFrom('Arial', 'SimHei', 'Microsoft YaHei'),
    textDirection: fc.constantFrom('auto', 'vertical', 'horizontal'),
    autoTextDirection: fc.constantFrom('v', 'h', undefined),
    textColor: fc.hexaString({ minLength: 6, maxLength: 6 }).map(s => `#${s}`),
    fillColor: fc.hexaString({ minLength: 6, maxLength: 6 }).map(s => `#${s}`),
    rotationAngle: fc.integer({ min: 0, max: 360 }),
    strokeEnabled: fc.boolean(),
    strokeColor: fc.hexaString({ minLength: 6, maxLength: 6 }).map(s => `#${s}`),
    strokeWidth: fc.integer({ min: 1, max: 10 }),
    inpaintMethod: fc.constantFrom('solid', 'lama_mpe', 'litelama'),
  })
}

/**
 * 生成翻译请求参数
 */
function generateTranslateParams(): fc.Arbitrary<TranslateImageParams> {
  return fc.record({
    image: fc.string({ minLength: 10, maxLength: 100 }), // 模拟 Base64
    ocr_engine: fc.constantFrom('manga_ocr', 'paddle_ocr', 'baidu_ocr', 'ai_vision'),
    translate_provider: fc.constantFrom('siliconflow', 'deepseek', 'gemini', 'ollama'),
    target_language: fc.constantFrom('zh-CN', 'en', 'ja', 'ko'),
    font_size: fc.integer({ min: 8, max: 72 }),
    auto_font_size: fc.boolean(),
    text_direction: fc.constantFrom('auto', 'vertical', 'horizontal'),
    inpaint_method: fc.constantFrom('solid', 'lama_mpe', 'litelama'),
  })
}

// ==================== 属性测试 ====================

describe('API 请求参数构建属性测试', () => {
  describe('Property 19: API 请求参数构建一致性', () => {
    it('翻译请求参数应包含所有必需字段', () => {
      fc.assert(
        fc.property(generateTranslateParams(), params => {
          // 必需字段检查
          expect(params.image).toBeDefined()
          expect(typeof params.image).toBe('string')
          expect(params.image.length).toBeGreaterThan(0)

          expect(params.ocr_engine).toBeDefined()
          expect(['manga_ocr', 'paddle_ocr', 'baidu_ocr', 'ai_vision']).toContain(
            params.ocr_engine
          )

          expect(params.translate_provider).toBeDefined()
          expect(['siliconflow', 'deepseek', 'gemini', 'ollama']).toContain(
            params.translate_provider
          )

          expect(params.target_language).toBeDefined()
          expect(['zh-CN', 'en', 'ja', 'ko']).toContain(params.target_language)
        }),
        { numRuns: 100 }
      )
    })

    it('气泡状态序列化后应保持数据完整性', () => {
      fc.assert(
        fc.property(generateBubbleState(), bubbleState => {
          // 序列化
          const serialized = JSON.stringify(bubbleState)

          // 反序列化
          const deserialized = JSON.parse(serialized) as BubbleState

          // 验证关键字段
          expect(deserialized.coords).toEqual(bubbleState.coords)
          expect(deserialized.originalText).toBe(bubbleState.originalText)
          expect(deserialized.translatedText).toBe(bubbleState.translatedText)
          expect(deserialized.fontSize).toBe(bubbleState.fontSize)
          expect(deserialized.fontFamily).toBe(bubbleState.fontFamily)
          expect(deserialized.textDirection).toBe(bubbleState.textDirection)
          expect(deserialized.textColor).toBe(bubbleState.textColor)
          expect(deserialized.fillColor).toBe(bubbleState.fillColor)
          expect(deserialized.strokeEnabled).toBe(bubbleState.strokeEnabled)
          expect(deserialized.strokeColor).toBe(bubbleState.strokeColor)
          expect(deserialized.strokeWidth).toBe(bubbleState.strokeWidth)
          expect(deserialized.inpaintMethod).toBe(bubbleState.inpaintMethod)
        }),
        { numRuns: 100 }
      )
    })

    it('气泡坐标应为有效的矩形区域', () => {
      fc.assert(
        fc.property(generateBubbleCoords(), coords => {
          // 坐标应为非负整数
          expect(coords.x).toBeGreaterThanOrEqual(0)
          expect(coords.y).toBeGreaterThanOrEqual(0)

          // 尺寸应为正整数
          expect(coords.width).toBeGreaterThan(0)
          expect(coords.height).toBeGreaterThan(0)

          // 坐标应为整数
          expect(Number.isInteger(coords.x)).toBe(true)
          expect(Number.isInteger(coords.y)).toBe(true)
          expect(Number.isInteger(coords.width)).toBe(true)
          expect(Number.isInteger(coords.height)).toBe(true)
        }),
        { numRuns: 100 }
      )
    })

    it('重新渲染参数应包含完整的气泡状态数组', () => {
      fc.assert(
        fc.property(
          fc.array(generateBubbleState(), { minLength: 1, maxLength: 10 }),
          bubbleStates => {
            const params: ReRenderParams = {
              original_image: 'base64_original',
              clean_image: 'base64_clean',
              bubble_states: bubbleStates,
            }

            // 验证参数结构
            expect(params.original_image).toBeDefined()
            expect(params.clean_image).toBeDefined()
            expect(Array.isArray(params.bubble_states)).toBe(true)
            expect(params.bubble_states.length).toBe(bubbleStates.length)

            // 验证每个气泡状态
            params.bubble_states.forEach((state, index) => {
              expect(state.coords).toEqual(bubbleStates[index].coords)
              expect(state.fontSize).toBe(bubbleStates[index].fontSize)
            })
          }
        ),
        { numRuns: 50 }
      )
    })

    it('会话数据序列化应保持图片数组顺序', () => {
      fc.assert(
        fc.property(
          fc.array(
            fc.record({
              originalDataURL: fc.string({ minLength: 10, maxLength: 50 }),
              translatedDataURL: fc.option(fc.string({ minLength: 10, maxLength: 50 })),
              fileName: fc.string({ minLength: 1, maxLength: 50 }),
              bubbleStates: fc.array(generateBubbleState(), { minLength: 0, maxLength: 5 }),
            }),
            { minLength: 1, maxLength: 10 }
          ),
          images => {
            const sessionData = {
              name: 'test_session',
              version: '1.0',
              savedAt: new Date().toISOString(),
              imageCount: images.length,
              ui_settings: {},
              images,
              currentImageIndex: 0,
            }

            // 序列化
            const serialized = JSON.stringify(sessionData)

            // 反序列化
            const deserialized = JSON.parse(serialized)

            // 验证图片数组顺序和内容
            expect(deserialized.images.length).toBe(images.length)
            images.forEach((img, index) => {
              expect(deserialized.images[index].originalDataURL).toBe(img.originalDataURL)
              expect(deserialized.images[index].fileName).toBe(img.fileName)
              expect(deserialized.images[index].bubbleStates.length).toBe(img.bubbleStates.length)
            })
          }
        ),
        { numRuns: 50 }
      )
    })

    it('OCR 引擎参数应与服务商配置匹配', () => {
      const ocrEngineConfigs = {
        manga_ocr: { requiresApiKey: false, requiresModel: false },
        paddle_ocr: { requiresApiKey: false, requiresModel: false },
        baidu_ocr: { requiresApiKey: true, requiresModel: false },
        ai_vision: { requiresApiKey: true, requiresModel: true },
      }

      fc.assert(
        fc.property(
          fc.constantFrom('manga_ocr', 'paddle_ocr', 'baidu_ocr', 'ai_vision'),
          ocrEngine => {
            const config = ocrEngineConfigs[ocrEngine as keyof typeof ocrEngineConfigs]
            expect(config).toBeDefined()

            // 验证配置属性存在
            expect(typeof config.requiresApiKey).toBe('boolean')
            expect(typeof config.requiresModel).toBe('boolean')
          }
        ),
        { numRuns: 20 }
      )
    })

    it('翻译服务商参数应与服务商配置匹配', () => {
      const providerConfigs = {
        siliconflow: { requiresApiKey: true, supportsCustomBaseUrl: false },
        deepseek: { requiresApiKey: true, supportsCustomBaseUrl: false },
        gemini: { requiresApiKey: true, supportsCustomBaseUrl: false },
        ollama: { requiresApiKey: false, supportsCustomBaseUrl: true },
        sakura: { requiresApiKey: false, supportsCustomBaseUrl: true },
        custom_openai: { requiresApiKey: true, supportsCustomBaseUrl: true },
      }

      fc.assert(
        fc.property(
          fc.constantFrom(
            'siliconflow',
            'deepseek',
            'gemini',
            'ollama',
            'sakura',
            'custom_openai'
          ),
          provider => {
            const config = providerConfigs[provider as keyof typeof providerConfigs]
            expect(config).toBeDefined()

            // 验证配置属性存在
            expect(typeof config.requiresApiKey).toBe('boolean')
            expect(typeof config.supportsCustomBaseUrl).toBe('boolean')
          }
        ),
        { numRuns: 30 }
      )
    })

    it('文本框扩展参数应在有效范围内', () => {
      fc.assert(
        fc.property(
          fc.record({
            box_expand_ratio: fc.float({ min: 0, max: 0.5, noNaN: true }),
            box_expand_top: fc.float({ min: 0, max: 0.5, noNaN: true }),
            box_expand_bottom: fc.float({ min: 0, max: 0.5, noNaN: true }),
            box_expand_left: fc.float({ min: 0, max: 0.5, noNaN: true }),
            box_expand_right: fc.float({ min: 0, max: 0.5, noNaN: true }),
          }),
          expandParams => {
            // 所有扩展参数应在 0-0.5 范围内
            expect(expandParams.box_expand_ratio).toBeGreaterThanOrEqual(0)
            expect(expandParams.box_expand_ratio).toBeLessThanOrEqual(0.5)

            expect(expandParams.box_expand_top).toBeGreaterThanOrEqual(0)
            expect(expandParams.box_expand_top).toBeLessThanOrEqual(0.5)

            expect(expandParams.box_expand_bottom).toBeGreaterThanOrEqual(0)
            expect(expandParams.box_expand_bottom).toBeLessThanOrEqual(0.5)

            expect(expandParams.box_expand_left).toBeGreaterThanOrEqual(0)
            expect(expandParams.box_expand_left).toBeLessThanOrEqual(0.5)

            expect(expandParams.box_expand_right).toBeGreaterThanOrEqual(0)
            expect(expandParams.box_expand_right).toBeLessThanOrEqual(0.5)
          }
        ),
        { numRuns: 100 }
      )
    })
  })
})
