/**
 * 图片URL转Base64属性测试
 * 
 * **Feature: vue-frontend-migration, Property 33: 图片URL转Base64一致性**
 * **Validates: Requirements 14.4**
 */

import { describe, it, expect } from 'vitest'
import { useImageConverter } from '@/composables/useImageConverter'

describe('图片URL转Base64属性测试', () => {
  // 有效的 Base64 PNG 图片（1x1 像素）
  const validPng = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='

  it('isValidBase64Image 验证有效图片返回 true', () => {
    const { isValidBase64Image } = useImageConverter()
    expect(isValidBase64Image(validPng)).toBe(true)
  })

  it('isValidBase64Image 验证无效字符串返回 false', () => {
    const { isValidBase64Image } = useImageConverter()
    expect(isValidBase64Image('')).toBe(false)
    expect(isValidBase64Image('not-a-base64')).toBe(false)
  })

  it('getBase64MimeType 提取 MIME 类型正确', () => {
    const { getBase64MimeType } = useImageConverter()
    expect(getBase64MimeType(validPng)).toBe('image/png')
    expect(getBase64MimeType('not-a-base64')).toBeNull()
  })

  it('getBase64Extension 提取扩展名正确', () => {
    const { getBase64Extension } = useImageConverter()
    expect(getBase64Extension(validPng)).toBe('png')
    expect(getBase64Extension('not-a-base64')).toBe('png')
  })

  it('getBase64Size 计算大小正确', () => {
    const { getBase64Size } = useImageConverter()
    expect(getBase64Size(validPng)).toBeGreaterThan(0)
    expect(getBase64Size('')).toBe(0)
  })

  it('base64ToBlob 转换正确', () => {
    const { base64ToBlob } = useImageConverter()
    const blob = base64ToBlob(validPng)
    expect(blob).not.toBeNull()
    expect(blob?.type).toBe('image/png')
  })

  it('base64ToBlob 无效输入返回 null', () => {
    const { base64ToBlob } = useImageConverter()
    expect(base64ToBlob('')).toBeNull()
    expect(base64ToBlob('not-a-base64')).toBeNull()
  })

  it('大小计算与 Blob 大小一致', () => {
    const { getBase64Size, base64ToBlob } = useImageConverter()
    const size = getBase64Size(validPng)
    const blob = base64ToBlob(validPng)
    expect(size).toBe(blob?.size)
  })
})
