import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { useImageConverter } from '@/composables/useImageConverter'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

const validPng = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='

describe('useImageConverter', () => {
  it('keeps the image conversion helper source free of scaffold narration', () => {
    const content = source('src/composables/useImageConverter.ts')

    for (const staleNarration of [
      '图片转换组合式函数',
      '处理图片 URL 转 Base64',
      '图片转换结果',
      '批量转换进度',
      '@param',
      '@returns',
      '// 状态',
      '// 核心转换方法',
      '// 图片处理方法',
      '// 工具方法',
    ]) {
      expect(content).not.toContain(staleNarration)
    }
  })

  it('keeps fixed image conversion examples out of the property-test suite', () => {
    expect(existsSync(resolve(process.cwd(), 'tests/property/imageConverter.property.ts'))).toBe(false)
  })

  it('computes base64 metadata and blob size from the real helper', () => {
    const { base64ToBlob, getBase64Extension, getBase64MimeType, getBase64Size, isValidBase64Image } = useImageConverter()

    const blob = base64ToBlob(validPng)

    expect(isValidBase64Image(validPng)).toBe(true)
    expect(getBase64MimeType(validPng)).toBe('image/png')
    expect(getBase64Extension(validPng)).toBe('png')
    expect(blob).not.toBeNull()
    expect(getBase64Size(validPng)).toBe(blob?.size)
  })

  it('rejects invalid base64 image input', () => {
    const { isValidBase64Image } = useImageConverter()

    expect(isValidBase64Image('')).toBe(false)
    expect(isValidBase64Image('not-a-base64')).toBe(false)
  })

  it('falls back to png extension for invalid input', () => {
    const { getBase64Extension, getBase64MimeType } = useImageConverter()

    expect(getBase64MimeType('not-a-base64')).toBeNull()
    expect(getBase64Extension('not-a-base64')).toBe('png')
  })

  it('returns zero decoded size for invalid input', () => {
    const { getBase64Size } = useImageConverter()

    expect(getBase64Size('')).toBe(0)
  })

  it('returns null when base64 input cannot be converted to a blob', () => {
    const { base64ToBlob } = useImageConverter()

    expect(base64ToBlob('')).toBeNull()
    expect(base64ToBlob('not-a-base64')).toBeNull()
  })
})
