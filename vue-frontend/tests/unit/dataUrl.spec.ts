import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  extractBase64Payload,
  readBlobAsDataUrl,
  readFileAsText,
  toImageDataUrl,
} from '@/utils/dataUrl'

describe('dataUrl utilities', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('extracts the payload from data URLs and leaves raw payloads unchanged', () => {
    expect(extractBase64Payload('data:image/png;base64,abc123')).toBe('abc123')
    expect(extractBase64Payload('data:image/webp;base64,with,comma')).toBe('with,comma')
    expect(extractBase64Payload('raw-base64')).toBe('raw-base64')
    expect(extractBase64Payload('')).toBe('')
  })

  it('wraps payloads as image data URLs without double-wrapping existing data URLs', () => {
    expect(toImageDataUrl('abc123')).toBe('data:image/png;base64,abc123')
    expect(toImageDataUrl('abc123', 'image/jpeg')).toBe('data:image/jpeg;base64,abc123')
    expect(toImageDataUrl('data:image/webp;base64,abc123')).toBe('data:image/webp;base64,abc123')
    expect(toImageDataUrl('/api/images/page-1')).toBe('/api/images/page-1')
  })

  it('reads FileReader results through typed helpers instead of caller-side casts', async () => {
    class DataUrlFileReader {
      result: string | ArrayBuffer | null = null
      onload: (() => void) | null = null
      onerror: (() => void) | null = null

      readAsDataURL(): void {
        this.result = 'data:image/png;base64,aW1hZ2U='
        this.onload?.()
      }

      readAsText(): void {
        this.result = '导入文本'
        this.onload?.()
      }
    }

    vi.stubGlobal('FileReader', DataUrlFileReader)

    await expect(readBlobAsDataUrl(new Blob(['image']))).resolves.toBe('data:image/png;base64,aW1hZ2U=')
    await expect(readFileAsText(new File(['text'], 'import.json'))).resolves.toBe('导入文本')
  })

  it('rejects non-string FileReader results at the helper boundary', async () => {
    class ArrayBufferFileReader {
      result: string | ArrayBuffer | null = null
      onload: (() => void) | null = null
      onerror: (() => void) | null = null

      readAsDataURL(): void {
        this.result = new ArrayBuffer(4)
        this.onload?.()
      }

      readAsText(): void {
        this.result = new ArrayBuffer(4)
        this.onload?.()
      }
    }

    vi.stubGlobal('FileReader', ArrayBufferFileReader)

    await expect(readBlobAsDataUrl(new Blob(['image']), '图片读取失败')).rejects.toThrow('图片读取失败')
    await expect(readFileAsText(new File(['{}'], 'import.json'), '文本读取失败')).rejects.toThrow('文本读取失败')
  })

  it('keeps translation steps on the shared payload helper', () => {
    const stepFiles = [
      'src/composables/translation/core/steps/detection.ts',
      'src/composables/translation/core/steps/ocr.ts',
      'src/composables/translation/core/steps/inpaint.ts',
      'src/composables/translation/core/steps/color.ts',
      'src/composables/translation/core/steps/aiTranslate.ts',
      'src/composables/useTextStyleSync.ts',
    ]

    for (const file of stepFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).toContain("from '@/utils/dataUrl'")
      expect(source, file).not.toMatch(/function\s+extractBase64\b/)
      expect(source, file).not.toMatch(/\.split\(['"]base64,/)
    }
  })

  it('keeps task projection on the shared data URL helper', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/composables/translation/core/taskProjector.ts'), 'utf8')

    expect(source).toContain("from '@/utils/dataUrl'")
    expect(source).not.toMatch(/function\s+ensureDataUrl\b/)
    expect(source).not.toContain('data:image/png;base64,${data}')
  })

  it('keeps persistence payload extraction on the shared data URL helper', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/composables/translation/core/persistenceService.ts'), 'utf8')

    expect(source).toContain("from '@/utils/dataUrl'")
    expect(source).toContain('extractBase64Payload(')
    expect(source).not.toMatch(/\.split\(['"],['"]\)/)
  })

  it('keeps file and blob readers behind shared typed helpers', () => {
    const files = [
      'src/components/bookshelf/BookModal.vue',
      'src/components/translate/ImageUpload.vue',
      'src/composables/translation/core/persistenceService.ts',
      'src/stores/sessionStore.ts',
      'src/composables/useImageConverter.ts',
      'src/composables/useExportImport.ts',
    ]

    for (const file of files) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).toContain("from '@/utils/dataUrl'")
      expect(source, file).not.toContain('new FileReader()')
      expect(source, file).not.toMatch(/reader\.result\s+as\s+string/)
      expect(source, file).not.toMatch(/target\?\.result\s+as\s+string/)
      expect(source, file).not.toMatch(/e\.target\.result\s+as\s+string/)
    }
  })
})
