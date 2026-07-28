import { ref } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it, vi } from 'vitest'

import { useCharacterManagement } from './useCharacterManagement'

const { uploadFormImageMock } = vi.hoisted(() => ({
  uploadFormImageMock: vi.fn(),
}))

vi.mock('@/api/continuation', () => ({
  uploadFormImage: uploadFormImageMock,
}))

describe('useCharacterManagement', () => {
  it('keeps mutation message handling behind the shared continuation action helper', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/composables/continuation/useCharacterManagement.ts'), 'utf8')
    const helperSource = readFileSync(resolve(process.cwd(), 'src/composables/continuation/continuationActionRunner.ts'), 'utf8')

    expect(source).toContain("import { runContinuationMutation, toContinuationActionError } from './continuationActionRunner'")
    expect(source).not.toContain("error instanceof Error ? error.message : '网络错误'")
    expect(source).not.toMatch(/state\.showMessage\('[^']+失败: ' \+ result\.error, 'error'\)/)
    expect(helperSource).toContain('function formatContinuationActionError')
  })

  it('uploads form images using the v2 file field', async () => {
    uploadFormImageMock.mockResolvedValue({ success: true, image_path: '/tmp/form.png' })

    const state = {
      characters: ref([]),
      imageRefreshKey: ref(0),
      initializeData: vi.fn().mockResolvedValue(undefined),
      showMessage: vi.fn(),
    }

    const management = useCharacterManagement(ref('book-1'), state as never)
    const file = new File(['demo'], 'form.png', { type: 'image/png' })

    await management.uploadFormImage('Saber', 'form_1', file)

    expect(uploadFormImageMock).toHaveBeenCalledTimes(1)
    const [, , , formData] = uploadFormImageMock.mock.calls[0]
    expect(formData).toBeInstanceOf(FormData)
    expect(formData.has('file')).toBe(true)
    expect(formData.get('file')).toBe(file)
    expect(formData.has('image')).toBe(false)
  })
})
