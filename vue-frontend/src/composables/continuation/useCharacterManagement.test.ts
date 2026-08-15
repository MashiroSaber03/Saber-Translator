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

    expect(source).toContain('runContinuationMutation,')
    expect(source).toContain("from './continuationActionRunner'")
    expect(source).not.toContain("error instanceof Error ? error.message : '网络错误'")
    expect(source).not.toMatch(/state\.showMessage\('[^']+失败: ' \+ result\.error, 'error'\)/)
    expect(helperSource).toContain('function formatContinuationActionError')
  })

  it('uploads form images using the v2 file field', async () => {
    uploadFormImageMock.mockResolvedValue('/api/v2/assets/form-image')

    const state = {
      characters: ref([]),
      imageRefreshKey: ref(0),
      initializeData: vi.fn().mockResolvedValue(undefined),
      showMessage: vi.fn(),
    }

    const management = useCharacterManagement(ref('book-1'), state as never)
    const file = new File(['demo'], 'form.png', { type: 'image/png' })

    const succeeded = await management.uploadFormImage('Saber', 'form_1', file)

    expect(succeeded).toBe(true)
    expect(uploadFormImageMock).toHaveBeenCalledTimes(1)
    expect(uploadFormImageMock).toHaveBeenCalledWith('book-1', 'Saber', 'form_1', file)
  })

  it('returns failure after reporting a rejected mutation', async () => {
    uploadFormImageMock.mockRejectedValueOnce(new Error('upload failed'))
    const state = {
      characters: ref([]),
      imageRefreshKey: ref(0),
      initializeData: vi.fn().mockResolvedValue(undefined),
      showMessage: vi.fn(),
    }
    const management = useCharacterManagement(ref('book-1'), state as never)
    const file = new File(['demo'], 'form.png', { type: 'image/png' })

    await expect(management.uploadFormImage('Saber', 'form_1', file)).resolves.toBe(false)
    expect(state.showMessage).toHaveBeenCalledWith('上传失败: upload failed', 'error')
  })

  it('does not refresh or report an old-book mutation after switching books', async () => {
    let resolveUpload!: (value: string) => void
    uploadFormImageMock.mockImplementationOnce(() => new Promise(resolve => {
      resolveUpload = resolve
    }))
    const currentBookId = ref<string | undefined>('book-1')
    const state = {
      characters: ref([]),
      imageRefreshKey: ref(0),
      initializeData: vi.fn().mockResolvedValue(undefined),
      showMessage: vi.fn(),
    }
    const management = useCharacterManagement(currentBookId, state as never)
    const file = new File(['demo'], 'form.png', { type: 'image/png' })

    const pending = management.uploadFormImage('Saber', 'form_1', file)
    currentBookId.value = 'book-2'
    await Promise.resolve()
    resolveUpload('/api/v2/assets/form-image')

    await expect(pending).resolves.toBe(false)
    expect(state.initializeData).not.toHaveBeenCalled()
    expect(state.showMessage).not.toHaveBeenCalled()
  })
})
