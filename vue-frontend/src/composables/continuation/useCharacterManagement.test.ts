import { ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import { useCharacterManagement } from './useCharacterManagement'

const { uploadFormImageMock } = vi.hoisted(() => ({
  uploadFormImageMock: vi.fn(),
}))

vi.mock('@/api/continuation', () => ({
  uploadFormImage: uploadFormImageMock,
}))

describe('useCharacterManagement', () => {
  it('uploads form images using the image field expected by the backend', async () => {
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
    expect(formData.has('image')).toBe(true)
    expect(formData.get('image')).toBe(file)
    expect(formData.has('file')).toBe(false)
  })
})
