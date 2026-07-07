import { beforeEach, describe, expect, it, vi } from 'vitest'

const { postMock } = vi.hoisted(() => ({
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    post: postMock,
  },
}))

describe('parallel translate api', () => {
  beforeEach(() => {
    postMock.mockReset()
  })

  it('routes every parallel step through the shared endpoint root', async () => {
    const {
      parallelColor,
      parallelDetect,
      parallelInpaint,
      parallelOcr,
      parallelRender,
      parallelTranslate,
    } = await import('@/api/parallelTranslate')

    await parallelDetect({ image: 'img' })
    await parallelOcr({ image: 'img', bubble_coords: [[1, 2, 3, 4]] })
    await parallelColor({ image: 'img', bubble_coords: [[1, 2, 3, 4]] })
    await parallelTranslate({
      original_texts: ['hello'],
      target_language: '中文',
      model_provider: 'openai',
    })
    await parallelInpaint({ image: 'img', bubble_coords: [[1, 2, 3, 4]] })
    await parallelRender({ clean_image: 'img', bubble_states: [] })

    expect(postMock.mock.calls.map(call => call[0])).toEqual([
      '/api/parallel/detect',
      '/api/parallel/ocr',
      '/api/parallel/color',
      '/api/parallel/translate',
      '/api/parallel/inpaint',
      '/api/parallel/render',
    ])
  })
})
