import { beforeEach, describe, expect, it, vi } from 'vitest'

const { postMock } = vi.hoisted(() => ({
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    post: postMock,
  },
}))

describe('translate api', () => {
  beforeEach(() => {
    postMock.mockReset()
  })

  it('routes translate endpoints through the current API paths', async () => {
    const {
      extractGlossaryEntries,
      hqTranslateBatch,
      reRenderImage,
    } = await import('@/api/translate')

    await reRenderImage({
      clean_image: 'clean',
      bubble_texts: ['hello'],
      bubble_coords: [[0, 0, 10, 10]],
    })
    await hqTranslateBatch({
      provider: 'openai',
      api_key: 'key',
      model_name: 'model',
      jsonData: [],
      imageBase64Array: [],
    })
    await extractGlossaryEntries({
      original_texts: ['hello'],
      target_language: '中文',
      model_provider: 'openai',
    })

    expect(postMock.mock.calls.map(call => call[0])).toEqual([
      '/api/re_render_image',
      '/api/hq_translate_batch',
      '/api/translation/glossary/extract',
    ])
  })

  it('wraps single text translation success and failure in ApiResponse shape', async () => {
    const { translateSingleText } = await import('@/api/translate')
    const params = {
      original_text: 'hello',
      model_provider: 'openai',
      target_language: '中文',
    }

    postMock.mockResolvedValueOnce({
      translated_text: '你好',
      warnings: [{ type: 'glossary' }],
    })
    await expect(translateSingleText(params)).resolves.toEqual({
      success: true,
      data: {
        translated_text: '你好',
        warnings: [{ type: 'glossary' }],
      },
    })
    expect(postMock).toHaveBeenCalledWith('/api/translate_single_text', params)

    postMock.mockRejectedValueOnce(new Error('model failed'))
    await expect(translateSingleText(params)).resolves.toEqual({
      success: false,
      error: 'model failed',
    })
  })

  it('maps single-bubble OCR and inpaint payloads to backend wire fields', async () => {
    const { inpaintSingleBubble, ocrSingleBubble } = await import('@/api/translate')

    await ocrSingleBubble('image-data', [1, 2, 3, 4], 'ai_vision', {
      ai_vision_provider: 'openai',
      bubble_textlines: [{ text: 'line' }],
    })
    await inpaintSingleBubble('image-data', [1, 2, 3, 4], {
      bubbleAngle: 15,
      lamaModel: 'litelama',
      maskData: 'mask-data',
    })
    await inpaintSingleBubble('image-data', [5, 6, 7, 8])

    expect(postMock).toHaveBeenNthCalledWith(1, '/api/ocr_single_bubble', {
      image_data: 'image-data',
      bubble_coords: [1, 2, 3, 4],
      ocr_engine: 'ai_vision',
      ai_vision_provider: 'openai',
      bubble_textlines: [{ text: 'line' }],
    })
    expect(postMock).toHaveBeenNthCalledWith(2, '/api/inpaint_single_bubble', {
      image_data: 'image-data',
      bubble_coords: [1, 2, 3, 4],
      bubble_angle: 15,
      method: 'lama',
      lama_model: 'litelama',
      mask_data: 'mask-data',
    })
    expect(postMock).toHaveBeenNthCalledWith(3, '/api/inpaint_single_bubble', {
      image_data: 'image-data',
      bubble_coords: [5, 6, 7, 8],
      bubble_angle: 0,
      method: 'lama',
      lama_model: 'lama_mpe',
    })
  })
})
