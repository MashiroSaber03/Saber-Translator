import { describe, expect, it, vi } from 'vitest'

const { postMock } = vi.hoisted(() => ({
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    post: postMock,
  },
}))

import { extractGlossaryEntries } from '@/api/translate'

describe('extractGlossaryEntries', () => {
  it('posts glossary extraction requests to the dedicated backend endpoint', async () => {
    postMock.mockResolvedValueOnce({
      success: true,
      new_entries: [],
      candidate_count: 0,
      duplicate_count: 0,
    })

    const payload = {
      original_texts: ['Alice'],
      source_language: 'japanese',
      target_language: 'zh',
      model_provider: 'siliconflow',
      api_key: 'test-key',
      model_name: 'test-model',
      existing_entries: [],
      openai_options: {
        request: {
          force_json_output: true,
        },
        execution: {
          use_stream: false,
          rpm_limit: 0,
          transport_retries: 1,
          business_retries: 0,
        },
      },
    }

    await extractGlossaryEntries(payload)

    expect(postMock).toHaveBeenCalledWith('/api/translation/glossary/extract', payload)
  })
})
