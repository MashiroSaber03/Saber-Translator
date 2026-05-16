import { describe, expect, it, vi } from 'vitest'

const { postMock } = vi.hoisted(() => ({
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    post: postMock,
  },
}))

import { notifyPipelineBefore, PipelineCancelledError } from '@/api/pipeline'

describe('pipeline API error compatibility', () => {
  it('still recognizes plugin-cancel errors when apiClient rejects Error instances', async () => {
    class MockApiClientError extends Error {
      readonly code = 'PIPELINE_CANCELLED'
      readonly status = 409
      readonly details = { cancelled_by_plugin: true, plugin_name: 'guard' }
    }

    postMock.mockRejectedValueOnce(new MockApiClientError('任务被插件取消'))

    const promise = notifyPipelineBefore({
      pipeline_id: 'pipe-1',
      mode: 'standard',
      scope: 'current',
      page_indexes: [0],
      total_images: 1,
    })

    await expect(
      promise
    ).rejects.toBeInstanceOf(PipelineCancelledError)

    await expect(
      promise
    ).rejects.toMatchObject({
      message: '任务被插件取消',
      pipelineId: 'pipe-1',
      details: {
        cancelled_by_plugin: true,
        plugin_name: 'guard',
      },
    })
  })
})
