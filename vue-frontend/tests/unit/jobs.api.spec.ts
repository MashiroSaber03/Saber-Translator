import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, postMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
  },
}))

import { jobsApi } from '@/api/v2/jobs'

describe('jobs v2 api contracts', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
    vi.spyOn(crypto, 'randomUUID').mockReturnValue('00000000-0000-4000-8000-000000000001')
  })

  it('supports one combined queue and history snapshot', async () => {
    getMock.mockResolvedValue({ items: [], nextCursor: null })

    await jobsApi.list('queue')
    await jobsApi.list('history')
    await jobsApi.list('all')

    expect(getMock).toHaveBeenNthCalledWith(1, '/api/v2/jobs?scope=queue')
    expect(getMock).toHaveBeenNthCalledWith(2, '/api/v2/jobs?scope=history&limit=200')
    expect(getMock).toHaveBeenNthCalledWith(3, '/api/v2/jobs?scope=all&limit=200')
  })

  it.each(['pause', 'resume', 'continue', 'cancel'] as const)(
    'encodes the job identity for the %s command',
    async (command) => {
      postMock.mockResolvedValue({ jobId: 'job/id one' })

      await jobsApi[command]('job/id one')

      expect(postMock).toHaveBeenCalledWith(
        `/api/v2/jobs/job%2Fid%20one/${command}`,
      )
    },
  )

  it('keeps a replay key for retry commands that can create a new attempt', async () => {
    postMock.mockResolvedValue({ jobId: 'retry-1' })

    await jobsApi.retry('job/id one')

    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/jobs/job%2Fid%20one/retry',
      { strategy: 'current' },
      {
        headers: {
          'Idempotency-Key': '00000000-0000-4000-8000-000000000001',
        },
      },
    )
  })

  it('deduplicates snapshot identities while preserving their order', async () => {
    getMock.mockResolvedValue({ items: [] })

    await jobsApi.snapshot(['job-1', 'job-1', 'job/id two'])

    expect(getMock).toHaveBeenCalledWith(
      '/api/v2/jobs/snapshot?job_id=job-1&job_id=job%2Fid+two',
    )
  })

  it('rejects a snapshot beyond the backend contract before issuing a request', () => {
    const jobIds = Array.from({ length: 201 }, (_, index) => `job-${index}`)

    expect(() => jobsApi.snapshot(jobIds)).toThrow('一次最多读取 200 个任务快照')
    expect(getMock).not.toHaveBeenCalled()
  })
})
