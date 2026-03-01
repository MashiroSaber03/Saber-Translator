import { describe, expect, it } from 'vitest'
import { resolveAnalysisStatus } from '@/utils/insightStatus'

describe('resolveAnalysisStatus', () => {
  it('does not treat analyzed=true as completed when fully_analyzed=false', () => {
    const status = resolveAnalysisStatus({
      success: true,
      analyzed: true,
      fully_analyzed: false,
    } as any)

    expect(status).toBe('idle')
  })

  it('returns completed when no current_task and fully_analyzed=true', () => {
    const status = resolveAnalysisStatus({
      success: true,
      fully_analyzed: true,
    } as any)

    expect(status).toBe('completed')
  })

  it('does not return completed when current_task=completed but fully_analyzed=false', () => {
    const status = resolveAnalysisStatus({
      success: true,
      fully_analyzed: false,
      current_task: {
        status: 'completed',
      },
    } as any)

    expect(status).toBe('idle')
  })
})
