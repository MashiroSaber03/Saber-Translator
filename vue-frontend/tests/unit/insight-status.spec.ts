import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { resolveAnalysisStatus } from '@/utils/insightStatus'
import type { InsightAnalysisSnapshot } from '@/types/insight'

function createStatusSnapshot(overrides: Partial<InsightAnalysisSnapshot>): InsightAnalysisSnapshot {
  return {
    fullyAnalyzed: false,
    analyzedPagesCount: 0,
    ...overrides,
  }
}

describe('resolveAnalysisStatus', () => {
  it('keeps Insight status fixtures typed to the current response contract', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/insight-status.spec.ts'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('does not report completion while the book is only partially analyzed', () => {
    const status = resolveAnalysisStatus(createStatusSnapshot({ fullyAnalyzed: false }))

    expect(status).toBe('idle')
  })

  it('returns completed when no current task and the whole book is analyzed', () => {
    const status = resolveAnalysisStatus(createStatusSnapshot({ fullyAnalyzed: true }))

    expect(status).toBe('completed')
  })

  it('does not return completed when the latest task completed only part of the book', () => {
    const status = resolveAnalysisStatus(createStatusSnapshot({
      fullyAnalyzed: false,
      currentTask: {
        jobId: 'job-1',
        status: 'completed',
        progress: { analyzedPages: 1, totalPages: 2 },
      },
    }))

    expect(status).toBe('idle')
  })

  it.each(['queued', 'pausing', 'cancelling', 'interrupted', 'completed_with_errors'] as const)(
    'preserves the backend %s state instead of projecting a fake lifecycle state',
    taskStatus => {
      const status = resolveAnalysisStatus(createStatusSnapshot({
        currentTask: {
          jobId: 'job-1',
          status: taskStatus,
          progress: { analyzedPages: 1, totalPages: 2 },
        },
      }))

      expect(status).toBe(taskStatus)
    },
  )
})
