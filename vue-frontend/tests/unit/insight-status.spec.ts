import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { resolveAnalysisStatus } from '@/utils/insightStatus'
import type { AnalysisTask, InsightStatusResponse } from '@/types/insight'

function createStatusResponse(overrides: Partial<InsightStatusResponse>): InsightStatusResponse {
  return {
    success: true,
    ...overrides,
  }
}

function createAnalysisTask(overrides: Partial<AnalysisTask>): AnalysisTask {
  return {
    task_id: 'task-1',
    book_id: 'book-1',
    task_type: 'full_book',
    status: 'running',
    progress: {
      current_phase: 'analysis',
      current_page: 0,
      analyzed_pages: 0,
      total_pages: 1,
    },
    created_at: '2026-01-01T00:00:00.000Z',
    ...overrides,
  }
}

describe('resolveAnalysisStatus', () => {
  it('keeps Insight status fixtures typed to the current response contract', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/insight-status.spec.ts'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('does not treat analyzed=true as completed when fully_analyzed=false', () => {
    const status = resolveAnalysisStatus(createStatusResponse({
      analyzed: true,
      fully_analyzed: false,
    }))

    expect(status).toBe('idle')
  })

  it('returns completed when no current_task and fully_analyzed=true', () => {
    const status = resolveAnalysisStatus(createStatusResponse({
      fully_analyzed: true,
    }))

    expect(status).toBe('completed')
  })

  it('does not return completed when current_task=completed but fully_analyzed=false', () => {
    const status = resolveAnalysisStatus(createStatusResponse({
      fully_analyzed: false,
      current_task: createAnalysisTask({
        status: 'completed',
      }),
    }))

    expect(status).toBe('idle')
  })
})
