import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useInsightStore, type AnalysisStatus } from '@/stores/insightStore'

type ProgressPair = {
  current: number
  total: number
}

function createStore(): ReturnType<typeof useInsightStore> {
  setActivePinia(createPinia())
  return useInsightStore()
}

const analysisStatusArbitrary = fc.constantFrom<AnalysisStatus>(
  'idle',
  'running',
  'paused',
  'completed',
  'failed',
)

const progressArbitrary: fc.Arbitrary<ProgressPair> = fc
  .record({
    current: fc.integer({ min: 0, max: 1000 }),
    total: fc.integer({ min: 0, max: 1000 }),
  })
  .filter(progress => progress.total === 0 || progress.current <= progress.total)

const bookIdArbitrary = fc.stringOf(
  fc.constantFrom(...'abcdef0123456789'.split('')),
  { minLength: 8, maxLength: 8 },
)

describe('insight progress properties', () => {
  it('computes progress percent from current and total', () => {
    fc.assert(
      fc.property(progressArbitrary, ({ current, total }) => {
        const store = createStore()
        const expectedPercent = total === 0 ? 0 : Math.round((current / total) * 100)

        store.updateProgress(current, total)

        expect(store.progressPercent).toBe(expectedPercent)
        expect(store.progressPercent).toBeGreaterThanOrEqual(0)
        expect(store.progressPercent).toBeLessThanOrEqual(100)
      }),
    )
  })

  it('treats zero-total progress as zero percent for every current value', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 1000 }), current => {
        const store = createStore()

        store.updateProgress(current, 0)

        expect(store.progress.current).toBe(current)
        expect(store.progress.total).toBe(0)
        expect(store.progressPercent).toBe(0)
      }),
    )
  })

  it('mirrors the last analysis status into derived state', () => {
    fc.assert(
      fc.property(fc.array(analysisStatusArbitrary, { minLength: 1, maxLength: 10 }), statusSequence => {
        const store = createStore()

        for (const status of statusSequence) {
          store.setAnalysisStatus(status)
        }

        const lastStatus = statusSequence[statusSequence.length - 1] as AnalysisStatus
        expect(store.analysisStatus).toBe(lastStatus)
        expect(store.progress.status).toBe(lastStatus)
        expect(store.isAnalyzing).toBe(lastStatus === 'running')
        expect(store.isAnalysisCompleted).toBe(lastStatus === 'completed')
      }),
    )
  })

  it('keeps pause and resume transitions deterministic', () => {
    fc.assert(
      fc.property(fc.array(fc.boolean(), { minLength: 1, maxLength: 20 }), pauseSequence => {
        const store = createStore()
        store.setAnalysisStatus('running')

        for (const shouldPause of pauseSequence) {
          store.setAnalysisStatus(shouldPause ? 'paused' : 'running')
        }

        const lastAction = pauseSequence[pauseSequence.length - 1]
        const expectedStatus: AnalysisStatus = lastAction ? 'paused' : 'running'
        expect(store.analysisStatus).toBe(expectedStatus)
        expect(store.isAnalyzing).toBe(expectedStatus === 'running')
      }),
    )
  })

  it('does not let progress updates change the current analysis status', () => {
    fc.assert(
      fc.property(
        analysisStatusArbitrary,
        fc.array(progressArbitrary, { minLength: 1, maxLength: 10 }),
        (initialStatus, progressUpdates) => {
          const store = createStore()
          store.setAnalysisStatus(initialStatus)

          for (const { current, total } of progressUpdates) {
            store.updateProgress(current, total)
          }

          expect(store.analysisStatus).toBe(initialStatus)
          expect(store.progress.status).toBe(initialStatus)
        },
      ),
    )
  })

  it('resets analysis progress without changing the selected book', () => {
    fc.assert(
      fc.property(
        analysisStatusArbitrary,
        progressArbitrary,
        bookIdArbitrary,
        (status, progress, bookId) => {
          const store = createStore()
          store.setCurrentBook(bookId)
          store.setAnalysisStatus(status)
          store.updateProgress(progress.current, progress.total)

          store.resetAnalysis()

          expect(store.currentBookId).toBe(bookId)
          expect(store.analysisStatus).toBe('idle')
          expect(store.progress).toEqual({ current: 0, total: 0, status: 'idle' })
          expect(store.progressPercent).toBe(0)
        },
      ),
    )
  })
})
