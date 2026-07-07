import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it } from 'vitest'
import { ParallelProgressTracker } from '@/composables/translation/parallel/ParallelProgressTracker'
import { useParallelTranslation } from '@/composables/translation/parallel/useParallelTranslation'

describe('parallel progress defaults', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('uses one shared source for tracker and global progress pool metadata', () => {
    const tracker = new ParallelProgressTracker()
    const parallel = useParallelTranslation()

    expect(parallel.progress.value.pools).toEqual(tracker.getAllPoolStatuses())
    expect(parallel.progress.value.pools).not.toBe(tracker.getAllPoolStatuses())
    expect(parallel.progress.value.pools[0]).not.toBe(tracker.getAllPoolStatuses()[0])
  })

  it('keeps parallel progress source files free of duplicated pool metadata and scaffold narration', () => {
    const trackerSource = readFileSync(
      resolve(process.cwd(), 'src/composables/translation/parallel/ParallelProgressTracker.ts'),
      'utf8',
    )
    const entrySource = readFileSync(
      resolve(process.cwd(), 'src/composables/translation/parallel/useParallelTranslation.ts'),
      'utf8',
    )

    expect(trackerSource).toContain("from './progressDefaults'")
    expect(entrySource).toContain("from './progressDefaults'")
    expect(entrySource).not.toContain("name: '检测'")
    expect(entrySource).not.toContain("name: 'OCR'")
    expect(entrySource).not.toContain('并行翻译 Composable')
    expect(entrySource).not.toContain('全局响应式进度状态')
    expect(entrySource).not.toContain('/**')
    expect(entrySource).not.toContain('  //')
  })
})
