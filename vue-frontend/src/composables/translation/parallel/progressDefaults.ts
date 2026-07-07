import type { ParallelProgress, PoolStatus } from './types'

const PARALLEL_POOL_METADATA: Array<{ name: string; icon: PoolStatus['icon'] }> = [
  { name: '检测', icon: 'map-pin' },
  { name: 'OCR', icon: 'book-open' },
  { name: '颜色', icon: 'palette' },
  { name: '术语', icon: 'tags' },
  { name: '翻译', icon: 'globe' },
  { name: '修复', icon: 'paintbrush' },
  { name: '渲染', icon: 'sparkles' },
]

export function createParallelPoolStatuses(): PoolStatus[] {
  return PARALLEL_POOL_METADATA.map((pool) => ({
    name: pool.name,
    icon: pool.icon,
    waiting: 0,
    processing: false,
    completed: 0,
    isWaitingLock: false,
  }))
}

export function createInitialParallelProgress(): ParallelProgress {
  return {
    pools: createParallelPoolStatuses(),
    totalCompleted: 0,
    totalFailed: 0,
    totalPages: 0,
    estimatedTimeRemaining: 0,
    preSave: undefined,
    save: undefined,
  }
}
