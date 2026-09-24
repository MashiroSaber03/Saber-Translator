import { describe, expect, it } from 'vitest'
import { buildTimelineItems } from '@/components/settings/pluginAgentTimeline'

describe('plugin agent event time', () => {
  it.each([
    [19, 31, 18, '19:31:18'],
    [0, 5, 9, '00:05:09'],
  ] as const)('shows local time for a UTC event at %s:%s:%s', (hour, minute, second, expected) => {
    const timestamp = new Date(2026, 8, 24, hour, minute, second).toISOString()
    const items = buildTimelineItems([
      { id: 1, type: 'state', timestamp, payload: { run_state: 'completed' } },
    ], {}, {})

    expect(items[0]?.timestampLabel).toBe(expected)
  })

  it('treats equivalent UTC and explicit-offset timestamps as the same instant', () => {
    const items = buildTimelineItems([
      { id: 1, type: 'state', timestamp: '2026-09-24T11:31:18Z', payload: { run_state: 'completed' } },
      { id: 2, type: 'state', timestamp: '2026-09-24T19:31:18+08:00', payload: { run_state: 'completed' } },
    ], {}, {})

    expect(items[0]?.timestampLabel).toBe(items[1]?.timestampLabel)
  })
})
