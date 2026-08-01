import { describe, expect, it } from 'vitest'

import {
  cloneBubbleStates,
  createBubbleState,
  detectTextDirection,
} from '@/utils/bubbleFactory'

describe('bubble factory contracts', () => {
  it('creates independent current bubble state objects', () => {
    const first = createBubbleState({
      coords: [1, 2, 30, 80],
      polygon: [[1, 2], [30, 2], [30, 80], [1, 80]],
      translatedText: '译文',
    })
    const second = createBubbleState()

    first.coords[0] = 99
    first.polygon[0]![0] = 99

    expect(second.coords[0]).toBe(0)
    expect(second.polygon).toEqual([])
    expect(first.translatedText).toBe('译文')
  })

  it('detects direction from current coordinates', () => {
    expect(detectTextDirection([0, 0, 20, 80])).toBe('vertical')
    expect(detectTextDirection([80, 20, 0, 0])).toBe('horizontal')
  })

  it('clones all nested mutable bubble fields', () => {
    const source = createBubbleState({
      coords: [1, 2, 3, 4],
      polygon: [[1, 2], [3, 2], [3, 4]],
      position: { x: 5, y: 6 },
      textlines: [{
        polygon: [[1, 2], [3, 2], [3, 4], [1, 4]],
        direction: 'v',
        confidence: 0.9,
      }],
    })
    const [cloned] = cloneBubbleStates([source])

    expect(cloned).toEqual(source)
    expect(cloned).not.toBe(source)
    expect(cloned?.coords).not.toBe(source.coords)
    expect(cloned?.polygon).not.toBe(source.polygon)
    expect(cloned?.textlines).not.toBe(source.textlines)
  })
})
