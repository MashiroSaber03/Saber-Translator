import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it } from 'vitest'
import { saveDetectionResultToImage } from '@/composables/translation/core/detectionResultWriter'
import { useImageStore } from '@/stores/imageStore'
import { createBubbleState } from '@/utils/bubbleFactory'
import type { DetectionOutput } from '@/composables/translation/core/steps/detection'

function createDetectionOutput(): DetectionOutput {
  return {
    bubbleCoords: [[0, 0, 120, 80]],
    bubbleAngles: [5],
    bubblePolygons: [[[0, 0], [120, 0], [120, 80], [0, 80]]],
    autoDirections: ['vertical'],
    textlinesPerBubble: [[
      {
        text: '原文',
        polygon: [[0, 0], [100, 0], [100, 20], [0, 20]],
      },
    ]],
    originalTexts: ['原文'],
    textMask: 'mask-data',
    bubbleStates: [
      createBubbleState({
        coords: [0, 0, 120, 80],
        polygon: [[0, 0], [120, 0], [120, 80], [0, 80]],
        originalText: '原文',
      }),
    ],
  }
}

describe('detectionResultWriter', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('keeps the helper compact and injectable', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/translation/core/detectionResultWriter.ts'),
      'utf8',
    )

    expect(source).not.toContain('/**')
    expect(source).not.toContain('统一保存检测结果')
    expect(source).toContain('imageStore?: ReturnType<typeof useImageStore>')
    expect(source).toContain('options.imageStore ?? useImageStore()')
  })

  it('writes detection fields when no bubble-state override is provided', () => {
    const imageStore = useImageStore()
    imageStore.addImage('page.png', 'data:image/png;base64,page')
    const detection = createDetectionOutput()
    detection.bubbleStates = []

    saveDetectionResultToImage(0, detection, { imageStore })

    expect(imageStore.images[0]).toMatchObject({
      bubbleCoords: detection.bubbleCoords,
      bubbleAngles: detection.bubbleAngles,
      textMask: 'mask-data',
      textlinesPerBubble: detection.textlinesPerBubble,
    })
  })

  it('preserves explicit bubble textlines when bubble states are injected', () => {
    const imageStore = useImageStore()
    imageStore.addImage('page.png', 'data:image/png;base64,page')
    const detection = createDetectionOutput()
    const bubbleStates = [
      createBubbleState({
        coords: [0, 0, 120, 80],
        polygon: [],
        textlines: [
          {
            text: 'kept',
            polygon: [[1, 1], [2, 1], [2, 2], [1, 2]],
          },
        ],
      }),
    ]

    saveDetectionResultToImage(0, detection, {
      imageStore,
      updateBubbleStates: true,
      bubbleStates,
    })

    expect(imageStore.images[0]?.bubbleStates?.[0]?.textlines).toEqual(bubbleStates[0]?.textlines)
    expect(imageStore.images[0]?.textlinesPerBubble).toEqual(bubbleStates.map((bubble) => bubble.textlines))
  })

  it('uses detection textlines when injected bubble states do not already have them', () => {
    const imageStore = useImageStore()
    imageStore.addImage('page.png', 'data:image/png;base64,page')
    const detection = createDetectionOutput()

    saveDetectionResultToImage(0, detection, {
      imageStore,
      updateBubbleStates: true,
      bubbleStates: [
        createBubbleState({
          coords: [0, 0, 120, 80],
          polygon: [],
          textlines: [],
        }),
      ],
    })

    expect(imageStore.images[0]?.bubbleStates?.[0]?.textlines).toEqual(detection.textlinesPerBubble[0])
  })
})
