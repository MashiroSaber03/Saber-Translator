import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { createBubbleState } from '@/utils/bubbleFactory'

const { executeRenderMock, showToastMock } = vi.hoisted(() => ({
  executeRenderMock: vi.fn(),
  showToastMock: vi.fn(),
}))

vi.mock('@/composables/translation/core/steps', () => ({
  executeRender: executeRenderMock,
}))

vi.mock('@/composables/useTranslationPipeline', () => ({
  useTranslation: () => ({
    progress: {
      value: {
        isInProgress: false,
        current: 0,
        total: 0,
        completed: 0,
        failed: 0,
        label: '',
        percentage: 0,
      },
    },
  }),
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

describe('useTextStyleSync', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    executeRenderMock.mockReset()
    showToastMock.mockReset()
  })

  it('explicit auto font size re-application requests initialize_auto once and writes back the new sizes', async () => {
    executeRenderMock.mockResolvedValue({
      finalImage: 'rendered-auto-font',
      bubbleStates: [
        createBubbleState({
          coords: [0, 0, 120, 80],
          polygon: [],
          translatedText: '第一页译文',
          fontSize: 33,
          textColor: '#123456',
          fillColor: '#abcdef',
        }),
      ],
    })

    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const settingsStore = useSettingsStore()

    settingsStore.updateTextStyle({ autoFontSize: true })

    imageStore.addImage('page-1.png', 'data:image/png;base64,page1')
    imageStore.updateCurrentImage({
      translatedDataURL: 'data:image/png;base64,old-render',
      cleanImageData: 'clean-image',
    })

    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 120, 80],
        polygon: [],
        translatedText: '第一页译文',
        fontSize: 22,
        textColor: '#123456',
        fillColor: '#abcdef',
      }),
    ])

    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')
    const { handleAutoFontSizeChanged } = useTextStyleSync()

    await handleAutoFontSizeChanged(true)

    expect(executeRenderMock).toHaveBeenCalledWith(expect.objectContaining({
      renderStylePolicy: {
        fontSize: 'initialize_auto',
        color: 'preserve',
      },
    }))
    expect(imageStore.currentImage?.bubbleStates?.[0]?.fontSize).toBe(33)
  })

  it('explicit auto text color re-application materializes auto colors into bubble states before a preserve render', async () => {
    executeRenderMock.mockResolvedValue({
      finalImage: 'rendered-auto-color',
      bubbleStates: [
        createBubbleState({
          coords: [0, 0, 120, 80],
          polygon: [],
          translatedText: '第一页译文',
          fontSize: 22,
          textColor: '#010203',
          fillColor: '#0a0b0c',
          autoFgColor: [1, 2, 3],
          autoBgColor: [10, 11, 12],
        }),
      ],
    })

    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const settingsStore = useSettingsStore()

    settingsStore.updateTextStyle({ useAutoTextColor: true })

    imageStore.addImage('page-1.png', 'data:image/png;base64,page1')
    imageStore.updateCurrentImage({
      translatedDataURL: 'data:image/png;base64,old-render',
      cleanImageData: 'clean-image',
    })

    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 120, 80],
        polygon: [],
        translatedText: '第一页译文',
        fontSize: 22,
        textColor: '#123456',
        fillColor: '#abcdef',
        autoFgColor: [1, 2, 3],
        autoBgColor: [10, 11, 12],
      }),
    ])

    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')
    const { handleAutoTextColorChanged } = useTextStyleSync() as any

    await handleAutoTextColorChanged(true)

    expect(executeRenderMock).toHaveBeenCalledWith(expect.objectContaining({
      renderStylePolicy: {
        fontSize: 'preserve',
        color: 'preserve',
      },
      existingBubbleStates: [
        expect.objectContaining({
          textColor: '#010203',
          fillColor: '#0a0b0c',
        }),
      ],
    }))
    expect(imageStore.currentImage?.bubbleStates?.[0]?.textColor).toBe('#010203')
    expect(imageStore.currentImage?.bubbleStates?.[0]?.fillColor).toBe('#0a0b0c')
  })
})
