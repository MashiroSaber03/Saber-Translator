import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useSettingsStore } from '@/stores/settings'
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

async function expectNoRoutineConsoleLogs(run: () => Promise<void>) {
  const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
  try {
    await run()
    expect(logSpy).not.toHaveBeenCalled()
  } finally {
    logSpy.mockRestore()
  }
}

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
      translatedDataURL: 'data:image/png;base64,existing-render',
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

    await expectNoRoutineConsoleLogs(async () => {
      await handleAutoFontSizeChanged(true)
    })

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
      translatedDataURL: 'data:image/png;base64,existing-render',
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

    await expectNoRoutineConsoleLogs(async () => {
      await handleAutoTextColorChanged(true)
    })

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

  it('style changes and apply-to-all keep normal render orchestration quiet', async () => {
    executeRenderMock.mockResolvedValue({
      finalImage: 'rendered-batch',
      bubbleStates: [
        createBubbleState({
          coords: [0, 0, 120, 80],
          polygon: [],
          translatedText: '批量译文',
          fontSize: 24,
          textColor: '#123456',
          fillColor: '#abcdef',
        }),
      ],
    })

    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const settingsStore = useSettingsStore()
    const firstBubble = createBubbleState({
      coords: [0, 0, 120, 80],
      polygon: [],
      translatedText: '第一页译文',
      fontSize: 22,
      textColor: '#123456',
      fillColor: '#abcdef',
    })
    const secondBubble = createBubbleState({
      coords: [5, 5, 100, 60],
      polygon: [],
      translatedText: '第二页译文',
      fontSize: 20,
      textColor: '#654321',
      fillColor: '#fedcba',
    })

    imageStore.addImage('page-1.png', 'data:image/png;base64,page1', {
      translatedDataURL: 'data:image/png;base64,existing-render-1',
      cleanImageData: 'clean-image-1',
      bubbleStates: [firstBubble],
    })
    imageStore.addImage('page-2.png', 'data:image/png;base64,page2', {
      translatedDataURL: 'data:image/png;base64,existing-render-2',
      cleanImageData: 'clean-image-2',
      bubbleStates: [secondBubble],
    })
    bubbleStore.setBubbles([firstBubble])
    settingsStore.updateTextStyle({ fontSize: 24 })

    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')
    const { handleTextStyleChanged, handleApplyToAll } = useTextStyleSync()

    await expectNoRoutineConsoleLogs(async () => {
      await handleTextStyleChanged('fontSize', 24)
      await handleApplyToAll({
        fontSize: true,
        fontFamily: false,
        layoutDirection: false,
        textColor: false,
        fillColor: false,
        strokeEnabled: false,
        strokeColor: false,
        strokeWidth: false,
        lineSpacing: false,
        textAlign: false,
      })
    })

    expect(showToastMock).toHaveBeenCalledWith('已将 自动字号 应用到 2 张图片', 'success')
    expect(imageStore.images[0]?.bubbleStates?.[0]?.fontSize).toBe(24)
    expect(imageStore.images[1]?.bubbleStates?.[0]?.fontSize).toBe(24)
  })
})
