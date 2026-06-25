import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { beforeEach, describe, expect, it, vi, afterEach } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { createBubbleState } from '@/utils/bubbleFactory'
import { useBubbleActions } from '@/composables/useBubbleActions'

const { ocrSingleBubbleMock } = vi.hoisted(() => ({
  ocrSingleBubbleMock: vi.fn(async () => ({
    success: true,
    text: '识别结果',
    textlines: [],
    ocr_result: null,
  })),
}))

vi.mock('@/api/translate', () => ({
  ocrSingleBubble: ocrSingleBubbleMock,
  inpaintSingleBubble: vi.fn(),
}))

vi.mock('@/utils/toast', () => ({
  showToast: vi.fn(),
}))

describe('useBubbleActions', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
    ocrSingleBubbleMock.mockClear()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('runs one trailing preview render after updates arrive during an in-flight render', async () => {
    let resolveFirstRender!: () => void
    const onDelayedPreview = vi
      .fn<() => Promise<void>>()
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            resolveFirstRender = resolve
          })
      )
      .mockResolvedValueOnce()

    const Harness = defineComponent({
      setup() {
        const bubbleStore = useBubbleStore()
        bubbleStore.setBubbles([
          createBubbleState({
            coords: [0, 0, 100, 60],
            polygon: [],
          }),
        ])
        bubbleStore.selectBubble(0)

        const actions = useBubbleActions({ onDelayedPreview })

        return {
          ...actions,
          triggerUpdate(text: string) {
            actions.handleBubbleUpdate({ translatedText: text })
          },
        }
      },
      render() {
        return h('div')
      },
    })

    const wrapper = mount(Harness)

    ;(wrapper.vm as unknown as { triggerUpdate: (text: string) => void }).triggerUpdate('第一次更新')
    await vi.advanceTimersByTimeAsync(150)
    expect(onDelayedPreview).toHaveBeenCalledTimes(1)

    ;(wrapper.vm as unknown as { triggerUpdate: (text: string) => void }).triggerUpdate('第二次更新')
    await vi.advanceTimersByTimeAsync(150)
    expect(onDelayedPreview).toHaveBeenCalledTimes(1)

    resolveFirstRender()
    await flushPromises()
    await vi.runOnlyPendingTimersAsync()
    await flushPromises()

    expect(onDelayedPreview).toHaveBeenCalledTimes(2)
  })

  it('propagates AI vision prompt mode and exact custom prompt for single-bubble OCR', async () => {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const settingsStore = useSettingsStore()

    settingsStore.settings.ocrEngine = 'ai_vision'
    settingsStore.settings.sourceLanguage = 'japanese'
    settingsStore.settings.aiVisionOcr.provider = 'custom'
    settingsStore.settings.aiVisionOcr.apiKey = 'vision-key'
    settingsStore.settings.aiVisionOcr.modelName = 'vision-model'
    settingsStore.settings.aiVisionOcr.prompt = '对图中的日语进行OCR:'
    settingsStore.settings.aiVisionOcr.promptMode = 'paddleocr_vl'
    settingsStore.settings.aiVisionOcr.openaiOptions.request.forceJsonOutput = false
    settingsStore.settings.aiVisionOcr.customBaseUrl = 'https://example.com/v1'

    imageStore.setImages([
      {
        id: 'img-1',
        fileName: 'test.png',
        originalDataURL: 'data:image/png;base64,abc',
        translatedDataURL: null,
        cleanImageData: null,
        bubbleStates: null,
        translationStatus: 'pending',
        translationFailed: false,
        hasUnsavedChanges: false,
      } as any,
    ])
    imageStore.setCurrentImageIndex(0)

    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 100, 60],
        polygon: [],
        textlines: [],
      }),
    ])

    const Harness = defineComponent({
      setup() {
        const actions = useBubbleActions()
        return { ...actions }
      },
      render() {
        return h('div')
      },
    })

    const wrapper = mount(Harness)
    await (wrapper.vm as unknown as { handleOcrRecognize: (index: number) => Promise<void> }).handleOcrRecognize(0)

    expect(ocrSingleBubbleMock).toHaveBeenCalledWith(
      'abc',
      [0, 0, 100, 60],
      'ai_vision',
      expect.objectContaining({
        ai_vision_provider: 'custom',
        ai_vision_ocr_prompt: '对图中的日语进行OCR:',
        ai_vision_prompt_mode: 'paddleocr_vl',
        openai_options: expect.objectContaining({
          execution: expect.objectContaining({
            transport_retries: 1,
            business_retries: 3,
          }),
        }),
      }),
    )
  })

  it('keeps drawing rectangle style limited to geometry so owner CSS controls visuals', () => {
    const Harness = defineComponent({
      setup() {
        const actions = useBubbleActions()
        actions.currentDrawingRect.value = [10, 20, 40, 70]
        return { ...actions }
      },
      render() {
        return h('div')
      },
    })

    const wrapper = mount(Harness)
    const actions = wrapper.vm as unknown as ReturnType<typeof useBubbleActions>
    const style = actions.getDrawingRectStyle()

    expect(style).toEqual({
      position: 'absolute',
      left: '10px',
      top: '20px',
      width: '30px',
      height: '50px',
    })
  })

  it('does not write routine console logs during common bubble interactions', async () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const onReRender = vi.fn()
    const onDelayedPreview = vi.fn()

    imageStore.setImages([
      {
        id: 'img-1',
        fileName: 'test.png',
        originalDataURL: 'data:image/png;base64,abc',
        translatedDataURL: null,
        cleanImageData: null,
        bubbleStates: null,
        translationStatus: 'pending',
        translationFailed: false,
        hasUnsavedChanges: false,
      } as any,
    ])
    imageStore.setCurrentImageIndex(0)
    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 100, 60],
        polygon: [],
        textlines: [],
      }),
      createBubbleState({
        coords: [20, 20, 140, 100],
        polygon: [],
      }),
    ])
    bubbleStore.selectBubble(0)

    const Harness = defineComponent({
      setup() {
        return {
          ...useBubbleActions({ onReRender, onDelayedPreview }),
        }
      },
      render() {
        return h('div')
      },
    })

    const wrapper = mount(Harness)
    const actions = wrapper.vm as unknown as ReturnType<typeof useBubbleActions>

    actions.handleBubbleDragStart(0, new MouseEvent('mousedown'))
    actions.handleBubbleDragEnd(0, [1, 2, 101, 62])
    actions.handleBubbleResizeStart(0, 'se', new MouseEvent('mousedown'))
    actions.handleBubbleResizeEnd(0, [1, 2, 120, 80])
    actions.handleBubbleRotateStart(0, new MouseEvent('mousedown'))
    actions.handleBubbleRotateEnd(0, 12)
    actions.toggleDrawingMode()
    actions.toggleDrawingMode()
    actions.handleDrawBubble([30, 30, 120, 120])
    actions.handleBubbleUpdate({ translatedText: 'updated' })
    await actions.handleOcrRecognize(0)
    actions.deleteSelectedBubbles()

    await vi.runOnlyPendingTimersAsync()
    await flushPromises()

    expect(onReRender).toHaveBeenCalled()
    expect(onDelayedPreview).toHaveBeenCalled()
    expect(ocrSingleBubbleMock).toHaveBeenCalled()
    expect(consoleLog).not.toHaveBeenCalled()
  })
})
