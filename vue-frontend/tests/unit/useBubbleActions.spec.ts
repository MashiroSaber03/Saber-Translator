import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { beforeEach, describe, expect, it, vi, afterEach } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
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
    settingsStore.settings.aiVisionOcr.isJsonMode = false
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

    const actions = useBubbleActions()
    await actions.handleOcrRecognize(0)

    expect(ocrSingleBubbleMock).toHaveBeenCalledWith(
      'abc',
      [0, 0, 100, 60],
      'ai_vision',
      expect.objectContaining({
        ai_vision_provider: 'custom',
        ai_vision_ocr_prompt: '对图中的日语进行OCR:',
        use_json_format_for_ai_vision: false,
        ai_vision_prompt_mode: 'paddleocr_vl',
      }),
    )
  })
})
