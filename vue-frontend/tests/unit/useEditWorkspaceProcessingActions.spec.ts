import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, ref } from 'vue'
import { mount } from '@vue/test-utils'
import { storeToRefs } from 'pinia'
import { createPinia, setActivePinia } from 'pinia'
import { useEditWorkspaceProcessingActions } from '@/composables/edit/useEditWorkspaceProcessingActions'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { createBubbleState } from '@/utils/bubbleFactory'

const {
  translateSingleTextMock,
  executeDetectionMock,
  saveDetectionResultToImageMock,
  translateWithCurrentBubblesMock,
  confirmProductActionMock,
  showToastMock,
} = vi.hoisted(() => ({
  translateSingleTextMock: vi.fn(),
  executeDetectionMock: vi.fn(),
  saveDetectionResultToImageMock: vi.fn(),
  translateWithCurrentBubblesMock: vi.fn(),
  confirmProductActionMock: vi.fn(),
  showToastMock: vi.fn(),
}))

vi.mock('@/api/translate', () => ({
  translateSingleText: translateSingleTextMock,
}))

vi.mock('@/composables/translation/core/steps', () => ({
  executeDetection: executeDetectionMock,
}))

vi.mock('@/composables/translation/core/detectionResultWriter', () => ({
  saveDetectionResultToImage: saveDetectionResultToImageMock,
}))

vi.mock('@/composables/useTranslationPipeline', () => ({
  useTranslation: () => ({
    translateWithCurrentBubbles: translateWithCurrentBubblesMock,
  }),
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

describe('useEditWorkspaceProcessingActions', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    translateSingleTextMock.mockResolvedValue({
      success: true,
      data: {
        translated_text: 'translated bubble',
        warnings: [],
      },
    })
    confirmProductActionMock.mockReset()
    confirmProductActionMock.mockResolvedValue(true)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('retranslates the selected bubble without routine console logs', async () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const settingsStore = useSettingsStore()
    const reRenderFullImage = vi.fn(async () => true)

    settingsStore.settings.translation.provider = 'custom'
    settingsStore.settings.translation.apiKey = 'key'
    settingsStore.settings.translation.modelName = 'model'
    settingsStore.settings.translation.openaiOptions.request.forceJsonOutput = false

    imageStore.addImage('page.png', 'data:image/png;base64,page')
    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 120, 80],
        polygon: [],
        originalText: '原文',
      }),
    ])

    const { bubbles } = storeToRefs(bubbleStore)
    const actions = useEditWorkspaceProcessingActions({
      images: ref(imageStore.images),
      currentImage: ref(imageStore.currentImage),
      currentImageIndex: ref(imageStore.currentImageIndex),
      bubbles,
      reRenderFullImage,
      loadBubbleStatesFromImage: vi.fn(),
      selectFirstBubbleIfExists: vi.fn(),
    })

    await actions.handleReTranslateBubble(0)

    expect(translateSingleTextMock).toHaveBeenCalledWith(expect.objectContaining({
      original_text: '原文',
      model_provider: 'custom',
    }))
    expect(bubbleStore.bubbles[0]?.translatedText).toBe('translated bubble')
    expect(reRenderFullImage).toHaveBeenCalled()
    expect(consoleLog).not.toHaveBeenCalled()
  })

  it('clears the batch detection completion timer when the owner unmounts', async () => {
    vi.useFakeTimers()
    const windowConfirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    executeDetectionMock.mockResolvedValue({
      bubbleCoords: [],
      bubbleStates: [],
    })

    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,page1')
    imageStore.addImage('page-2.png', 'data:image/png;base64,page2')

    const { bubbles } = storeToRefs(bubbleStore)
    let actions!: ReturnType<typeof useEditWorkspaceProcessingActions>

    const Harness = defineComponent({
      setup() {
        actions = useEditWorkspaceProcessingActions({
          images: ref(imageStore.images),
          currentImage: ref(imageStore.currentImage),
          currentImageIndex: ref(imageStore.currentImageIndex),
          bubbles,
          reRenderFullImage: vi.fn(async () => true),
          loadBubbleStatesFromImage: vi.fn(),
          selectFirstBubbleIfExists: vi.fn(),
        })
        return () => null
      },
    })

    const wrapper = mount(Harness)

    await actions.detectAllImages()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '批量检测文本框',
      message: '此操作将对所有图片进行文本框检测，可能会覆盖已有的检测结果。确定继续吗？',
      confirmText: '开始检测',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(windowConfirm).not.toHaveBeenCalled()
    expect(actions.isProcessing.value).toBe(true)

    wrapper.unmount()
    expect(actions.isProcessing.value).toBe(false)

    await vi.advanceTimersByTimeAsync(2000)
    expect(actions.isProcessing.value).toBe(false)
  })
})
