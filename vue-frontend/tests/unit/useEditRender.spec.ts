import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { createPinia, setActivePinia } from 'pinia'
import type { RenderInput } from '@/composables/translation/core/steps'
import { buildSavedTextStylesFromSettings } from '@/composables/translation/core/runtime'
import { buildEditRenderInput } from '@/composables/edit/editRenderRequest'
import type { BubbleState } from '@/types/bubble'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'
import { createBubbleState } from '@/utils/bubbleFactory'

const { executeRenderMock } = vi.hoisted(() => ({
  executeRenderMock: vi.fn(),
}))

vi.mock('@/composables/translation/core/steps', () => ({
  executeRender: executeRenderMock,
}))

describe('useEditRender', () => {
  it('keeps lifecycle orchestration separate from edit render request projection', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/composables/useEditRender.ts'), 'utf8')

    expect(source).not.toContain('/**\n * 编辑模式渲染组合式函数')
    expect(source).not.toContain('// ============================================================')
    expect(source).not.toContain('bubbleCoords: bubbleStates.map')
    expect(source).toContain('buildEditRenderInput(')
  })

  it('builds edit render input from bubbles and current settings in a pure helper', async () => {
    const settings = createDefaultSettings()
    settings.textStyle.textColor = '#112233'
    settings.textStyle.fillColor = '#ddeeff'

    const textline = {
      polygon: [[0, 0], [40, 0], [40, 12], [0, 12]],
      direction: 'h' as const,
      confidence: 0.98,
    }
    const ocrResult = {
      text: 'OCR text',
      confidence: 0.87,
      confidenceSupported: true,
      engine: 'manga_ocr',
      primaryEngine: 'manga_ocr',
      fallbackUsed: false,
    }
    const bubbleStates = [
      createBubbleState({
        coords: [1.2, 2.6, 20.4, 30.8],
        originalText: '原文一',
        translatedText: '译文一',
        textboxText: '框一',
        rotationAngle: 7,
        textDirection: 'horizontal',
        autoTextDirection: 'vertical',
        textlines: [textline],
        textColor: '',
        fillColor: '',
        ocrResult: null,
      }),
      createBubbleState({
        coords: [50, 60, 100, 130],
        originalText: '原文二',
        translatedText: '译文二',
        textboxText: '框二',
        rotationAngle: 0,
        autoTextDirection: 'horizontal',
        textColor: '#445566',
        fillColor: '#abcdef',
        autoFgColor: [1, 2, 3],
        autoBgColor: [9, 8, 7],
        ocrResult,
      }),
    ]

    const input: RenderInput = buildEditRenderInput({
      imageIndex: 3,
      cleanImage: 'clean-image',
      bubbleStates,
      settings,
    })

    expect(input).toMatchObject({
      imageIndex: 3,
      cleanImage: 'clean-image',
      bubbleCoords: [[1, 3, 20, 31], [50, 60, 100, 130]],
      bubbleAngles: [7, 0],
      autoDirections: ['vertical', 'horizontal'],
      textlinesPerBubble: [[textline], []],
      existingBubbleStates: bubbleStates,
      originalTexts: ['原文一', '原文二'],
      translatedTexts: ['译文一', '译文二'],
      textboxTexts: ['框一', '框二'],
      colors: [
        {
          textColor: '#112233',
          bgColor: '#ddeeff',
          autoFgColor: null,
          autoBgColor: null,
        },
        {
          textColor: '#445566',
          bgColor: '#abcdef',
          autoFgColor: [1, 2, 3],
          autoBgColor: [9, 8, 7],
        },
      ],
      currentMode: 'standard',
      renderStylePolicy: {
        fontSize: 'preserve',
        color: 'preserve',
      },
    })
    expect(input.settingsSnapshot).toBe(settings)
    expect(input.savedTextStyles).toEqual(buildSavedTextStylesFromSettings(settings))
    expect(input.ocrResults).toEqual([
      {
        text: '原文一',
        confidence: null,
        confidenceSupported: false,
        engine: '',
        primaryEngine: '',
        fallbackUsed: false,
      },
      ocrResult,
    ])
  })

  beforeEach(() => {
    setActivePinia(createPinia())
    executeRenderMock.mockReset()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('ignores render results when the user switches to another image before the request completes', async () => {
    let resolveRender!: (value: { finalImage: string; bubbleStates: BubbleState[] }) => void
    const pendingRender = new Promise<{ finalImage: string; bubbleStates: BubbleState[] }>((resolve) => {
      resolveRender = resolve
    })
    executeRenderMock.mockReturnValueOnce(pendingRender)

    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()

    imageStore.addImage('page-1.png', 'data:image/png;base64,page1')
    imageStore.addImage('page-2.png', 'data:image/png;base64,page2')
    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 120, 80],
        polygon: [],
        translatedText: '第一页译文',
      }),
    ])

    const { useEditRender } = await import('@/composables/useEditRender')
    const { reRenderFullImage } = useEditRender()

    const renderPromise = reRenderFullImage()

    imageStore.setCurrentImageIndex(1)
    resolveRender({ finalImage: 'rendered-page-1', bubbleStates: bubbleStore.bubbles })

    await expect(renderPromise).resolves.toBe(false)
    expect(imageStore.images[0]?.translatedDataURL).toBeNull()
    expect(imageStore.images[1]?.translatedDataURL).toBeNull()
  })

  it('uses preserve render policy for ordinary edit rerenders even if page-level auto options remain enabled', async () => {
    executeRenderMock.mockResolvedValue({
      finalImage: 'rendered-page-1',
      bubbleStates: [
        createBubbleState({
          coords: [0, 0, 120, 80],
          polygon: [],
          translatedText: '第一页译文',
          fontSize: 27,
          textColor: '#123456',
          fillColor: '#abcdef',
          autoFgColor: [1, 2, 3],
          autoBgColor: [10, 11, 12],
        }),
      ],
    })

    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const settingsStore = useSettingsStore()

    settingsStore.updateTextStyle({
      autoFontSize: true,
      useAutoTextColor: true,
    })

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
        fontSize: 27,
        textColor: '#123456',
        fillColor: '#abcdef',
        autoFgColor: [1, 2, 3],
        autoBgColor: [10, 11, 12],
      }),
    ])

    const { useEditRender } = await import('@/composables/useEditRender')
    const { reRenderFullImage } = useEditRender()

    await expect(reRenderFullImage()).resolves.toBe(true)
    expect(executeRenderMock).toHaveBeenCalledWith(expect.objectContaining({
      renderStylePolicy: {
        fontSize: 'preserve',
        color: 'preserve',
      },
    }))
  })

  it('does not write render results after the owning component unmounts', async () => {
    let resolveRender!: (value: { finalImage: string; bubbleStates: BubbleState[] }) => void
    const pendingRender = new Promise<{ finalImage: string; bubbleStates: BubbleState[] }>((resolve) => {
      resolveRender = resolve
    })
    executeRenderMock.mockReturnValueOnce(pendingRender)

    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const onRenderSuccess = vi.fn()
    const onRenderEnd = vi.fn()

    imageStore.addImage('page-1.png', 'data:image/png;base64,page1')
    imageStore.updateCurrentImage({ cleanImageData: 'clean-image' })
    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 120, 80],
        polygon: [],
        translatedText: '第一页译文',
      }),
    ])

    const { useEditRender } = await import('@/composables/useEditRender')
    const Harness = defineComponent({
      setup() {
        const { reRenderFullImage } = useEditRender({ onRenderSuccess, onRenderEnd })
        return {
          start: () => reRenderFullImage(),
        }
      },
      render() {
        return h('div')
      },
    })

    const wrapper = mount(Harness)
    const renderPromise = (wrapper.vm as unknown as { start: () => Promise<boolean> }).start()
    wrapper.unmount()

    resolveRender({
      finalImage: 'rendered-after-unmount',
      bubbleStates: bubbleStore.bubbles,
    })

    await expect(renderPromise).resolves.toBe(false)
    expect(imageStore.currentImage?.translatedDataURL).toBeNull()
    expect(onRenderSuccess).not.toHaveBeenCalled()
    expect(onRenderEnd).not.toHaveBeenCalled()
  })

  it('does not write routine console logs during successful edit rerenders', async () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    executeRenderMock.mockResolvedValue({
      finalImage: 'rendered-page-1',
      bubbleStates: [
        createBubbleState({
          coords: [0, 0, 120, 80],
          polygon: [],
          translatedText: '第一页译文',
        }),
      ],
    })

    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,page1')
    imageStore.updateCurrentImage({ cleanImageData: 'clean-image' })
    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 120, 80],
        polygon: [],
        translatedText: '第一页译文',
      }),
    ])

    const { useEditRender } = await import('@/composables/useEditRender')
    const { reRenderFullImage } = useEditRender()

    await expect(reRenderFullImage()).resolves.toBe(true)
    expect(consoleLog).not.toHaveBeenCalled()
  })

  it('does not write routine console logs when rendering the clean background without bubbles', async () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    const onRenderSuccess = vi.fn()
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()

    imageStore.addImage('page-1.png', 'data:image/png;base64,page1')
    imageStore.updateCurrentImage({ cleanImageData: 'clean-image' })
    bubbleStore.clearBubblesLocal()

    const { useEditRender } = await import('@/composables/useEditRender')
    const { reRenderFullImage } = useEditRender({ onRenderSuccess })

    await expect(reRenderFullImage()).resolves.toBe(true)
    expect(executeRenderMock).not.toHaveBeenCalled()
    expect(imageStore.currentImage?.translatedDataURL).toBe('data:image/png;base64,clean-image')
    expect(onRenderSuccess).toHaveBeenCalledWith('data:image/png;base64,clean-image')
    expect(consoleLog).not.toHaveBeenCalled()
  })
})
