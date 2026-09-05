import { mount } from '@vue/test-utils'
import { defineComponent, h, shallowRef } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useImageColorPicker } from '@/composables/edit/useImageColorPicker'
import { createBubbleState } from '@/utils/bubbleFactory'
import { sampleImageColor } from '@/utils/imageColorSampling'
import type { BubbleColorField } from '@/types/bubble'

vi.mock('@/utils/imageColorSampling', () => ({ sampleImageColor: vi.fn() }))
const wrappers: Array<ReturnType<typeof mount>> = []

function setupPicker() {
  const options = {
    pageId: shallowRef<string | undefined>('page-1'),
    bubble: shallowRef(createBubbleState({ backendBubbleId: 'bubble-1' })),
    bubbleIndex: shallowRef(0),
    disabled: shallowRef(false),
    onPick: vi.fn(), onError: vi.fn(),
  }
  let picker!: ReturnType<typeof useImageColorPicker>
  const wrapper = mount(defineComponent({ setup() { picker = useImageColorPicker(options); return () => h('div') } }))
  wrappers.push(wrapper)
  return { picker, options, wrapper }
}

describe('image color picking sessions', () => {
  beforeEach(() => { vi.clearAllMocks(); vi.mocked(sampleImageColor).mockReturnValue('#123456') })
  afterEach(() => { wrappers.splice(0).forEach(wrapper => wrapper.unmount()) })
  const image = document.createElement('img')
  const point = { clientX: 10, clientY: 20 }

  it.each(['textColor', 'fillColor', 'strokeColor'] as BubbleColorField[])('commits %s once and restores normal editing', field => {
    const { picker, options } = setupPicker()
    expect(picker.startColorPick(field)).toBe(true)
    picker.pickImageColor(image, point)
    picker.pickImageColor(image, point)
    expect(options.onPick).toHaveBeenCalledExactlyOnceWith(field, '#123456')
    expect(picker.isPickingColor.value).toBe(false)
  })

  it.each(['page', 'bubble', 'index', 'busy', 'cancel', 'unmount'])('prevents a stale result after %s changes', reason => {
    const { picker, options, wrapper } = setupPicker()
    picker.startColorPick('textColor')
    if (reason === 'page') options.pageId.value = 'page-2'
    if (reason === 'bubble') options.bubble.value = createBubbleState({ backendBubbleId: 'bubble-2' })
    if (reason === 'index') options.bubbleIndex.value = 1
    if (reason === 'busy') options.disabled.value = true
    if (reason === 'cancel') picker.cancelColorPick()
    if (reason === 'unmount') wrapper.unmount()
    expect(picker.isPickingColor.value).toBe(false)
    picker.pickImageColor(image, point)
    expect(options.onPick).not.toHaveBeenCalled()
  })

  it('keeps picking on blank space but restores controls after a read error', () => {
    const { picker, options } = setupPicker()
    picker.startColorPick('fillColor')
    vi.mocked(sampleImageColor).mockReturnValue(null)
    picker.pickImageColor(image, point)
    expect(picker.isPickingColor.value).toBe(true)
    vi.mocked(sampleImageColor).mockImplementation(() => { throw new DOMException('tainted', 'SecurityError') })
    picker.pickImageColor(image, point)
    expect(picker.isPickingColor.value).toBe(false)
    expect(options.onPick).not.toHaveBeenCalled()
    expect(options.onError).toHaveBeenCalledOnce()
  })

  it('does not misreport a failed update as an image read error', () => {
    const { picker, options } = setupPicker()
    options.onPick.mockImplementation(() => { throw new Error('update failed') })
    picker.startColorPick('textColor')
    expect(() => picker.pickImageColor(image, point)).toThrow('update failed')
    expect(picker.isPickingColor.value).toBe(false)
    expect(options.onError).not.toHaveBeenCalled()
  })
})
