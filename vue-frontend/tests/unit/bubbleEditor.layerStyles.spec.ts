import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { createPinia, setActivePinia } from 'pinia'
import BubbleEditor from '@/components/edit/BubbleEditor.vue'
import type { BubbleState } from '@/types/bubble'

const { apiGetMock } = vi.hoisted(() => ({
  apiGetMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: apiGetMock,
  },
}))

function makeBubble(): BubbleState {
  return {
    originalText: 'ぎゅーっ',
    translatedText: '紧握',
    textboxText: '',
    coords: [10, 20, 110, 220],
    polygon: [],
    fontSize: 32,
    fontFamily: 'fonts/STXIHEI.TTF',
    textDirection: 'vertical',
    autoTextDirection: 'vertical',
    textColor: '#000000',
    fillColor: '#FFFFFF',
    rotationAngle: 0,
    position: { x: 0, y: 0 },
    strokeEnabled: false,
    strokeColor: '#FFFFFF',
    strokeWidth: 0,
    lineSpacing: 1.2,
    inlineAlign: 'center',
    blockAlign: 'center',
    inpaintMethod: 'solid',
    textlines: [],
    ocrResult: null,
  }
}

describe('BubbleEditor layer styles in 常用样式', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
    apiGetMock.mockReset()
    apiGetMock.mockImplementation(async (url: string) => {
      if (url === '/api/v2/editor/layer-styles') {
        return {
          groups: [
            {
              name: '漫画拟声词 · 立体特效',
              items: [
                {
                  name: '黑底立体白字（图款·标准）',
                  effectCount: 4,
                  overridesTextColor: true,
                  hollow: false,
                },
              ],
            },
            {
              name: '漫画嵌字 · 常规描边与光晕',
              items: [
                {
                  name: '空心白边字',
                  effectCount: 1,
                  overridesTextColor: false,
                  hollow: true,
                },
              ],
            },
          ],
        }
      }
      return { items: [] }
    })
  })

  it('lists grouped manga layer styles, keeps built-ins undeletable, and emits applyLayerStyle on apply/clear', async () => {
    localStorage.setItem(
      'saber.text-styles.local',
      JSON.stringify([
        {
          name: '对白',
          style: {
            character: { size_pt: 28 },
            box: { strokeEnabled: false },
          },
        },
      ]),
    )

    const wrapper = mount(BubbleEditor, {
      props: { bubble: makeBubble(), bubbleIndex: 0 },
      attachTo: document.body,
    })
    await flushPromises()
    await nextTick()

    const details = wrapper.get('details.bubble-editor__advanced')
    expect(details.attributes('open')).toBeUndefined()
    await details.get('summary').trigger('click')

    const styleTrigger = wrapper.get('button[aria-label="常用样式"]')
    await styleTrigger.trigger('click')
    await nextTick()

    const options = Array.from(document.body.querySelectorAll('.ui-combobox-option'))
    const labels = options.map(node => node.textContent?.trim())
    expect(labels).toEqual([
      '对白',
      '黑底立体白字（图款·标准）',
      '空心白边字',
      '清除图层特效（还原纯文字）',
    ])

    // Select a built-in manga style -> trash disabled, 套用样式 emits applyLayerStyle(name)
    options[1]!.dispatchEvent(new MouseEvent('click', { bubbles: true }))
    await nextTick()
    const deleteBtn = wrapper.get('button[aria-label="删除样式收藏"]')
    expect((deleteBtn.element as HTMLButtonElement).disabled).toBe(true)

    const applyBtn = wrapper.findAll('button').find(btn => btn.text().trim() === '套用样式')!
    expect((applyBtn.element as HTMLButtonElement).disabled).toBe(false)
    await applyBtn.trigger('click')
    expect(wrapper.emitted('applyLayerStyle')).toEqual([['黑底立体白字（图款·标准）']])

    // Select clear option -> emits applyLayerStyle(null)
    await styleTrigger.trigger('click')
    await nextTick()
    const clearOption = Array.from(document.body.querySelectorAll('.ui-combobox-option')).at(-1)!
    clearOption.dispatchEvent(new MouseEvent('click', { bubbles: true }))
    await nextTick()
    await applyBtn.trigger('click')
    expect(wrapper.emitted('applyLayerStyle')?.at(-1)).toEqual([null])

    // Select user favorite -> trash enabled and 套用样式 emits applyStyle
    await styleTrigger.trigger('click')
    await nextTick()
    const favOption = Array.from(document.body.querySelectorAll('.ui-combobox-option'))[0]!
    favOption.dispatchEvent(new MouseEvent('click', { bubbles: true }))
    await nextTick()
    expect((deleteBtn.element as HTMLButtonElement).disabled).toBe(false)
    await applyBtn.trigger('click')
    expect(wrapper.emitted('applyStyle')?.at(-1)?.[0]).toBe('selected')

    // Delete user favorite removes it from storage and resets selection
    await deleteBtn.trigger('click')
    await nextTick()
    expect(JSON.parse(localStorage.getItem('saber.text-styles.local')!)).toEqual([])
    expect((applyBtn.element as HTMLButtonElement).disabled).toBe(true)

    wrapper.unmount()
  })
})
