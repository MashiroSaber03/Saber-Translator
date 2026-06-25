import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import ScriptGenerationPanel from './ScriptGenerationPanel.vue'
import { getAvailableImages } from '@/api/continuation'

vi.mock('@/api/continuation', () => ({
  getAvailableImages: vi.fn().mockResolvedValue({
    success: true,
    original_images: [
      { page_number: 1, path: '/tmp/page-1.png', has_image: true, token: 'original:1' },
    ],
  }),
}))

function getButtonByText(wrapper: ReturnType<typeof mount>, text: string) {
  const button = wrapper.findAll('button').find(node => node.text().includes(text))
  expect(button).toBeTruthy()
  return button!
}

function createDeferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(resolver => {
    resolve = resolver
  })
  return { promise, resolve }
}

describe('ScriptGenerationPanel', () => {
  it('emits script updates and includes reference count when generating', async () => {
    const wrapper = mount(ScriptGenerationPanel, {
      props: {
        script: {
          chapter_title: '测试章节',
          page_count: 10,
          script_text: '旧脚本',
          generated_at: '2026-05-11T00:00:00',
        },
        isGenerating: false,
        bookId: 'book-1',
      },
      global: {
        stubs: {
          ReferenceImageSelector: {
            template: '<div />',
          },
        },
      },
    })

    const textarea = wrapper.find('textarea.script-textarea')
    await textarea.setValue('新脚本内容')

    const updateEvents = wrapper.emitted('update-script') || []
    expect(updateEvents[updateEvents.length - 1]).toEqual(['新脚本内容'])

    await getButtonByText(wrapper, '生成脚本').trigger('click')
    await nextTick()

    const generateEvents = wrapper.emitted('generate') || []
    expect(generateEvents[generateEvents.length - 1]).toEqual([
      {
        referenceTokens: null,
        referenceImageCount: 5,
      },
    ])
  })

  it('associates the reference-count label with the numeric input', () => {
    const wrapper = mount(ScriptGenerationPanel, {
      props: {
        script: null,
        isGenerating: false,
        bookId: 'book-1',
      },
      global: {
        stubs: {
          ReferenceImageSelector: {
            template: '<div />',
          },
        },
      },
    })

    expect(wrapper.find('label[for="script-reference-count"]').exists()).toBe(true)
    expect(wrapper.find('input#script-reference-count').exists()).toBe(true)
  })

  it('clears stale manual reference selections when the workflow is reset', async () => {
    const selectorStub = {
      template: '<button class="selector-confirm" @click="$emit(\'confirm\', [\'original:1\'])">选择参考图</button>',
    }

    const wrapper = mount(ScriptGenerationPanel, {
      props: {
        script: {
          chapter_title: '测试章节',
          page_count: 10,
          script_text: '旧脚本',
          generated_at: '2026-05-11T00:00:00',
        },
        isGenerating: false,
        bookId: 'book-1',
      },
      global: {
        stubs: {
          ReferenceImageSelector: selectorStub,
        },
      },
    })

    await wrapper.find('.selector-confirm').trigger('click')
    await nextTick()

    await wrapper.setProps({ script: null })
    await nextTick()

    await getButtonByText(wrapper, '生成脚本').trigger('click')

    const generateEvents = wrapper.emitted('generate') || []
    expect(generateEvents[generateEvents.length - 1]).toEqual([
      {
        referenceTokens: null,
        referenceImageCount: 5,
      },
    ])
  })

  it('ignores stale reference-image responses after the book changes', async () => {
    const bookOneImages = createDeferred<{
      success: boolean
      original_images: Array<{ page_number: number; path: string; has_image: boolean; token: string }>
    }>()
    const bookTwoImages = createDeferred<{
      success: boolean
      original_images: Array<{ page_number: number; path: string; has_image: boolean; token: string }>
    }>()
    vi.mocked(getAvailableImages)
      .mockReturnValueOnce(bookOneImages.promise)
      .mockReturnValueOnce(bookTwoImages.promise)

    const wrapper = mount(ScriptGenerationPanel, {
      props: {
        script: null,
        isGenerating: false,
        bookId: 'book-1',
      },
      global: {
        stubs: {
          ReferenceImageSelector: {
            props: ['originalImages'],
            template: '<div class="selector-state">{{ originalImages.map(image => image.token).join(",") }}</div>',
          },
        },
      },
    })

    await wrapper.setProps({ bookId: 'book-2' })

    bookTwoImages.resolve({
      success: true,
      original_images: [
        { page_number: 2, path: '/tmp/book-2.png', has_image: true, token: 'book-2-token' },
      ],
    })
    await nextTick()
    await Promise.resolve()

    expect(wrapper.find('.selector-state').text()).toBe('book-2-token')

    bookOneImages.resolve({
      success: true,
      original_images: [
        { page_number: 1, path: '/tmp/book-1.png', has_image: true, token: 'book-1-token' },
      ],
    })
    await nextTick()
    await Promise.resolve()

    expect(wrapper.find('.selector-state').text()).toBe('book-2-token')
  })
})
