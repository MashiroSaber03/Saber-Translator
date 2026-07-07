import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { nextTick } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import ScriptGenerationPanel from './ScriptGenerationPanel.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import { getAvailableImages } from '@/api/continuation'

const componentSourcePath = resolve(process.cwd(), 'src/components/insight/continuation/ScriptGenerationPanel.vue')

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
  it('uses product form and action-row primitives for script controls', () => {
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

    const referenceField = wrapper.getComponent(UiField)
    expect(referenceField.props('label')).toBe('VLM参考图数')
    expect(referenceField.props('controlId')).toBe('script-reference-count')

    const numberField = wrapper.getComponent(UiNumberField)
    expect(numberField.props('inputId')).toBe('script-reference-count')
    expect(numberField.props('min')).toBe(1)
    expect(numberField.props('max')).toBe(10)
    expect(numberField.props('modelValue')).toBe(5)

    const rows = wrapper.findAllComponents(ProductActionRow)
    expect(rows.some(row => row.props('ariaLabel') === '续写脚本编辑操作')).toBe(true)
    expect(rows.some(row => row.props('ariaLabel') === '脚本参考图操作')).toBe(true)
    expect(wrapper.getComponent(UiTextarea).props()).toMatchObject({
      variant: 'panel',
      size: 'lg',
    })

    const source = readFileSync(componentSourcePath, 'utf8')
    const rootStyle = source.match(/\.script-generation-panel \{(?<body>[\s\S]*?)\n\}/)
    expect(source).not.toContain('ref-count-input')
    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-(?:input|textarea)-/)
  })

  it('does not override shared button primitive variables at the panel root', () => {
    const source = readFileSync(componentSourcePath, 'utf8')
    const rootStyle = source.match(/\.script-generation-panel \{(?<body>[\s\S]*?)\n\}/)

    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-button-/)
  })

  it('keeps the reference controls responsive in narrow continuation panels', () => {
    const source = readFileSync(componentSourcePath, 'utf8')
    const configRowStyle = source.match(/\.script-generation-panel__reference-row \{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''
    const refCountStyle = source.match(/\.script-generation-panel__reference-count-field \{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''

    expect(configRowStyle).toContain('flex-wrap: wrap')
    expect(refCountStyle).not.toMatch(/\bwidth:\s*150px/)
    expect(refCountStyle).toContain('flex:')
  })

  it('keeps generated script metadata responsive in narrow continuation panels', () => {
    const source = readFileSync(componentSourcePath, 'utf8')
    const headerStyle = source.match(/\.script-generation-panel__header \{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''
    const titleStyle = source.match(/\.script-generation-panel__title \{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''
    const metaStyle = source.match(/\.script-generation-panel__meta \{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''

    expect(headerStyle).toContain('flex-wrap: wrap')
    expect(headerStyle).toContain('min-width: 0')
    expect(source).toContain('class="script-generation-panel__title"')
    expect(source).not.toContain('.script-generation-panel__header h4')
    expect(titleStyle).toContain('min-width: 0')
    expect(titleStyle).toContain('overflow-wrap: anywhere')
    expect(metaStyle).toContain('min-width: 0')
  })

  it('keeps local script-generation hooks owner-prefixed', () => {
    const source = readFileSync(componentSourcePath, 'utf8')

    expect(source).toContain('class="script-generation-panel"')
    expect(source).toContain('script-generation-panel__reference-row')
    expect(source).toContain('script-generation-panel__reference-count-field')
    expect(source).toContain('script-generation-panel__title')
    expect(source).toContain('script-generation-panel__textarea')
    expect(source).not.toContain('class="script-panel"')
    expect(source).not.toContain('class="config-row"')
    expect(source).not.toContain('class="ref-count-field"')
    expect(source).not.toContain('script-textarea')
  })

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

    const textarea = wrapper.find('textarea.script-generation-panel__textarea')
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

  it('renders the empty script state through product status feedback', () => {
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

    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props('tone')).toBe('neutral')
    expect(banner.props('role')).toBe('note')
    expect(banner.props('iconName')).toBe('file-text')
    expect(wrapper.text()).toContain('点击下方按钮生成续写脚本')
    expect(wrapper.find('.no-script').exists()).toBe(false)
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
