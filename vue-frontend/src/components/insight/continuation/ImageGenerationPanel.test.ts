import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ImageGenerationPanel from './ImageGenerationPanel.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductDetailSection from '@/components/product/ProductDetailSection.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import { getAvailableImages, loadMoreAvailableCharacterForms } from '@/api/continuation'

const componentSourcePath = resolve(
  process.cwd(),
  'src/components/insight/continuation/ImageGenerationPanel.vue'
)

vi.mock('@/api/continuation', () => ({
  getAvailableImages: vi.fn().mockResolvedValue({
    original_images: [],
    continuation_images: [],
    character_forms: [],
    original_cursor: 0,
    character_forms_cursor: null,
  }),
  loadMoreAvailableCharacterForms: vi.fn(),
}))

const stateStub = {
  styleRefPages: ref(3),
  showMessage: vi.fn(),
}

const referenceSelectorStub = {
  template: '<div class="reference-selector-stub" />',
}

function createPage(overrides: Record<string, unknown> = {}) {
  return {
    page_number: 1,
    continuity_text: '上一页剧情很长很长很长很长很长很长很长很长很长很长。',
    story_text: '本页剧情很长很长很长很长很长很长很长很长很长很长。',
    dialogue_text: '二乃：这是一段很长很长很长很长的对白。',
    characters: [],
    final_prompt:
      '上一页剧情：foo\n本页剧情：bar\n关键对白：baz\n风格约束：保持原作漫画线条、脸型、上色、页面密度和分镜节奏。',
    image_url: '/tmp/page.png',
    previous_url: '',
    status: 'generated',
    ...overrides,
  }
}

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

type AvailableImagesResponse = Awaited<ReturnType<typeof getAvailableImages>>

describe('ImageGenerationPanel', () => {
  beforeEach(() => {
    vi.mocked(getAvailableImages).mockReset()
    vi.mocked(getAvailableImages).mockResolvedValue({
      original_images: [],
      continuation_images: [],
      character_forms: [],
      original_cursor: 0,
      character_forms_cursor: null,
    })
    vi.mocked(loadMoreAvailableCharacterForms).mockReset()
    stateStub.styleRefPages.value = 3
    stateStub.showMessage.mockClear()
  })

  it('uses product form, card, and action-row primitives for generation controls', () => {
    const wrapper = mount(ImageGenerationPanel, {
      props: {
        pages: [
          createPage({ page_number: 1 }),
          createPage({ page_number: 2, status: 'generated', image_url: '' }),
        ],
        isGenerating: false,
        progress: 0,
        bookId: 'book-1',
        state: stateStub,
      },
      global: {
        stubs: {
          ReferenceImageSelector: referenceSelectorStub,
        },
      },
    })

    const fields = wrapper.findAllComponents(UiField)
    expect(
      fields.some(
        field =>
          field.props('label') === '画风参考图数量' &&
          field.props('controlId') === 'continuation-style-reference-count'
      )
    ).toBe(true)

    const numberField = wrapper.getComponent(UiNumberField)
    expect(numberField.props('inputId')).toBe('continuation-style-reference-count')
    expect(numberField.props('min')).toBe(1)
    expect(numberField.props('max')).toBeUndefined()
    expect(numberField.props('modelValue')).toBe(3)

    expect(wrapper.findAllComponents(ProductRecordCard)).toHaveLength(2)
    expect(
      wrapper
        .findAllComponents(ProductChipList)
        .some(
          chips =>
            chips.props('ariaLabel') === '页面 2 生成状态' &&
            chips.props('items')[0]?.label === '待生成'
        )
    ).toBe(true)
    expect(wrapper.findAllComponents(ProductActionRow).length).toBeGreaterThanOrEqual(2)
  })

  it('shows story sections as collapsed previews by default and expands them independently', async () => {
    const wrapper = mount(ImageGenerationPanel, {
      props: {
        pages: [createPage()],
        isGenerating: false,
        progress: 0,
        bookId: 'book-1',
        state: stateStub,
      },
      global: {
        stubs: {
          ReferenceImageSelector: referenceSelectorStub,
        },
      },
    })

    const previews = wrapper.findAll('.image-generation-panel__context-text')
    expect(previews).toHaveLength(3)
    expect(previews[0]?.classes()).toContain('image-generation-panel__context-text--clamped')
    expect(previews[1]?.classes()).toContain('image-generation-panel__context-text--clamped')
    expect(previews[2]?.classes()).toContain('image-generation-panel__context-text--clamped')
    expect(previews[0]?.classes()).toContain('image-generation-panel__context-text--lines-3')
    expect(previews[2]?.classes()).toContain('image-generation-panel__context-text--lines-2')

    const toggleButtons = wrapper
      .findAllComponents(UiButton)
      .filter(button => button.text().includes('展开'))
    expect(toggleButtons).toHaveLength(3)
    expect(toggleButtons.every(button => button.props('variant') === 'link')).toBe(true)
    expect(toggleButtons.every(button => button.props('size') === 'xs')).toBe(true)

    await toggleButtons[0]!.trigger('click')

    expect(wrapper.findAll('.image-generation-panel__context-text')[0]?.classes()).toContain(
      'image-generation-panel__context-text--expanded'
    )
    expect(wrapper.findAll('.image-generation-panel__context-text')[1]?.classes()).toContain(
      'image-generation-panel__context-text--clamped'
    )
    const collapseButton = wrapper
      .findAllComponents(UiButton)
      .find(button => button.text().includes('收起'))
    expect(collapseButton?.props('variant')).toBe('link')

    const source = readFileSync(componentSourcePath, 'utf8')
    expect(source).not.toContain("'is-clamped'")
    expect(source).not.toContain("'is-expanded'")
    expect(source).not.toContain('`lines-${maxLines}`')
    expect(source).not.toContain('class="context-toggle"')
    expect(source).not.toContain('.context-toggle')
    expect(source).not.toContain('variant="toolbar"')
  })

  it('renders story and prompt headings without non-form label markup', () => {
    const wrapper = mount(ImageGenerationPanel, {
      props: {
        pages: [createPage()],
        isGenerating: false,
        progress: 0,
        bookId: 'book-1',
        state: stateStub,
      },
      global: {
        stubs: {
          ReferenceImageSelector: referenceSelectorStub,
        },
      },
    })

    const detailSections = wrapper.findAllComponents(ProductDetailSection)
    expect(detailSections.map(section => section.props('label'))).toEqual([
      '上一页剧情',
      '本页剧情',
      '关键对白',
      '最终生图提示词',
    ])
    expect(
      detailSections.every(section =>
        section.find('.product-detail-section__label-actions').exists()
      )
    ).toBe(true)

    const source = readFileSync(componentSourcePath, 'utf8')
    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('.context-block')
    expect(source).not.toContain('.context-header')
    expect(source).not.toContain('.context-title')
    expect(source).not.toContain('.prompt-header')
    expect(source).not.toContain('.prompt-title')
    expect(source).not.toContain('class="prompt-collapsed"')
    expect(source).not.toContain('.prompt-collapsed {')
    expect(source).not.toContain('.prompt-header label')
  })

  it('keeps the final prompt collapsed until editing and emits prompt updates while editing', async () => {
    const page = createPage()
    const originalPrompt = page.final_prompt
    const wrapper = mount(ImageGenerationPanel, {
      props: {
        pages: [page],
        isGenerating: false,
        progress: 0,
        bookId: 'book-1',
        state: stateStub,
      },
      global: {
        stubs: {
          ReferenceImageSelector: referenceSelectorStub,
        },
      },
    })

    expect(wrapper.find('.prompt-preview').exists()).toBe(false)
    expect(wrapper.find('.image-generation-panel__prompt-edit').exists()).toBe(false)
    expect(wrapper.find('.image-generation-panel__prompt-collapsed-hint').text()).toContain(
      '默认已折叠'
    )

    const promptToggle = wrapper
      .findAllComponents(UiButton)
      .find(button => button.text().includes('编辑'))
    expect(promptToggle?.props('variant')).toBe('secondary')
    expect(promptToggle?.props('size')).toBe('xs')
    const source = readFileSync(componentSourcePath, 'utf8')
    expect(source).not.toContain('btn-mini')

    await promptToggle!.trigger('click')

    expect(wrapper.find('.image-generation-panel__prompt-edit').exists()).toBe(true)
    const promptTextarea = wrapper.getComponent(UiTextarea)
    expect(promptTextarea.props('variant')).toBe('panel')
    expect(promptTextarea.props('size')).toBe('md')
    const textarea = wrapper.find('textarea.image-generation-panel__prompt-input')
    await textarea.setValue('手动修改后的 prompt')

    const emitted = wrapper.emitted('prompt-change')
    expect(emitted?.length).toBeGreaterThan(0)
    expect(emitted?.[0]).toEqual([1, '手动修改后的 prompt'])
    expect(page.final_prompt).toBe(originalPrompt)

    await getButtonByText(wrapper, '收起').trigger('click')
    expect(wrapper.find('.image-generation-panel__prompt-edit').exists()).toBe(false)
    expect(wrapper.find('.image-generation-panel__prompt-collapsed-hint').exists()).toBe(true)
  })

  it('renders missing final prompts through product status feedback', () => {
    const wrapper = mount(ImageGenerationPanel, {
      props: {
        pages: [createPage({ final_prompt: '' })],
        isGenerating: false,
        progress: 0,
        bookId: 'book-1',
        state: stateStub,
      },
      global: {
        stubs: {
          ReferenceImageSelector: referenceSelectorStub,
        },
      },
    })

    const source = readFileSync(componentSourcePath, 'utf8')
    expect(source).toContain('ProductStatusBanner')
    expect(source).not.toContain('prompt-empty')

    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props()).toMatchObject({
      iconName: 'message',
      role: 'note',
      tone: 'neutral',
      title: '暂无最终提示词',
    })
  })

  it('exposes form and progress semantics for generation controls', () => {
    const wrapper = mount(ImageGenerationPanel, {
      props: {
        pages: [createPage()],
        isGenerating: true,
        progress: 42,
        bookId: 'book-1',
        state: stateStub,
      },
      global: {
        stubs: {
          ReferenceImageSelector: referenceSelectorStub,
        },
      },
    })

    expect(wrapper.find('label[for="continuation-style-reference-count"]').exists()).toBe(true)
    expect(wrapper.find('input#continuation-style-reference-count').exists()).toBe(true)

    const progressbar = wrapper.get('[role="progressbar"]')
    expect(progressbar.attributes('aria-label')).toBe('图片生成进度')
    expect(progressbar.attributes('aria-valuemin')).toBe('0')
    expect(progressbar.attributes('aria-valuemax')).toBe('100')
    expect(progressbar.attributes('aria-valuenow')).toBe('42')
    expect(wrapper.getComponent(UiProgressBar).props('value')).toBe(42)
  })

  it('maps local preview and focus roles through semantic tokens', () => {
    const source = readFileSync(componentSourcePath, 'utf8')
    const rootStyle = source.match(/\.image-generation-panel \{(?<body>[\s\S]*?)\n\}/)

    expect(source).not.toContain('--image-generation-panel-empty-preview-background: #f7f7f7')
    expect(source).not.toContain('--image-generation-panel-focus-ring: rgba(99, 102, 241, .25)')
    expect(source).toContain(
      '--image-generation-panel-empty-preview-background: var(--color-surface-muted)'
    )
    expect(source).not.toContain('ref-count-input')
    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-(?:input|textarea)-/)
  })

  it('sizes generated image cards from the continuation panel container', () => {
    const source = readFileSync(componentSourcePath, 'utf8')

    expect(source).toContain('container: continuation-image-generation / inline-size')
    expect(source).toContain('repeat(auto-fit, minmax(min(100%, 360px), 1fr))')
    expect(source).toContain('@container continuation-image-generation')
    expect(source).not.toContain('grid-template-columns: repeat(2, minmax(0, 1fr))')
    expect(source).not.toContain('@media (--breakpoint-xl-down)')
  })

  it('wraps long generated story and prompt text inside narrow continuation panels', () => {
    const source = readFileSync(componentSourcePath, 'utf8')
    const contextStyle =
      source.match(/\.image-generation-panel__context-text \{(?<body>[\s\S]*?)\n\}/)?.groups
        ?.body ?? ''
    const promptStyle =
      source
        .match(/\.image-generation-panel__prompt-(?:input|collapsed-hint) \{(?<body>[\s\S]*?)\n\}/g)
        ?.join('\n') ?? ''

    expect(contextStyle).toContain('overflow-wrap: anywhere')
    expect(promptStyle).toContain('overflow-wrap: anywhere')
  })

  it('keeps local image-generation hooks owner-prefixed', () => {
    const source = readFileSync(componentSourcePath, 'utf8')

    expect(source).toContain('image-generation-panel__controls')
    expect(source).toContain('image-generation-panel__reference-row')
    expect(source).toContain('image-generation-panel__reference-count-field')
    expect(source).toContain('image-generation-panel__image-card')
    expect(source).toContain('image-generation-panel__image-title')
    expect(source).toContain('image-generation-panel__image')
    expect(source).toContain('image-generation-panel__empty-preview-text')
    expect(source).toContain('image-generation-panel__progress-text')
    expect(source).not.toContain('class="generation-controls"')
    expect(source).not.toContain('class="config-row"')
    expect(source).not.toContain('class="ref-count-field"')
    expect(source).not.toContain('class="image-card"')
    expect(source).not.toContain('class="progress-text"')
    expect(source).not.toContain('.image-generation-panel__image-header h4')
    expect(source).not.toContain('.image-generation-panel__preview img')
    expect(source).not.toContain('.image-generation-panel__empty-preview p')
  })

  it('ignores stale reference-image responses after the book changes', async () => {
    const bookOneImages = createDeferred<AvailableImagesResponse>()
    const bookTwoImages = createDeferred<AvailableImagesResponse>()
    vi.mocked(getAvailableImages)
      .mockReturnValueOnce(bookOneImages.promise)
      .mockReturnValueOnce(bookTwoImages.promise)

    const selectorWithImagesStub = {
      props: {
        originalImages: {
          type: Array,
          default: () => [],
        },
      },
      template:
        '<div class="reference-selector-stub">{{ originalImages.map(image => image.token).join(",") }}</div>',
    }

    const wrapper = mount(ImageGenerationPanel, {
      props: {
        pages: [createPage()],
        isGenerating: false,
        progress: 0,
        bookId: 'book-1',
        state: stateStub,
      },
      global: {
        stubs: {
          ReferenceImageSelector: selectorWithImagesStub,
        },
      },
    })

    await getButtonByText(wrapper, '选择初始参考图').trigger('click')
    await wrapper.setProps({ bookId: 'book-2' })
    await getButtonByText(wrapper, '选择初始参考图').trigger('click')

    bookTwoImages.resolve({
      original_images: [
        { page_number: 2, path: '/tmp/book-2.png', has_image: true, token: 'book-2-token' },
      ],
      continuation_images: [],
      character_forms: [],
      original_cursor: 0,
      character_forms_cursor: null,
    })
    await Promise.resolve()
    await wrapper.vm.$nextTick()

    expect(wrapper.find('.reference-selector-stub').text()).toBe('book-2-token')

    bookOneImages.resolve({
      original_images: [
        { page_number: 1, path: '/tmp/book-1.png', has_image: true, token: 'book-1-token' },
      ],
      continuation_images: [],
      character_forms: [],
      original_cursor: 0,
      character_forms_cursor: null,
    })
    await Promise.resolve()
    await wrapper.vm.$nextTick()

    expect(wrapper.find('.reference-selector-stub').text()).toBe('book-2-token')
  })

  it('merges cursor-paginated character forms without replacing the first page', async () => {
    vi.mocked(getAvailableImages).mockResolvedValue({
      original_images: [],
      continuation_images: [],
      character_forms: [
        {
          token: 'form-token-1',
          character_name: '主角',
          form_id: 'form-1',
          form_name: '常服',
          path: '/form-1.png',
          has_image: true,
        },
      ],
      original_cursor: 0,
      character_forms_cursor: 100,
    })
    vi.mocked(loadMoreAvailableCharacterForms).mockResolvedValue({
      character_forms: [
        {
          token: 'form-token-2',
          character_name: '主角',
          form_id: 'form-2',
          form_name: '战斗服',
          path: '/form-2.png',
          has_image: true,
        },
      ],
      next_cursor: null,
    })
    const selectorStub = {
      props: ['characterForms', 'hasMoreCharacterForms'],
      emits: ['load-more-character-forms'],
      template: `
        <div class="reference-selector-stub">
          <span class="form-tokens">{{ characterForms.map(form => form.token).join(',') }}</span>
          <button type="button" @click="$emit('load-more-character-forms')">more</button>
        </div>
      `,
    }
    const wrapper = mount(ImageGenerationPanel, {
      props: {
        pages: [createPage()],
        isGenerating: false,
        progress: 0,
        bookId: 'book-1',
        state: stateStub,
      },
      global: { stubs: { ReferenceImageSelector: selectorStub } },
    })

    await getButtonByText(wrapper, '选择初始参考图').trigger('click')
    await Promise.resolve()
    await wrapper.vm.$nextTick()
    await wrapper.get('.reference-selector-stub button').trigger('click')
    await Promise.resolve()
    await wrapper.vm.$nextTick()

    expect(loadMoreAvailableCharacterForms).toHaveBeenCalledWith('book-1', 100)
    expect(wrapper.get('.form-tokens').text()).toBe('form-token-1,form-token-2')
  })
})
