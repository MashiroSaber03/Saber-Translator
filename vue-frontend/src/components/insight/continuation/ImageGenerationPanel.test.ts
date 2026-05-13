import { mount } from '@vue/test-utils'
import { defineComponent, ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import ImageGenerationPanel from './ImageGenerationPanel.vue'

vi.mock('@/api/continuation', () => ({
  getAvailableImages: vi.fn().mockResolvedValue({
    success: true,
    original_images: [],
    continuation_images: [],
    character_forms: [],
  }),
}))

const stateStub = {
  styleRefPages: ref(3),
  getGeneratedImageUrl: vi.fn((path: string) => path),
}

vi.mock('@/composables/continuation/useContinuationState', () => ({
  useContinuationStateInject: () => stateStub,
}))

const referenceSelectorStub = defineComponent({
  name: 'ReferenceImageSelector',
  template: '<div class="reference-selector-stub" />',
})

function createPage(overrides: Record<string, unknown> = {}) {
  return {
    page_number: 1,
    continuity_text: '上一页剧情很长很长很长很长很长很长很长很长很长很长。',
    story_text: '本页剧情很长很长很长很长很长很长很长很长很长很长。',
    dialogue_text: '二乃：这是一段很长很长很长很长的对白。',
    characters: [],
    character_forms: [],
    final_prompt: '上一页剧情：foo\n本页剧情：bar\n关键对白：baz\n风格约束：保持原作漫画线条、脸型、上色、页面密度和分镜节奏。',
    image_url: '/tmp/page.png',
    previous_url: '',
    status: 'generated',
    ...overrides,
  }
}

describe('ImageGenerationPanel', () => {
  it('keeps a desktop two-column grid with a mobile fallback and fully visible images', () => {
    const filePath = resolve(process.cwd(), 'src/components/insight/continuation/ImageGenerationPanel.vue')
    const source = readFileSync(filePath, 'utf-8')

    expect(source).toContain('grid-template-columns: repeat(2, minmax(0, 1fr));')
    expect(source).toContain('@media (max-width: 1024px)')
    expect(source).toContain('grid-template-columns: 1fr;')
    expect(source).toContain('object-fit: contain;')
  })

  it('shows story sections as collapsed previews by default and expands them independently', async () => {
    const wrapper = mount(ImageGenerationPanel, {
      props: {
        pages: [createPage()],
        isGenerating: false,
        progress: 0,
        bookId: 'book-1',
      },
      global: {
        stubs: {
          ReferenceImageSelector: referenceSelectorStub,
        },
      },
    })

    const previews = wrapper.findAll('.context-text')
    expect(previews).toHaveLength(3)
    expect(previews[0]?.classes()).toContain('is-clamped')
    expect(previews[1]?.classes()).toContain('is-clamped')
    expect(previews[2]?.classes()).toContain('is-clamped')

    const toggleButtons = wrapper.findAll('.context-toggle')
    expect(toggleButtons).toHaveLength(3)
    expect(toggleButtons[0]?.text()).toContain('展开')

    await toggleButtons[0]!.trigger('click')

    expect(wrapper.findAll('.context-text')[0]?.classes()).toContain('is-expanded')
    expect(wrapper.findAll('.context-text')[1]?.classes()).toContain('is-clamped')
    expect(wrapper.findAll('.context-toggle')[0]?.text()).toContain('收起')
  })

  it('keeps the final prompt collapsed until editing and emits prompt updates while editing', async () => {
    const wrapper = mount(ImageGenerationPanel, {
      props: {
        pages: [createPage()],
        isGenerating: false,
        progress: 0,
        bookId: 'book-1',
      },
      global: {
        stubs: {
          ReferenceImageSelector: referenceSelectorStub,
        },
      },
    })

    expect(wrapper.find('.prompt-preview').exists()).toBe(false)
    expect(wrapper.find('.prompt-edit').exists()).toBe(false)
    expect(wrapper.find('.prompt-collapsed-hint').text()).toContain('默认已折叠')

    await wrapper.find('.btn-mini').trigger('click')

    expect(wrapper.find('.prompt-edit').exists()).toBe(true)
    const textarea = wrapper.find('textarea.prompt-input')
    await textarea.setValue('手动修改后的 prompt')

    const emitted = wrapper.emitted('prompt-change')
    expect(emitted?.length).toBeGreaterThan(0)
    expect(emitted?.[0]).toEqual([1])

    await wrapper.find('.btn-mini').trigger('click')
    expect(wrapper.find('.prompt-edit').exists()).toBe(false)
    expect(wrapper.find('.prompt-collapsed-hint').exists()).toBe(true)
  })
})
