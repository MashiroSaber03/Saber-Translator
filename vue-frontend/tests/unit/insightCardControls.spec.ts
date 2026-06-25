import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

const {
  deletePromptFromLibraryMock,
  getDefaultPromptsMock,
  getPromptsLibraryMock,
} = vi.hoisted(() => ({
  deletePromptFromLibraryMock: vi.fn(),
  getDefaultPromptsMock: vi.fn(),
  getPromptsLibraryMock: vi.fn(),
}))

vi.mock('@/api/insight', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/insight')>()
  return {
    ...actual,
    __v_isRef: false,
    PROMPT_METADATA: {
      batch_analysis: { label: '批量分析', hint: '批量分析提示词' },
    },
    deletePromptFromLibrary: deletePromptFromLibraryMock,
    getDefaultPrompts: getDefaultPromptsMock,
    getPromptsLibrary: getPromptsLibraryMock,
    importPromptsLibrary: vi.fn(),
    savePromptToLibrary: vi.fn(),
  }
})

vi.mock('@/api/continuation', () => ({
  exportAsImages: vi.fn(),
  exportAsPdf: vi.fn(),
}))

import ExportPanel from '@/components/insight/continuation/ExportPanel.vue'
import PromptsSettingsTab from '@/components/insight/settings/PromptsSettingsTab.vue'
import ChatComposer from '@/components/insight/studio/preview/ChatComposer.vue'

describe('Insight card-like controls', () => {
  beforeEach(() => {
    const pinia = createPinia()
    setActivePinia(pinia)
    deletePromptFromLibraryMock.mockReset()
    getDefaultPromptsMock.mockReset()
    getPromptsLibraryMock.mockReset()
    getDefaultPromptsMock.mockResolvedValue({
      success: true,
      prompts: {
        batch_analysis: '默认提示词',
        chapter_summary: '',
        qa_response: '',
        segment_summary: '',
      },
    })
    getPromptsLibraryMock.mockResolvedValue({
      success: true,
      library: [
        {
          id: 'prompt-1',
          name: '战斗分析',
          type: 'batch_analysis',
          content: '分析战斗场面',
          created_at: '2026-05-21T10:00:00Z',
        },
      ],
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('uses separate controls for loading and deleting saved prompts', async () => {
    const wrapper = mount(PromptsSettingsTab, {
      global: {
        plugins: [createPinia()],
      },
    })
    await flushPromises()

    const promptRow = wrapper.find('.saved-prompt-item')
    expect(promptRow.element.tagName).toBe('DIV')

    const loadButton = wrapper.find('.saved-prompt-load')
    expect(loadButton.element.tagName).toBe('BUTTON')
    expect(loadButton.attributes('aria-label')).toBe('加载提示词：战斗分析')

    await loadButton.trigger('click')
    expect(wrapper.emitted('showMessage')?.[0]).toEqual(['已加载提示词: 战斗分析', 'success'])

    const deleteButton = wrapper.find('.saved-prompt-delete')
    expect(deleteButton.element.tagName).toBe('BUTTON')
    expect(deleteButton.attributes('aria-label')).toBe('删除提示词：战斗分析')
  })

  it('uses a passive pending attachment card with an explicit remove button', async () => {
    const createObjectURLSpy = vi
      .spyOn(URL, 'createObjectURL')
      .mockReturnValue('blob:attachment-preview')
    const revokeObjectURLSpy = vi
      .spyOn(URL, 'revokeObjectURL')
      .mockImplementation(() => {})
    const wrapper = mount(ChatComposer, {
      props: {
        chatStreaming: false,
      },
    })
    const file = new File(['image'], 'scene.png', { type: 'image/png' })
    const input = wrapper.find('input[type="file"]')
    Object.defineProperty(input.element, 'files', {
      value: [file],
      configurable: true,
    })

    await input.trigger('change')

    expect(createObjectURLSpy).toHaveBeenCalledWith(file)
    expect(wrapper.find('.pending-image-card').element.tagName).toBe('DIV')

    const removeButton = wrapper.find('.pending-remove')
    expect(removeButton.element.tagName).toBe('BUTTON')
    expect(removeButton.attributes('aria-label')).toBe('移除附件：scene.png')

    await removeButton.trigger('click')
    expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:attachment-preview')
  })

  it('uses pressed buttons for export format selection', async () => {
    const state = {
      pages: ref([{ page_number: 1 }]),
      showMessage: vi.fn(),
    }

    const wrapper = mount(ExportPanel, {
      props: {
        bookId: 'book-1',
        generatedCount: 1,
        state: state as any,
      },
    })

    const formatCards = wrapper.findAll('.format-card')
    expect(formatCards).toHaveLength(2)
    expect(formatCards[0].element.tagName).toBe('BUTTON')
    expect(formatCards[0].attributes('aria-pressed')).toBe('true')
    expect(formatCards[1].attributes('aria-pressed')).toBe('false')

    await formatCards[1].trigger('click')
    expect(formatCards[1].attributes('aria-pressed')).toBe('true')
  })
})
