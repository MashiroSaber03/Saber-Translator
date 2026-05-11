import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import ScriptGenerationPanel from './ScriptGenerationPanel.vue'

vi.mock('@/api/continuation', () => ({
  getAvailableImages: vi.fn().mockResolvedValue({
    success: true,
    original_images: [
      { page_number: 1, path: '/tmp/page-1.png', has_image: true, token: 'original:1' },
    ],
  }),
}))

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

    const generateButton = wrapper.find('button.btn.primary')
    await generateButton.trigger('click')
    await nextTick()

    const generateEvents = wrapper.emitted('generate') || []
    expect(generateEvents[generateEvents.length - 1]).toEqual([
      {
        referenceTokens: null,
        referenceImageCount: 5,
      },
    ])
  })
})
