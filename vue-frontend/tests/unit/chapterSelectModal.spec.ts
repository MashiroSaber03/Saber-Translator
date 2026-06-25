import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import ChapterSelectModal from '@/components/insight/ChapterSelectModal.vue'

function mountModal() {
  return mount(ChapterSelectModal, {
    props: {
      chapters: [
        { id: 'chapter-1', title: '第一章', startPage: 1, endPage: 12 },
        { id: 'chapter-2', title: '第二章', startPage: 13, endPage: 24 },
      ],
    },
    global: {
      stubs: {
        BaseModal: {
          template: '<section><slot /><footer><slot name="footer" /></footer></section>',
        },
      },
    },
  })
}

describe('ChapterSelectModal', () => {
  it('renders chapter choices as buttons and emits the confirmed chapter', async () => {
    const wrapper = mountModal()

    const chapterItems = wrapper.findAll('.chapter-item')
    expect(chapterItems).toHaveLength(2)
    expect(chapterItems.map(item => item.element.tagName)).toEqual(['BUTTON', 'BUTTON'])

    await chapterItems[1]!.trigger('click')
    expect(wrapper.get('.chapter-item.selected').text()).toContain('第二章')

    const confirmButton = wrapper.findAll('button').find(button => button.text() === '确定')
    expect(confirmButton).toBeTruthy()
    await confirmButton!.trigger('click')

    expect(wrapper.emitted('select')).toEqual([[ 'chapter-2' ]])
  })
})
