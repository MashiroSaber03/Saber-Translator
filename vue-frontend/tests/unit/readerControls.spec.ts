import { mount, enableAutoUnmount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import ReaderControls from '@/components/reader/ReaderControls.vue'
import { DEFAULT_READER_SETTINGS } from '@/components/reader/readerSettings'
enableAutoUnmount(afterEach)
function create() {
  return mount(ReaderControls, {
    props: { settings: { ...DEFAULT_READER_SETTINGS }, page: 1, total: 5, range: '1', offset: false,
      canPrev: false, canNext: true, hasPrevChapter: false, hasNextChapter: true,
      chapterId: 'a', chapters: [{ id: 'a', title: '第一章' }, { id: 'b', title: '第二章' }] },
    global: { stubs: { UiSelect: true } },
  })
}
describe('reader controls', () => {
  it('publishes mode changes instead of maintaining a second copy of settings', async () => {
    const wrapper = create()
    await wrapper.get('[aria-label="循环切换阅读模式"]').trigger('click')
    expect(wrapper.emitted('settingsChange')?.[0]?.[0]).toMatchObject({ layout: 'horizontal' })
    expect(wrapper.props('settings').layout).toBe('vertical')
  })
  it('only shows the offset control in double-page mode', async () => {
    const wrapper = create()
    expect(wrapper.text()).not.toContain('双页错开一页')
    await wrapper.setProps({ settings: { ...DEFAULT_READER_SETTINGS, layout: 'double' } })
    await wrapper.findAll('button').find(b => b.text() === '双页错开一页')!.trigger('click')
    expect(wrapper.emitted('offset')).toHaveLength(1)
  })
  it('separates page navigation, chapter navigation and explicit page jumps', async () => {
    const wrapper = create()
    expect(wrapper.get('[aria-label="上一页"]').attributes('disabled')).toBeDefined()
    await wrapper.get('[aria-label="下一页"]').trigger('click')
    expect(wrapper.emitted('navigate')).toEqual([[1]])
    await wrapper.findAll('button').find(b => b.text() === '下一章')!.trigger('click')
    expect(wrapper.emitted('chapter')).toEqual([['b']])
    await wrapper.get('input[aria-label="跳转页码"]').setValue('4')
    await wrapper.get('form').trigger('submit')
    expect(wrapper.emitted('jump')).toEqual([[4]])
  })
})
