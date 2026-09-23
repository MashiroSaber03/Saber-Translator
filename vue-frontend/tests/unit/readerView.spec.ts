import { mount, flushPromises, enableAutoUnmount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import ReaderView from '@/views/ReaderView.vue'
import ReaderCanvas from '@/components/reader/ReaderCanvas.vue'
import ReaderControls from '@/components/reader/ReaderControls.vue'
import { DEFAULT_READER_SETTINGS } from '@/components/reader/readerSettings'
import { readerHistoryKey, saveReaderRecord, loadReaderRecord } from '@/components/reader/readerHistory'
const { getBook, listPages, push, toast } = vi.hoisted(() => ({ getBook: vi.fn(), listPages: vi.fn(), push: vi.fn(), toast: vi.fn() }))
vi.mock('@/api/v2/content', () => ({ getBook, listChapterPages: listPages }))
vi.mock('vue-router', () => ({ useRouter: () => ({ push }) }))
vi.mock('@/utils/toast', () => ({ useToast: () => ({ error: toast }) }))
enableAutoUnmount(afterEach)
const page = (id: string, chapterId = 'c1') => ({ id, chapterId, ordinal: 1, width: 800, height: 1200, sourceUrl: `/${id}`, translatedUrl: null })
const book = { id: 'b', title: 'Book', chapters: [{ id: 'c1', title: 'One' }, { id: 'c2', title: 'Two' }] }
function create() {
  return mount(ReaderView, { props: { bookId: 'b', chapterId: 'c1' }, global: { stubs: { ReaderCanvas: true, ReaderControls: true, ReaderProgress: true } } })
}
beforeEach(() => {
  setActivePinia(createPinia()); localStorage.clear(); vi.clearAllMocks()
  vi.stubGlobal('matchMedia', vi.fn(() => ({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() })))
  getBook.mockResolvedValue(book)
  listPages.mockResolvedValue({ items: [page('p1'), page('p2'), page('p3')], nextCursor: null })
})
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers() })
describe('reader state ownership', () => {
  it('shows zero for empty chapters', async () => {
    listPages.mockResolvedValue({ items: [], nextCursor: null })
    const wrapper = create(); await flushPromises()
    expect(wrapper.get('.reader-header__page-info').text()).toBe('0 / 0')
  })
  it('restores and saves chapter-local progress and offset', async () => {
    const key = readerHistoryKey('local', 'b', 'c1')
    saveReaderRecord({ key, pageId: 'p2', index: 1, fraction: .3, offset: true })
    const wrapper = create(); await flushPromises()
    expect(wrapper.getComponent(ReaderCanvas).props('position')).toMatchObject({ pageId: 'p2', fraction: .3 })
    expect(wrapper.getComponent(ReaderControls).props('offset')).toBe(true)
    wrapper.getComponent(ReaderControls).vm.$emit('jump', 3)
    await flushPromises(); wrapper.unmount()
    expect(loadReaderRecord(key)).toMatchObject({ pageId: 'p3', index: 2, fraction: 0 })
  })
  it('preserves the page when switching layout and original/translated views', async () => {
    const wrapper = create(); await flushPromises()
    wrapper.getComponent(ReaderControls).vm.$emit('jump', 2)
    await flushPromises()
    wrapper.getComponent(ReaderControls).vm.$emit('settingsChange', { ...DEFAULT_READER_SETTINGS, layout: 'double' })
    await wrapper.get('[data-mode="original"]').trigger('click')
    expect(wrapper.getComponent(ReaderCanvas).props('position').pageId).toBe('p2')
    expect(wrapper.get('.reader-header__page-info').text()).toBe('1–2 / 3')
  })
  it('page arrows never navigate chapters and chapter controls use the loaded chapter list', async () => {
    const wrapper = create(); await flushPromises()
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'End' }))
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }))
    await flushPromises(); expect(push).not.toHaveBeenCalled()
    expect(wrapper.getComponent(ReaderCanvas).props('position').pageId).toBe('p3')
    wrapper.getComponent(ReaderControls).vm.$emit('chapter', 'c2')
    expect(push).toHaveBeenCalledWith({ path: '/reader', query: { book: 'b', chapter: 'c2' } })
  })
  it.each([
    { items: [page('p1')], nextCursor: 1 },
    { items: [page('wrong', 'other')], nextCursor: null },
  ])('rejects incomplete or misattributed page lists', async result => {
    listPages.mockResolvedValue(result)
    const wrapper = create(); await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('章节页面列表')
    expect(wrapper.findComponent(ReaderCanvas).exists()).toBe(false)
  })
  it('rejects a chapter outside the current book', async () => {
    getBook.mockResolvedValue({ ...book, chapters: [] })
    const wrapper = create(); await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('章节不属于当前书籍')
  })
  it('keeps errors reviewable instead of redirecting away', async () => {
    getBook.mockRejectedValue(Error('network down'))
    const wrapper = create(); await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('network down')
    expect(push).not.toHaveBeenCalled()
  })
  it('ignores late responses for a previously selected chapter', async () => {
    let finish!: (value: unknown) => void
    listPages.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const wrapper = create()
    listPages.mockResolvedValueOnce({ items: [page('new', 'c2')], nextCursor: null })
    await wrapper.setProps({ chapterId: 'c2' }); await flushPromises()
    finish({ items: [page('old')], nextCursor: null }); await flushPromises()
    expect(wrapper.getComponent(ReaderCanvas).props('images')[0].id).toBe('new')
  })
  it('does not persist an unfinished request after leaving the reader', async () => {
    let finish!: (value: unknown) => void
    listPages.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const wrapper = create(); wrapper.unmount()
    finish({ items: [page('late')], nextCursor: null }); await flushPromises()
    expect(loadReaderRecord(readerHistoryKey('local', 'b', 'c1'))).toBeUndefined()
  })
})
