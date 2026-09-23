import { mount, enableAutoUnmount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ReaderCanvas from '@/components/reader/ReaderCanvas.vue'
import ReaderImage from '@/components/reader/ReaderImage.vue'
import VirtualPageStream from '@/components/virtual/VirtualPageStream.vue'
import { DEFAULT_READER_SETTINGS } from '@/components/reader/readerSettings'
import type { V2PageSummary } from '@/api/v2/content'
enableAutoUnmount(afterEach)
const image = { id: 'p1', chapterId: 'a', ordinal: 1, sourceUrl: '/source', translatedUrl: '/translated', width: 800, height: 1200 } as V2PageSummary
beforeEach(() => setActivePinia(createPinia()))
afterEach(() => vi.unstubAllGlobals())
describe('reader canvas', () => {
  it('restores long-page progress on mount and when changing its fit', async () => {
    let resize = () => {}
    vi.stubGlobal('ResizeObserver', class {
      constructor(callback: () => void) { resize = callback }
      observe() {}
      disconnect() {}
    })
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { callback(0); return 1 })
    const settings = { ...DEFAULT_READER_SETTINGS, layout: 'single' as const, fits: { ...DEFAULT_READER_SETTINGS.fits, single: 'original' as const } }
    const wrapper = mount(ReaderCanvas, { props: { images: [image], viewMode: 'original', isLoading: false, settings, position: { pageId: 'p1', index: 0, fraction: 0.25 } } })
    await flushPromises()
    const el = wrapper.get('.reader-canvas__paged').element as HTMLElement
    expect(el.scrollTop).toBe(300)
    Object.defineProperties(el, { clientWidth: { value: 400 }, clientHeight: { value: 300 } })
    resize()
    await wrapper.setProps({ settings: { ...settings, fits: { ...settings.fits, single: 'width' } } })
    await flushPromises()
    expect(el.scrollTop).toBe(150)
  })
  it('reuses the continuous stream for both axes and preserves page identity when changing image URLs', async () => {
    const wrapper = mount(ReaderCanvas, { props: { images: [image], viewMode: 'translated', isLoading: false }, global: { stubs: { VirtualPageStream: true } } })
    expect(wrapper.getComponent(VirtualPageStream).props('items')[0].url).toBe('/translated')
    await wrapper.setProps({ viewMode: 'original', settings: { ...DEFAULT_READER_SETTINGS, layout: 'horizontal', direction: 'rtl' } })
    expect(wrapper.getComponent(VirtualPageStream).props()).toMatchObject({ horizontal: true, direction: 'rtl' })
    expect(wrapper.getComponent(VirtualPageStream).props('items')[0]).toMatchObject({ id: 'p1', url: '/source' })
  })
  it('shows only the current spread in the selected reading direction', async () => {
    const wrapper = mount(ReaderCanvas, { props: { images: [image, { ...image, id: 'p2' }, { ...image, id: 'p3' }], viewMode: 'original', isLoading: false, settings: { ...DEFAULT_READER_SETTINGS, layout: 'double', direction: 'rtl' }, group: [0, 1] } })
    expect(wrapper.findAll('figure').map(f => f.attributes('data-page-id'))).toEqual(['p2', 'p1'])
    await wrapper.setProps({ group: [2] })
    expect(wrapper.findAll('figure').map(f => f.attributes('data-page-id'))).toEqual(['p3'])
  })
  it('marks source fallbacks only when viewing translations', async () => {
    const wrapper = mount(ReaderCanvas, { props: { images: [{ ...image, translatedUrl: null }], viewMode: 'translated', isLoading: false, settings: { ...DEFAULT_READER_SETTINGS, layout: 'single' } } })
    expect(wrapper.getComponent(ReaderImage).props()).toMatchObject({ src: '/source', badge: '未翻译' })
    await wrapper.setProps({ viewMode: 'original' })
    expect(wrapper.getComponent(ReaderImage).props('badge')).toBeUndefined()
  })
  it('offers image retry without removing the page', async () => {
    const wrapper = mount(ReaderImage, { props: { src: '/broken', alt: '第 1 页' } })
    await wrapper.get('img').trigger('error')
    expect(wrapper.text()).toContain('加载失败')
    await wrapper.get('button').trigger('click')
    expect(wrapper.get('img').attributes('src')).toBe('/broken')
    await wrapper.get('img').trigger('error')
    await wrapper.setProps({ src: '/replacement' })
    expect(wrapper.get('img').attributes('src')).toBe('/replacement')
  })
})
