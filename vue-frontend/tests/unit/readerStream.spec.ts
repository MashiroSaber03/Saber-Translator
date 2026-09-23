import { mount, enableAutoUnmount, flushPromises } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import VirtualPageStream from '@/components/virtual/VirtualPageStream.vue'

enableAutoUnmount(afterEach)
let resize: () => void
beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class {
    constructor(callback: () => void) { resize = callback }
    observe() {}
    disconnect() {}
  })
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { callback(0); return 1 })
})
afterEach(() => vi.unstubAllGlobals())
function items(count: number) {
  return Array.from({ length: count }, (_, i) => ({ id: `p${i}`, alt: `${i}`, url: `/page-${i}`, width: 800, height: 1200 }))
}
async function create(count = 1000) {
  const wrapper = mount(VirtualPageStream, { props: { items: items(count), position: { pageId: 'p0', index: 0, fraction: 0 }, navigationId: 0 }, global: { stubs: { ReaderImage: true } } })
  const el = wrapper.element as HTMLElement
  Object.defineProperties(el, { clientWidth: { value: 800 }, clientHeight: { value: 600 }, scrollWidth: { value: count * 808 }, scrollHeight: { value: count * 1208 } })
  resize(); await flushPromises()
  return { wrapper, el }
}
describe('virtual reader positioning', () => {
  it('reserves image geometry before the image loads', async () => {
    const { wrapper } = await create(2)
    const figure = wrapper.get('figure').element as HTMLElement
    expect(Number.parseFloat(figure.style.width) / Number.parseFloat(figure.style.height)).toBeCloseTo(800 / 1200)
  })
  it('keeps rendered images bounded in a thousand-page chapter', async () => {
    const { wrapper, el } = await create()
    expect(wrapper.findAll('figure').length).toBeLessThanOrEqual(6)
    el.scrollTop = 500 * 1208
    await wrapper.trigger('scroll')
    expect(wrapper.findAll('figure').length).toBeLessThanOrEqual(6)
    expect(wrapper.emitted('positionChange')!.at(-1)![0]).toMatchObject({ pageId: 'p500', index: 500, fraction: 0 })
  })
  it('does not reset the scroll position when switching source and translated URLs', async () => {
    const { wrapper, el } = await create()
    el.scrollTop = 8 * 1208 + 100
    await wrapper.trigger('scroll')
    await wrapper.setProps({ items: items(1000).map(item => ({ ...item, url: `${item.url}-translated` })) })
    await flushPromises()
    expect(el.scrollTop).toBe(8 * 1208 + 100)
  })
  it('restores the requested page on a different axis, including right-to-left layout', async () => {
    const { wrapper, el } = await create(20)
    await wrapper.setProps({ horizontal: true, direction: 'rtl', fit: 'original', position: { pageId: 'p12', index: 12, fraction: 0 }, navigationId: 1 })
    await flushPromises()
    expect(el.scrollLeft).toBe(20 * 808 - 800 - 12 * 808)
    expect(wrapper.emitted('positionChange')!.at(-1)![0]).toMatchObject({ pageId: 'p12', index: 12 })
  })
  it('also keeps horizontal rendering bounded when jumping deep into a thousand-page chapter', async () => {
    const { wrapper } = await create()
    await wrapper.setProps({ horizontal: true, fit: 'height', position: { pageId: 'p500', index: 500, fraction: 0 }, navigationId: 1 })
    await flushPromises()
    expect(wrapper.findAll('figure').length).toBeLessThanOrEqual(12)
    expect(wrapper.emitted('positionChange')!.at(-1)![0]).toMatchObject({ pageId: 'p500', index: 500 })
  })
  it('keeps an explicit target even when a short chapter fits within the viewport', async () => {
    const { wrapper } = await create(2)
    await wrapper.setProps({ horizontal: true, fit: 'screen', position: { pageId: 'p1', index: 1, fraction: 0 }, navigationId: 1 })
    await flushPromises()
    expect(wrapper.emitted('positionChange')!.at(-1)![0]).toMatchObject({ pageId: 'p1', index: 1 })
  })
})
