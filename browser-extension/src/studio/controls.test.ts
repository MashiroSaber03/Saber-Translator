// @vitest-environment jsdom
import { createApp, nextTick, type App } from 'vue'
import { afterEach, expect, it, vi } from 'vitest'
import SelectControl from './SelectControl.vue'
import ImportDialog from './ImportDialog.vue'
let app: App
function root() { document.body.innerHTML = '<div id="app"></div>'; return document.querySelector('#app')! }
const flush = async () => { await new Promise(resolve => setTimeout(resolve, 0)); await nextTick() }
afterEach(() => { app.unmount(); vi.restoreAllMocks() })
it('opens a keyboard-operable dropdown without scrolling its containing page', async () => {
  const focus = vi.spyOn(HTMLElement.prototype, 'focus')
  const change = vi.fn()
  app = createApp(SelectControl, { modelValue: 'a', label: '选择', options: [{ value: 'a', label: '甲' }, { value: 'b', label: '乙' }], 'onUpdate:modelValue': change })
  app.mount(root())
  document.querySelector<HTMLButtonElement>('[role=combobox]')!.click()
  await nextTick()
  expect(focus).toHaveBeenCalledWith({ preventScroll: true })
  document.dispatchEvent(new Event('scroll'))
  await nextTick()
  expect(document.querySelector('[role=listbox]')).not.toBeNull()
  const list = document.querySelector<HTMLElement>('[role=listbox]')!
  Element.prototype.scrollIntoView = vi.fn()
  list.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }))
  list.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))
  await nextTick()
  expect(change).toHaveBeenCalledWith('b')
  expect(document.querySelector('[role=listbox]')).toBeNull()
})
it('allows closing an import dialog while the library request is pending', async () => {
  let finish!: (value: []) => void
  const close = vi.fn()
  app = createApp(ImportDialog, { title: '章节', loadBooks: () => new Promise<[]>(resolve => { finish = resolve }), submit: vi.fn(), onClose: close })
  app.mount(root())
  const button = document.querySelector<HTMLButtonElement>('[aria-label="关闭导入"]')!
  expect(button.disabled).toBe(false)
  button.click()
  expect(close).toHaveBeenCalledOnce()
  finish([])
  await flush()
})
it('submits an existing book destination and blocks duplicate imports', async () => {
  let finish!: (value: any) => void
  const submit = vi.fn(() => new Promise<any>(resolve => { finish = resolve }))
  app = createApp(ImportDialog, { title: '章节', loadBooks: async () => [{ id: 'book-a', title: '书籍', chapterCount: 2 }], submit })
  app.mount(root())
  await flush()
  ;[...document.querySelectorAll('button')].find(b => b.textContent?.trim() === '已有书籍')!.click()
  await nextTick()
  document.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
  document.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
  await nextTick()
  expect(submit).toHaveBeenCalledExactlyOnceWith({ destination: 'existing', targetBookId: 'book-a', chapterTitle: '章节' })
  expect(document.querySelector<HTMLButtonElement>('[aria-label="关闭导入"]')!.disabled).toBe(true)
  finish({})
  await flush()
})
