import { mount, flushPromises } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import RichTextInput from '@/components/edit/RichTextInput.vue'
import { createBubbleState } from '@/utils/bubbleFactory'

const wrappers: ReturnType<typeof mount>[] = []
afterEach(() => { wrappers.splice(0).forEach(wrapper => wrapper.unmount()) })

function editor(text = '中文😀') {
  const bubble = createBubbleState({ coords: [0, 0, 100, 200], fontFamily: '', translatedText: text })
  const wrapper = mount(RichTextInput, { props: { bubble, fonts: [] }, attachTo: document.body })
  wrappers.push(wrapper)
  const vm = wrapper.vm as unknown as { format: (value: object) => void; insertText: (text: string) => void; deleteBackward: () => void }
  return { wrapper, vm, bubble }
}

describe('ordinary rich text input', () => {
  it('formats existing text immediately after typing without reselecting its canvas frame', async () => {
    const { wrapper, vm } = editor('')
    await wrapper.get('[role=textbox]').trigger('focus')
    vm.insertText('新建😀')
    vm.format({ size_pt: 70, font_family: 'Test', horizontal_scale: 1.5 })
    const value = wrapper.emitted('update')!.at(-1)![0] as { translatedText: string; richText: { runs: { len: number; style: object }[] } }
    expect(value.translatedText).toBe('新建😀')
    expect(value.richText.runs).toEqual([{ len: 10, style: expect.objectContaining({ size_pt: 70, font_family: 'Test', horizontal_scale: 1.5 }) }])
    expect(wrapper.emitted('selection')?.at(-1)?.[0]).toMatchObject({ active: false, count: 0 })
    vm.insertText('字')
    expect((wrapper.emitted('update')!.at(-1)![0] as typeof value).richText.runs)
      .toEqual([{ len: 13, style: expect.objectContaining({ size_pt: 70, font_family: 'Test', horizontal_scale: 1.5 }) }])
  })

  it('keeps character and paragraph styles when deleting all and immediately retyping', async () => {
    const { wrapper, vm, bubble } = editor('保留字号\n第二行')
    const style = { size_pt: 60, font_family: 'Test', tracking: 85, vertical_scale: 1.2, emphasis: true }
    const paragraph = { align: 'Center', space_before_pt: 5 }
    await wrapper.setProps({ bubble: { ...bubble, richText: {
      runs: [{ len: 22, style }], paragraphs: [{ len: 13, style: paragraph }, { len: 9, style: paragraph }],
    } } })
    const input = wrapper.get('[role=textbox]')
    await input.trigger('focus')
    await input.trigger('keydown', { key: 'a', ctrlKey: true })
    vm.deleteBackward()
    const empty = wrapper.emitted('update')!.at(-1)![0] as { translatedText: string; richText: { runs: { len: number; style: object }[]; paragraphs: { len: number; style: object }[] } }
    expect(empty).toEqual({ translatedText: '', richText: {
      runs: [{ len: 0, style }], paragraphs: [{ len: 0, style: paragraph }],
    } })
    vm.insertText('重输😀')
    expect((wrapper.emitted('update')!.at(-1)![0] as typeof empty).richText.runs).toEqual([{ len: 10, style }])
    vm.format({ size_pt: 80 })
    expect((wrapper.emitted('update')!.at(-1)![0] as typeof empty).richText.runs)
      .toEqual([{ len: 10, style: { ...style, size_pt: 80 } }])
  })

  it.each(['原文字', '\n原文字', '\n\n原文字'])('keeps the selected character style when replacing all text in %j directly', async text => {
    const { wrapper, vm } = editor(text)
    vm.format({ size_pt: 60, font_family: 'Test', tracking: 125, faux_bold: true })
    const input = wrapper.get('[role=textbox]')
    await input.trigger('focus')
    await input.trigger('keydown', { key: 'a', ctrlKey: true })
    vm.insertText('直接替换')
    const value = wrapper.emitted('update')!.at(-1)![0] as { richText: { runs: { style: object }[] } }
    expect(value.richText.runs[0]!.style).toMatchObject({ size_pt: 60, font_family: 'Test', tracking: 125, faux_bold: true })
  })

  it.each(['\n文字', '文字\n', '\n\n文字'])('preserves the empty style when clearing paragraphs in %j', async text => {
    const { wrapper, vm, bubble } = editor(text)
    const style = { size_pt: 60, font_family: 'Test', tracking: 125, faux_bold: true }
    await wrapper.setProps({ bubble: { ...bubble, fontSize: 60, richText: {
      runs: [{ len: new TextEncoder().encode(text).length, style }], paragraphs: [],
    } } })
    const input = wrapper.get('[role=textbox]')
    await input.trigger('focus')
    await input.trigger('keydown', { key: 'a', ctrlKey: true })
    vm.deleteBackward()
    vm.insertText('重输')
    const value = wrapper.emitted('update')!.at(-1)![0] as { richText: { runs: { style: object }[] } }
    expect(value.richText.runs[0]!.style).toMatchObject(style)
  })

  it('uses whole-frame mixed styles for controls and keyboard shortcuts when no text is selected', async () => {
    const { wrapper, bubble } = editor('AB')
    await wrapper.setProps({ bubble: { ...bubble, richText: {
      runs: [{ len: 1, style: { size_pt: 30, faux_bold: true } }, { len: 1, style: { size_pt: 60 } }],
      paragraphs: [{ len: 2, style: {} }],
    } } })
    const input = wrapper.get('[role=textbox]')
    await input.trigger('focus')
    expect(wrapper.emitted('selection')?.at(-1)?.[0]).toMatchObject({ active: false, count: 0, styles: [{ size_pt: 30, faux_bold: true }, { size_pt: 60 }] })
    await input.trigger('keydown', { key: 'b', ctrlKey: true })
    const value = wrapper.emitted('update')!.at(-1)![0] as { richText: { runs: { style: { size_pt: number; faux_bold: boolean } }[] } }
    expect(value.richText.runs.map(run => run.style.size_pt)).toEqual([30, 60])
    expect(value.richText.runs.every(run => run.style.faux_bold)).toBe(true)
  })

  it('reselects all after the browser caret moves while the previous logical selection is still all', async () => {
    const { wrapper } = editor('中文😀')
    const input = wrapper.get('[role=textbox]')
    ;(input.element as HTMLElement).focus()
    await input.trigger('keydown', { key: 'a', ctrlKey: true })
    expect(window.getSelection()!.toString()).toBe('中文😀')
    const text = document.createTreeWalker(input.element, NodeFilter.SHOW_TEXT).nextNode()!
    // Refocusing after a number-field edit can move the native caret before
    // ProseMirror receives selectionchange, leaving its logical range unchanged.
    window.getSelection()!.collapse(text, text.textContent!.length)
    await input.trigger('keydown', { key: 'a', ctrlKey: true })
    expect(window.getSelection()!.toString()).toBe('中文😀')
  })
  it('selects characters across paragraphs and keeps their formatting range after blur', async () => {
    const { wrapper, vm } = editor('今天\n中文😀')
    const input = wrapper.get('[role=textbox]')
    await input.trigger('focus')
    await input.trigger('keydown', { key: 'a', ctrlKey: true })
    expect(wrapper.emitted('selection')?.at(-1)?.[0]).toMatchObject({ active: true, count: 6 })
    expect(wrapper.findAll('.rich-text-input__selection').map(part => part.text()).join('')).toBe('今天中文😀')
    await input.trigger('blur')
    vm.format({ emphasis: true })
    const value = wrapper.emitted('update')!.at(-1)![0] as { translatedText: string; richText: { runs: { len: number; style: { emphasis?: boolean } }[] } }
    expect(value.translatedText).toBe('今天\n中文😀')
    expect(value.richText.runs.every(run => run.style.emphasis)).toBe(true)
    expect(value.richText.runs.reduce((sum, run) => sum + run.len, 0)).toBe(17)
    expect(wrapper.emitted('selection')?.at(-1)?.[0]).toMatchObject({ active: true, count: 6 })
  })
  it('installs editing/history for unsaved boxes and preserves a caret style on insertion', async () => {
    const { wrapper, vm } = editor('')
    await wrapper.get('[role=textbox]').trigger('focus')
    vm.format({ size_pt: 36, faux_bold: true })
    vm.insertText('汉😀')
    const value = wrapper.emitted('update')!.at(-1)![0] as { translatedText: string; richText: { runs: { len: number; style: object }[] } }
    expect(value.translatedText).toBe('汉😀')
    expect(value.richText.runs).toEqual([{ len: 7, style: expect.objectContaining({ size_pt: 36, faux_bold: true }) }])
    await wrapper.get('[role=textbox]').trigger('keydown', { key: 'z', ctrlKey: true })
    expect((wrapper.emitted('update')!.at(-1)![0] as { translatedText: string }).translatedText).toBe('')
  })

  it('persists the style of an empty box and keeps fine editor properties when styling the whole box', () => {
    const { wrapper, vm } = editor('')
    vm.format({ faux_italic: true, tracking: 125 })
    const value = wrapper.emitted('update')!.at(-1)![0] as { richText: { runs: { style: object }[] } }
    expect(value.richText.runs[0]!.style).toMatchObject({ faux_italic: true, tracking: 125 })
  })

  it('retains input history when first saving assigns the same new box its backend ID', async () => {
    const { wrapper, vm, bubble } = editor('')
    const draft = { ...bubble, clientMutationId: 'draft' }
    await wrapper.setProps({ bubble: draft })
    vm.insertText('新字')
    const saved = { ...draft, backendBubbleId: 'saved', translatedText: '新字' }
    Object.assign(draft, { backendBubbleId: 'saved' })
    delete (draft as Partial<typeof draft>).clientMutationId
    delete saved.clientMutationId
    await wrapper.setProps({ bubble: saved })
    await wrapper.get('[role=textbox]').trigger('keydown', { key: 'z', ctrlKey: true })
    expect((wrapper.emitted('update')!.at(-1)![0] as { translatedText: string }).translatedText).toBe('')
  })

  it('retains paragraph spacing when a canonical document is loaded again', async () => {
    const { wrapper, vm, bubble } = editor('A\nB')
    const richText = { runs: [{ len: 3, style: { size_pt: 24, language: 'ja', tracking: 85 } }], paragraphs: [
      { len: 2, style: { space_before_pt: 10 } }, { len: 1, style: { align: 'Right' } },
    ] }
    await wrapper.setProps({ bubble: { ...bubble, richText } })
    vm.format({ underline: true })
    await flushPromises()
    const value = wrapper.emitted('update')!.at(-1)![0] as { richText: typeof richText }
    expect(value.richText.paragraphs).toEqual(richText.paragraphs)
    expect(value.richText.runs[0]!.style).toMatchObject({ language: 'ja', tracking: 85, underline: true })
    expect(value.richText.runs).toHaveLength(1)
    expect(value.richText.runs[0]!.len).toBe(3)
  })
  it('inherits fine-editor styles when typing into an empty text layer', async () => {
    const { wrapper, vm, bubble } = editor('')
    const style = { size_pt: 36, tracking: 120, language: 'ja', baseline_shift_pt: 2 }
    await wrapper.setProps({ bubble: { ...bubble, backendBubbleId: 'empty-fine-layer', richText: {
      runs: [{ len: 0, style }], paragraphs: [{ len: 0, style: {} }],
    } } })
    vm.insertText('汉字')
    expect((wrapper.emitted('update')!.at(-1)![0] as { richText: { runs: { len: number; style: object }[] } }).richText.runs)
      .toEqual([{ len: 6, style }])
  })
})
