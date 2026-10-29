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
