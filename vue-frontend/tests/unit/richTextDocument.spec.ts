import { describe, expect, it } from 'vitest'
import { EditorState, TextSelection } from 'prosemirror-state'
import { DOMParser, DOMSerializer } from 'prosemirror-model'
import { copyTextStyle, applyTextStyle, copiedTextStyle } from '@/components/edit/textStyleClipboard'
import { createBubbleState } from '@/utils/bubbleFactory'
import { formatBubbleText, fromTextDocument, patchRunStyle, replaceBubbleText, textSchema, toTextDocument } from '@/components/edit/richTextDocument'

describe('ordinary rich text uses native character and paragraph runs', () => {
  it('changes only the requested character attribute across frames, preserving mixed styles and masks', () => {
    const bubble = createBubbleState({ coords: [10, 20, 120, 220], fontFamily: '', translatedText: '汉😀', richText: {
      runs: [{ len: 3, style: { size_pt: 24, leading_pt: 30, underline: true, language: 'ja' } },
        { len: 4, style: { size_pt: 40, leading_pt: 60, baseline_shift_pt: 3, color: { c: [1, 0, 0] } } }],
      paragraphs: [{ len: 7, style: { align: 'Center' } }],
    } })
    const result = formatBubbleText(bubble, { size_pt: 32 })
    expect(result.richText!.runs.map(run => run.style)).toEqual([
      { size_pt: 32, leading_pt: 40, underline: true, language: 'ja' },
      { size_pt: 32, leading_pt: 48, baseline_shift_pt: 3, color: { c: [1, 0, 0] } },
    ])
    expect(result.richText!.paragraphs).toEqual(bubble.richText!.paragraphs)
    expect(result).not.toHaveProperty('coords')
    expect(result).not.toHaveProperty('translatedText')
    expect(bubble.richText!.runs[0]!.style.size_pt).toBe(24)
    expect(patchRunStyle({ size_pt: 24, leading_pt: 30 }, { size_pt: 32, leading_pt: 35 })).toEqual({ size_pt: 32, leading_pt: 35 })
  })
  const normal = { font_family: 'Test', size_pt: 24, tracking: -25, variations: [{ tag: 'wght', value: 523 }], language: 'ja' }
  const red = { ...normal, color: { mode: 'Rgb', c: [1, 0, 0, 0], alpha: 0.7 }, faux_bold: true }
  const large = { ...normal, size_pt: 40, baseline_shift_pt: 3 }
  it('proofreads variable-length matches across paragraphs without resetting surrounding character styles', () => {
    const bubble = createBubbleState({ coords: [10, 20, 120, 220], fontFamily: '', translatedText: '甲😀\n乙😀丙', richText: {
      runs: [{ len: 3, style: normal }, { len: 4, style: red }, { len: 4, style: large }, { len: 4, style: red }, { len: 3, style: normal }],
      paragraphs: [{ len: 8, style: { align: 'Center' } }, { len: 10, style: { align: 'Right' } }],
    } })
    const result = replaceBubbleText(bubble, '😀', '强调！')
    expect(result.translatedText).toBe('甲强调！\n乙强调！丙')
    expect(result.richText!.runs.map(run => [run.len, run.style])).toEqual([[3, normal], [9, red], [4, large], [9, red], [3, normal]])
    expect(result.richText!.paragraphs).toEqual([{ len: 13, style: { align: 'Center' } }, { len: 15, style: { align: 'Right' } }])
    expect(result).not.toHaveProperty('coords')
    expect(replaceBubbleText(bubble, '甲😀\n乙😀丙', '').richText!.runs[0]!.len).toBe(0)
    const multiline = replaceBubbleText(bubble, '😀', '\n😀\n')
    expect(fromTextDocument(toTextDocument(multiline.translatedText!, multiline.richText, normal), normal)).toEqual(multiline)
  })
  it('round-trips Chinese, emoji, newlines and unexposed native properties without loss', () => {
    const rich = {
      runs: [{ len: 1, style: normal }, { len: 3, style: red }, { len: 4, style: large }, { len: 3, style: normal }],
      paragraphs: [{ len: 10, style: { align: 'Center', space_before_pt: 2 } }, { len: 1, style: { align: 'Right', hyphenate: true } }],
    }
    expect(fromTextDocument(toTextDocument('A中😀B\nC', rich, normal), normal)).toEqual({ translatedText: 'A中😀B\nC', richText: rich })
  })
  it('formats only the selected characters using the existing native style object', () => {
    const doc = toTextDocument('A中😀B', null, normal)
    let state = EditorState.create({ doc, selection: TextSelection.create(doc, 2, 5) })
    state = state.apply(state.tr.addMark(2, 5, textSchema.marks.run!.create({ style: red })))
    expect(fromTextDocument(state.doc, normal).richText.runs).toEqual([
      { len: 1, style: normal }, { len: 7, style: red }, { len: 1, style: normal },
    ])
  })
  it('inherits character styles on insertion and keeps UTF-8 lengths valid on deletion', () => {
    const doc = toTextDocument('中文', { runs: [{ len: 6, style: red }], paragraphs: [{ len: 6, style: {} }] }, normal)
    let state = EditorState.create({ doc, selection: TextSelection.create(doc, 2) })
    state = state.apply(state.tr.insertText('😀'))
    expect(fromTextDocument(state.doc, normal).richText.runs).toEqual([{ len: 10, style: red }])
    state = state.apply(state.tr.delete(2, 4))
    expect(fromTextDocument(state.doc, normal).translatedText).toBe('中文')
  })
  it('keeps empty paragraphs and an empty document style', () => {
    const rich = { runs: [{ len: 0, style: large }], paragraphs: [{ len: 0, style: { align: 'Center' } }] }
    expect(fromTextDocument(toTextDocument('', rich, normal), normal).richText).toEqual(rich)
    const value = fromTextDocument(toTextDocument('\n\n', null, normal), normal)
    expect(value.translatedText).toBe('\n\n')
    expect(value.richText.paragraphs.map(p => p.len)).toEqual([1, 1, 0])
  })
  it('retains native paragraph and character properties on rich clipboard roundtrips', () => {
    const original = toTextDocument('A\nB', { runs: [{ len: 3, style: red }], paragraphs: [
      { len: 2, style: { align: 'Center', space_before_pt: 3 } }, { len: 1, style: { align: 'Right' } },
    ] }, normal)
    const container = document.createElement('div')
    container.append(DOMSerializer.fromSchema(textSchema).serializeFragment(original.content))
    expect(DOMParser.fromSchema(textSchema).parse(container).eq(original)).toBe(true)
  })
  it('copies only typography and adapts byte lengths to each target without changing its content or repair settings', () => {
    const source = createBubbleState({ coords: [0, 0, 100, 200], fontFamily: '', translatedText: 'A', fillColor: '#ff0000', inpaintMethod: 'litelama' })
    const target = createBubbleState({ coords: [100, 100, 200, 300], fontFamily: '', translatedText: '汉😀' })
    copyTextStyle(source, red)
    const result = applyTextStyle(target, copiedTextStyle.value!)
    expect(result.richText!.runs).toEqual([{ len: 7, style: red }])
    expect(result).not.toHaveProperty('fillColor')
    expect(result).not.toHaveProperty('inpaintMethod')
    expect(result).not.toHaveProperty('coords')
    expect(result).not.toHaveProperty('translatedText')
  })
})
