import { Schema, type Node as TextNode } from 'prosemirror-model'
import type { BubbleState } from '@/types/bubble'
import type { operations } from '@/api/generated/v2'

export type RunStyle = Record<string, unknown>
export type NativeRichText = NonNullable<BubbleState['richText']>
export type EditorFont = operations['listEditorFonts']['responses'][200]['content']['application/json']['items'][number]
const encoder = new TextEncoder()
const bytes = (text: string) => encoder.encode(text).length

export function textColor(style: RunStyle): string | undefined {
  const color = style.color as { mode?: string; c?: number[]; alpha?: number } | undefined
  if (color?.mode !== 'Rgb' || !color.c) return undefined
  return `#${color.c.slice(0, 3).map(c => Math.round(Math.max(0, Math.min(1, c)) * 255).toString(16).padStart(2, '0')).join('')}`
}

export function nativeColor(hex: string): RunStyle {
  return { mode: 'Rgb', c: [1, 3, 5].map(at => Number.parseInt(hex.slice(at, at + 2), 16) / 255).concat(0), alpha: 1 }
}

function runCss(style: RunStyle): string {
  // Input uses a readable base size; relative sizes and all native attributes survive saving.
  const size = Math.max(1, Number(style.size_pt) || 16)
  const family = String(style.font_family ?? '').replace(/["\\;{}]/g, '')
  const color = textColor(style)
  const rgb = (style.color as { c?: number[] } | undefined)?.c
  const lightInk = color && rgb && 0.299 * rgb[0]! + 0.587 * rgb[1]! + 0.114 * rgb[2]! > 0.7
  return `font-family:"${family}",sans-serif;font-size:max(12px, calc(16px * ${size} / var(--rich-text-document-size)));`
    + `font-weight:${style.faux_bold ? 900 : Number(style.weight) || 400};font-style:${style.faux_italic || style.italic ? 'italic' : 'normal'};`
    + `text-decoration:${[style.underline && 'underline', style.strikethrough && 'line-through'].filter(Boolean).join(' ') || 'none'};`
    + `letter-spacing:${(Number(style.tracking) || 0) / 1000}em;`
    + (color ? `color:${color};` : '')
    // Display-only contrast for white/light lettering; native styles remain unchanged.
    + (lightInk ? '-webkit-text-stroke:1.4px var(--color-text-default);paint-order:stroke fill;' : '')
}

export const textSchema = new Schema({
  nodes: {
    doc: { content: 'paragraph+' },
    paragraph: {
      content: 'text*', group: 'block', attrs: { style: { default: {} }, breakStyle: { default: {} } },
      toDOM: node => ['p', { style: 'margin:0;min-height:1.6em', 'data-saber-paragraph': JSON.stringify(node.attrs) }, 0],
      parseDOM: [{ tag: 'p[data-saber-paragraph]', getAttrs: element => {
        try {
          const attrs = JSON.parse(element.getAttribute('data-saber-paragraph') ?? '{}')
          return attrs && typeof attrs.style === 'object' && typeof attrs.breakStyle === 'object' ? attrs : false
        } catch { return false }
      } }, { tag: 'p' }, { tag: 'div' }],
    },
    text: { group: 'inline' },
  },
  marks: {
    run: {
      attrs: { style: { default: {} } },
      toDOM: mark => ['span', { style: runCss(mark.attrs.style), 'data-saber-run': JSON.stringify(mark.attrs.style) }, 0],
      parseDOM: [{ tag: 'span[data-saber-run]', getAttrs: element => {
        try {
          const style: unknown = JSON.parse(element.getAttribute('data-saber-run') ?? '{}')
          return style && typeof style === 'object' && !Array.isArray(style) ? { style } : false
        } catch { return false }
      } }],
    },
  },
})

export function defaultRun(bubble: BubbleState, font?: RunStyle): RunStyle {
  return { ...font, size_pt: bubble.fontSize, color: nativeColor(bubble.textColor), leading_pt: bubble.fontSize * bubble.lineSpacing }
}

export function plainRun(bubble: BubbleState, font?: RunStyle): RunStyle {
  const first = bubble.richText?.runs[0]?.style ?? font ?? {}
  const face = Object.fromEntries(['font_family', 'font_style', 'postscript_name', 'weight', 'italic']
    .filter(key => first[key] !== undefined).map(key => [key, first[key]]))
  return defaultRun(bubble, face)
}

export function patchRunStyle(style: RunStyle, patch: RunStyle): RunStyle {
  const result = { ...style, ...patch }
  if (patch.size_pt !== undefined && patch.leading_pt === undefined && style.leading_pt != null)
    result.leading_pt = Number(style.leading_pt) * Number(patch.size_pt) / (Number(style.size_pt) || 1)
  return result
}

/** Whole-frame formatting uses the same character data as the rich-text input. */
export function formatBubbleText(bubble: BubbleState, patch: RunStyle, replace = false): Partial<BubbleState> {
  const fallback = defaultRun(bubble)
  const { richText } = fromTextDocument(toTextDocument(bubble.translatedText, bubble.richText, fallback), fallback)
  return { richText: { ...richText, runs: richText.runs.map(run => ({ ...run, style: replace ? patch : patchRunStyle(run.style, patch) })) } }
}

function styleReader(runs: NativeRichText['runs'], fallback: RunStyle): (at: number) => RunStyle {
  let index = 0
  let end = runs[0]?.len ?? 0
  return at => {
    while (index < runs.length - 1 && at >= end) end += runs[++index]!.len
    return runs[index]?.style ?? fallback
  }
}

export function toTextDocument(text: string, rich: NativeRichText | null | undefined, fallback: RunStyle): TextNode {
  let offset = 0
  const readRun = styleReader(rich?.runs ?? [], fallback)
  const readParagraph = styleReader(rich?.paragraphs ?? [], {})
  const paragraphs = text.split('\n').map(line => {
    const start = offset
    const nodes: TextNode[] = []
    let buffer = ''
    let previous = readRun(offset)
    for (const character of line) {
      const style = readRun(offset)
      if (style !== previous && buffer) { nodes.push(textSchema.text(buffer, [textSchema.marks.run!.create({ style: previous })])); buffer = '' }
      previous = style
      buffer += character
      const point = character.codePointAt(0)!
      offset += point <= 0x7f ? 1 : point <= 0x7ff ? 2 : point <= 0xffff ? 3 : 4
    }
    if (buffer) nodes.push(textSchema.text(buffer, [textSchema.marks.run!.create({ style: previous })]))
    const paragraph = textSchema.nodes.paragraph!.create({
      style: readParagraph(start),
      breakStyle: readRun(offset),
    }, nodes)
    offset += 1
    return paragraph
  })
  return textSchema.nodes.doc!.create(null, paragraphs)
}

export function fromTextDocument(doc: TextNode, fallback: RunStyle): { translatedText: string; richText: NativeRichText } {
  const runs: NativeRichText['runs'] = []
  const paragraphs: NativeRichText['paragraphs'] = []
  const lines: string[] = []
  function append(text: string, style: RunStyle): void {
    const len = bytes(text)
    if (!len) return
    const last = runs.at(-1)
    if (last && JSON.stringify(last.style) === JSON.stringify(style)) last.len += len
    else runs.push({ len, style })
  }
  doc.forEach((paragraph, _, index) => {
    const text = paragraph.textContent
    lines.push(text)
    paragraph.forEach(node => append(node.text ?? '', node.marks.find(mark => mark.type.name === 'run')?.attrs.style ?? fallback))
    if (index < doc.childCount - 1) append('\n', paragraph.attrs.breakStyle ?? fallback)
    paragraphs.push({ len: bytes(text) + (index < doc.childCount - 1 ? 1 : 0), style: paragraph.attrs.style })
  })
  if (!runs.length) runs.push({ len: 0, style: doc.firstChild?.attrs.breakStyle ?? fallback })
  return { translatedText: lines.join('\n'), richText: { runs, paragraphs } }
}

/** Literal proof-reading replacement. New characters inherit the first matched character;
 * characters outside the match keep their styles, including UTF-8 native run lengths. */
export function replaceBubbleText(bubble: BubbleState, find: string, replacement: string): Partial<BubbleState> {
  if (!find || !bubble.translatedText.includes(find)) return {}
  const rich = fromTextDocument(toTextDocument(bubble.translatedText, bubble.richText, defaultRun(bubble)), defaultRun(bubble)).richText
  const readRun = styleReader(rich.runs, defaultRun(bubble))
  const readParagraph = styleReader(rich.paragraphs, {})
  const runs: NativeRichText['runs'] = []
  const paragraphs: NativeRichText['paragraphs'] = []
  let text = '', byteOffset = 0, offset = 0, paragraphLength = 0, paragraphStyle: RunStyle = readParagraph(0)
  function append(char: string, style: RunStyle, pstyle: RunStyle): void {
    const len = bytes(char)
    if (!paragraphLength) paragraphStyle = pstyle
    text += char
    const last = runs.at(-1)
    if (last && JSON.stringify(last.style) === JSON.stringify(style)) last.len += len
    else runs.push({ len, style })
    paragraphLength += len
    if (char === '\n') { paragraphs.push({ len: paragraphLength, style: paragraphStyle }); paragraphLength = 0 }
  }
  function original(chunk: string): void {
    for (const char of chunk) { append(char, readRun(byteOffset), readParagraph(byteOffset)); byteOffset += bytes(char) }
  }
  let at: number
  while ((at = bubble.translatedText.indexOf(find, offset)) !== -1) {
    original(bubble.translatedText.slice(offset, at))
    const style = readRun(byteOffset), pstyle = readParagraph(byteOffset)
    for (const char of replacement) append(char, style, pstyle)
    byteOffset += bytes(find)
    offset = at + find.length
  }
  original(bubble.translatedText.slice(offset))
  paragraphs.push({ len: paragraphLength, style: paragraphStyle })
  if (!runs.length) runs.push({ len: 0, style: rich.runs[0]!.style })
  return { translatedText: text, richText: { runs, paragraphs } }
}
