import { shallowRef } from 'vue'
import type { BubbleState } from '@/types/bubble'
import { deepClone } from '@/utils/deepClone'
import type { RunStyle } from './richTextDocument'

// A small session clipboard; page content, geometry and repair settings are never copied.
export type TextStyleTemplate = { character: RunStyle; box: Partial<BubbleState> }
export const copiedTextStyle = shallowRef<TextStyleTemplate | null>(null)

export function textStyleTemplate(bubble: BubbleState, character: RunStyle): TextStyleTemplate {
  return deepClone({ character, box: {
    textDirection: bubble.textDirection, lineSpacing: bubble.lineSpacing,
    inlineAlign: bubble.inlineAlign, blockAlign: bubble.blockAlign,
    strokeEnabled: bubble.strokeEnabled, strokeColor: bubble.strokeColor, strokeWidth: bubble.strokeWidth,
  } })
}

export function copyTextStyle(bubble: BubbleState, character: RunStyle): void {
  copiedTextStyle.value = textStyleTemplate(bubble, character)
}

export function applyTextStyle(bubble: BubbleState, template: NonNullable<typeof copiedTextStyle.value>): Partial<BubbleState> {
  const len = new TextEncoder().encode(bubble.translatedText).length
  return { ...deepClone(template.box), richText: {
    runs: [{ len, style: deepClone(template.character) }],
    paragraphs: deepClone(bubble.richText?.paragraphs ?? [{ len, style: {} }]),
  } }
}
