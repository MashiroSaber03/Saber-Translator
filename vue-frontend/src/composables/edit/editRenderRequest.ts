import type { RenderInput } from '@/composables/translation/core/steps'
import { buildSavedTextStylesFromSettings } from '@/composables/translation/core/runtime'
import type { BubbleCoords, BubbleState } from '@/types/bubble'
import type { OcrResult } from '@/types/ocr'
import type { TranslationSettings } from '@/types/settings'

export interface EditRenderInputParams {
  imageIndex: number
  cleanImage: string
  bubbleStates: BubbleState[]
  settings: TranslationSettings
  renderStylePolicy?: RenderInput['renderStylePolicy']
}

function toRoundedBubbleCoords(coords: BubbleCoords): BubbleCoords {
  return coords.map((coord) => Math.round(coord)) as BubbleCoords
}

function toRenderOcrResult(state: BubbleState): OcrResult {
  return state.ocrResult || {
    text: state.originalText || '',
    confidence: null,
    confidenceSupported: false,
    engine: '',
    primaryEngine: '',
    fallbackUsed: false,
  }
}

export function buildEditRenderInput({
  imageIndex,
  cleanImage,
  bubbleStates,
  settings,
  renderStylePolicy = {
    fontSize: 'preserve',
    color: 'preserve',
  },
}: EditRenderInputParams): RenderInput {
  return {
    imageIndex,
    cleanImage,
    bubbleCoords: bubbleStates.map((state) => toRoundedBubbleCoords(state.coords)),
    bubbleAngles: bubbleStates.map((state) => state.rotationAngle || 0),
    autoDirections: bubbleStates.map((state) => state.autoTextDirection || state.textDirection || 'vertical'),
    textlinesPerBubble: bubbleStates.map((state) => state.textlines || []),
    existingBubbleStates: bubbleStates,
    originalTexts: bubbleStates.map((state) => state.originalText || ''),
    ocrResults: bubbleStates.map((state) => toRenderOcrResult(state)),
    translatedTexts: bubbleStates.map((state) => state.translatedText || ''),
    textboxTexts: bubbleStates.map((state) => state.textboxText || ''),
    colors: bubbleStates.map((state) => ({
      textColor: state.textColor || settings.textStyle.textColor,
      bgColor: state.fillColor || settings.textStyle.fillColor,
      autoFgColor: state.autoFgColor || null,
      autoBgColor: state.autoBgColor || null,
    })),
    savedTextStyles: buildSavedTextStylesFromSettings(settings),
    currentMode: 'standard',
    settingsSnapshot: settings,
    renderStylePolicy,
  }
}
