import textStyleDefaultsJson from '../../../src/shared/text_style_defaults_factory.json'
import type { BubbleGlobalDefaults, TextDirection } from '@/types/bubble'
import type { ImageData } from '@/types/image'
import type { TextStyleSettings } from '@/types/settings'
import { getTextStyleDefaults as fetchTextStyleDefaults } from '@/api/config'

export type TextStyleDefaults = TextStyleSettings

const rawDefaults = textStyleDefaultsJson as Record<string, unknown>

function failInvalidConfig(message: string): never {
  throw new Error(`[textStyleDefaults] ${message}`)
}

function expectTextDirection(value: unknown, fieldName: string): TextStyleSettings['layoutDirection'] {
  if (value === 'vertical' || value === 'horizontal' || value === 'auto') {
    return value
  }
  return failInvalidConfig(`${fieldName} must be one of auto/vertical/horizontal`)
}

function expectTextAlign(value: unknown, fieldName: string): TextStyleSettings['textAlign'] {
  if (value === 'start' || value === 'center' || value === 'end') {
    return value
  }
  return failInvalidConfig(`${fieldName} must be one of start/center/end`)
}

function expectInpaintMethod(value: unknown, fieldName: string): TextStyleSettings['inpaintMethod'] {
  if (value === 'solid' || value === 'lama_mpe' || value === 'litelama') {
    return value
  }
  return failInvalidConfig(`${fieldName} must be one of solid/lama_mpe/litelama`)
}

function expectPositiveInt(value: unknown, fieldName: string): number {
  const numberValue = Number(value)
  if (Number.isInteger(numberValue) && numberValue > 0) {
    return numberValue
  }
  return failInvalidConfig(`${fieldName} must be a positive integer`)
}

function expectNonNegativeInt(value: unknown, fieldName: string): number {
  const numberValue = Number(value)
  if (Number.isInteger(numberValue) && numberValue >= 0) {
    return numberValue
  }
  return failInvalidConfig(`${fieldName} must be a non-negative integer`)
}

function expectPositiveFloat(value: unknown, fieldName: string): number {
  const numberValue = Number(value)
  if (Number.isFinite(numberValue) && numberValue > 0) {
    return numberValue
  }
  return failInvalidConfig(`${fieldName} must be a positive number`)
}

function expectBoolean(value: unknown, fieldName: string): boolean {
  if (typeof value === 'boolean') {
    return value
  }
  return failInvalidConfig(`${fieldName} must be boolean`)
}

function expectNonEmptyString(value: unknown, fieldName: string): string {
  if (typeof value === 'string' && value.length > 0) {
    return value
  }
  return failInvalidConfig(`${fieldName} must be a non-empty string`)
}

function parseTextStyleDefaults(source: Record<string, unknown> | TextStyleSettings): TextStyleDefaults {
  return {
    fontSize: expectPositiveInt(source.fontSize, 'fontSize'),
    autoFontSize: expectBoolean(source.autoFontSize, 'autoFontSize'),
    fontFamily: expectNonEmptyString(source.fontFamily, 'fontFamily'),
    layoutDirection: expectTextDirection(source.layoutDirection, 'layoutDirection'),
    textColor: expectNonEmptyString(source.textColor, 'textColor'),
    fillColor: expectNonEmptyString(source.fillColor, 'fillColor'),
    strokeEnabled: expectBoolean(source.strokeEnabled, 'strokeEnabled'),
    strokeColor: expectNonEmptyString(source.strokeColor, 'strokeColor'),
    strokeWidth: expectNonNegativeInt(source.strokeWidth, 'strokeWidth'),
    inpaintMethod: expectInpaintMethod(source.inpaintMethod, 'inpaintMethod'),
    useAutoTextColor: expectBoolean(source.useAutoTextColor, 'useAutoTextColor'),
    lineSpacing: expectPositiveFloat(source.lineSpacing, 'lineSpacing'),
    textAlign: expectTextAlign(source.textAlign, 'textAlign')
  }
}

function applyTextStyleDefaults(nextDefaults: TextStyleDefaults): void {
  Object.assign(TEXT_STYLE_DEFAULTS, nextDefaults)
}

const BUNDLED_TEXT_STYLE_DEFAULTS = Object.freeze(parseTextStyleDefaults(rawDefaults))

export const TEXT_STYLE_DEFAULTS: TextStyleDefaults = {
  ...BUNDLED_TEXT_STYLE_DEFAULTS
}

export function resetTextStyleDefaultsToBundled(): void {
  applyTextStyleDefaults({ ...BUNDLED_TEXT_STYLE_DEFAULTS })
}

export async function reloadTextStyleDefaultsFromBackend(): Promise<boolean> {
  try {
    const response = await fetchTextStyleDefaults()
    if (!response.success || !response.defaults) {
      return false
    }

    applyTextStyleDefaults(parseTextStyleDefaults(response.defaults))
    return true
  } catch (error) {
    console.warn('[textStyleDefaults] 重新加载默认值失败，继续使用当前缓存:', error)
    return false
  }
}

applyTextStyleDefaults(parseTextStyleDefaults(rawDefaults))

export function getTextStyleDefaults(): TextStyleDefaults {
  return { ...TEXT_STYLE_DEFAULTS }
}

export function normalizeTextStyleSettings(
  style?: Partial<TextStyleSettings> | null
): TextStyleSettings {
  const base = getTextStyleDefaults()
  if (!style) {
    return base
  }

  return {
    fontSize: style.fontSize !== undefined ? expectPositiveInt(style.fontSize, 'fontSize') : base.fontSize,
    autoFontSize: style.autoFontSize !== undefined ? expectBoolean(style.autoFontSize, 'autoFontSize') : base.autoFontSize,
    fontFamily: style.fontFamily !== undefined ? expectNonEmptyString(style.fontFamily, 'fontFamily') : base.fontFamily,
    layoutDirection: style.layoutDirection !== undefined ? expectTextDirection(style.layoutDirection, 'layoutDirection') : base.layoutDirection,
    textColor: style.textColor !== undefined ? expectNonEmptyString(style.textColor, 'textColor') : base.textColor,
    fillColor: style.fillColor !== undefined ? expectNonEmptyString(style.fillColor, 'fillColor') : base.fillColor,
    strokeEnabled: style.strokeEnabled !== undefined ? expectBoolean(style.strokeEnabled, 'strokeEnabled') : base.strokeEnabled,
    strokeColor: style.strokeColor !== undefined ? expectNonEmptyString(style.strokeColor, 'strokeColor') : base.strokeColor,
    strokeWidth: style.strokeWidth !== undefined ? expectNonNegativeInt(style.strokeWidth, 'strokeWidth') : base.strokeWidth,
    inpaintMethod: style.inpaintMethod !== undefined ? expectInpaintMethod(style.inpaintMethod, 'inpaintMethod') : base.inpaintMethod,
    useAutoTextColor: style.useAutoTextColor !== undefined ? expectBoolean(style.useAutoTextColor, 'useAutoTextColor') : base.useAutoTextColor,
    lineSpacing: style.lineSpacing !== undefined ? expectPositiveFloat(style.lineSpacing, 'lineSpacing') : base.lineSpacing,
    textAlign: style.textAlign !== undefined ? expectTextAlign(style.textAlign, 'textAlign') : base.textAlign,
  }
}

export function resolveBubbleTextDirection(
  layoutDirection?: TextDirection | null
): 'vertical' | 'horizontal' {
  return layoutDirection === 'horizontal' ? 'horizontal' : 'vertical'
}

export function getBubbleDefaultsFromTextStyle(
  style?: Partial<TextStyleSettings> | null
): BubbleGlobalDefaults {
  const normalized = normalizeTextStyleSettings(style)
  return {
    fontSize: normalized.fontSize,
    fontFamily: normalized.fontFamily,
    textDirection: resolveBubbleTextDirection(normalized.layoutDirection),
    textColor: normalized.textColor,
    fillColor: normalized.fillColor,
    inpaintMethod: normalized.inpaintMethod,
    strokeEnabled: normalized.strokeEnabled,
    strokeColor: normalized.strokeColor,
    strokeWidth: normalized.strokeWidth,
    lineSpacing: normalized.lineSpacing,
    textAlign: normalized.textAlign
  }
}

type ImageTextStyleFields = Pick<
  ImageData,
  | 'fontSize'
  | 'autoFontSize'
  | 'fontFamily'
  | 'layoutDirection'
  | 'textColor'
  | 'fillColor'
  | 'inpaintMethod'
  | 'strokeEnabled'
  | 'strokeColor'
  | 'strokeWidth'
  | 'lineSpacing'
  | 'textAlign'
  | 'useAutoTextColor'
>

export function getImageTextStyleDefaults(): ImageTextStyleFields {
  const normalized = getTextStyleDefaults()
  return {
    fontSize: normalized.fontSize,
    autoFontSize: normalized.autoFontSize,
    fontFamily: normalized.fontFamily,
    layoutDirection: normalized.layoutDirection,
    textColor: normalized.textColor,
    fillColor: normalized.fillColor,
    inpaintMethod: normalized.inpaintMethod,
    strokeEnabled: normalized.strokeEnabled,
    strokeColor: normalized.strokeColor,
    strokeWidth: normalized.strokeWidth,
    lineSpacing: normalized.lineSpacing,
    textAlign: normalized.textAlign,
    useAutoTextColor: normalized.useAutoTextColor
  }
}

export function normalizeImageTextStyleFields(
  image?: Partial<ImageData> | null
): ImageTextStyleFields {
  const base = getImageTextStyleDefaults()
  if (!image) {
    return base
  }

  return {
    fontSize: image.fontSize !== undefined ? expectPositiveInt(image.fontSize, 'fontSize') : base.fontSize,
    autoFontSize: image.autoFontSize !== undefined ? expectBoolean(image.autoFontSize, 'autoFontSize') : base.autoFontSize,
    fontFamily: image.fontFamily !== undefined ? expectNonEmptyString(image.fontFamily, 'fontFamily') : base.fontFamily,
    layoutDirection: image.layoutDirection !== undefined ? expectTextDirection(image.layoutDirection, 'layoutDirection') : base.layoutDirection,
    textColor: image.textColor !== undefined ? expectNonEmptyString(image.textColor, 'textColor') : base.textColor,
    fillColor: image.fillColor !== undefined ? expectNonEmptyString(image.fillColor, 'fillColor') : base.fillColor,
    inpaintMethod: image.inpaintMethod !== undefined ? expectInpaintMethod(image.inpaintMethod, 'inpaintMethod') : base.inpaintMethod,
    strokeEnabled: image.strokeEnabled !== undefined ? expectBoolean(image.strokeEnabled, 'strokeEnabled') : base.strokeEnabled,
    strokeColor: image.strokeColor !== undefined ? expectNonEmptyString(image.strokeColor, 'strokeColor') : base.strokeColor,
    strokeWidth: image.strokeWidth !== undefined ? expectNonNegativeInt(image.strokeWidth, 'strokeWidth') : base.strokeWidth,
    lineSpacing: image.lineSpacing !== undefined ? expectPositiveFloat(image.lineSpacing, 'lineSpacing') : base.lineSpacing,
    textAlign: image.textAlign !== undefined ? expectTextAlign(image.textAlign, 'textAlign') : base.textAlign,
    useAutoTextColor: image.useAutoTextColor !== undefined ? expectBoolean(image.useAutoTextColor, 'useAutoTextColor') : base.useAutoTextColor
  }
}
