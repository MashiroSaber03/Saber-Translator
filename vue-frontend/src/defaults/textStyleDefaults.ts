import textStyleDefaultsJson from '../../../src/shared/text_style_defaults_factory.json'
import type { BubbleGlobalDefaults, TextDirection } from '@/types/bubble'
import type { ImageData } from '@/types/image'
import type { TextStyleSettings } from '@/types/settings'
import { getTextStyleDefaults as fetchTextStyleDefaults } from '@/api/config'

export type TextStyleDefaults = TextStyleSettings

const rawDefaults = textStyleDefaultsJson as Record<string, unknown>

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

type TextStyleFieldSource = Partial<Record<keyof TextStyleSettings, unknown>>

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

function readTextStyleField<T>(
  source: TextStyleFieldSource,
  fieldName: keyof TextStyleSettings,
  parser: (value: unknown, fieldName: string) => T,
  base?: TextStyleSettings
): T {
  const value = source[fieldName]
  if (value !== undefined) {
    return parser(value, fieldName)
  }
  if (base) {
    return base[fieldName] as T
  }
  return failInvalidConfig(`${fieldName} is required`)
}

function buildTextStyleFields(
  source: Record<string, unknown> | Partial<TextStyleSettings> | Partial<ImageTextStyleFields>,
  base?: TextStyleSettings
): TextStyleSettings {
  const fields = source as TextStyleFieldSource
  return {
    fontSize: readTextStyleField(fields, 'fontSize', expectPositiveInt, base),
    autoFontSize: readTextStyleField(fields, 'autoFontSize', expectBoolean, base),
    fontFamily: readTextStyleField(fields, 'fontFamily', expectNonEmptyString, base),
    layoutDirection: readTextStyleField(fields, 'layoutDirection', expectTextDirection, base),
    textColor: readTextStyleField(fields, 'textColor', expectNonEmptyString, base),
    fillColor: readTextStyleField(fields, 'fillColor', expectNonEmptyString, base),
    strokeEnabled: readTextStyleField(fields, 'strokeEnabled', expectBoolean, base),
    strokeColor: readTextStyleField(fields, 'strokeColor', expectNonEmptyString, base),
    strokeWidth: readTextStyleField(fields, 'strokeWidth', expectNonNegativeInt, base),
    inpaintMethod: readTextStyleField(fields, 'inpaintMethod', expectInpaintMethod, base),
    useAutoTextColor: readTextStyleField(fields, 'useAutoTextColor', expectBoolean, base),
    lineSpacing: readTextStyleField(fields, 'lineSpacing', expectPositiveFloat, base),
    textAlign: readTextStyleField(fields, 'textAlign', expectTextAlign, base)
  }
}

function parseTextStyleDefaults(source: Record<string, unknown> | TextStyleSettings): TextStyleDefaults {
  return buildTextStyleFields(source)
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
  } catch {
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
  return buildTextStyleFields(style ?? {}, base)
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

export function getImageTextStyleDefaults(): ImageTextStyleFields {
  return buildTextStyleFields({}, getTextStyleDefaults())
}

export function normalizeImageTextStyleFields(
  image?: Partial<ImageData> | Record<string, unknown> | null
): ImageTextStyleFields {
  return buildTextStyleFields(image ?? {}, getTextStyleDefaults())
}
