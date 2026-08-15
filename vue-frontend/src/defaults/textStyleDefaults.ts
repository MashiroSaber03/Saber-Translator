import textStyleDefaultsJson from '../../../src/shared/text_style_defaults_factory.json'
import type { ImageData } from '@/types/image'
import type { TextStyleSettings } from '@/types/settings'

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
const TEXT_STYLE_FIELDS = [
  'fontSize',
  'autoFontSize',
  'fontFamily',
  'layoutDirection',
  'textColor',
  'fillColor',
  'strokeEnabled',
  'strokeColor',
  'strokeWidth',
  'inpaintMethod',
  'useAutoTextColor',
  'lineSpacing',
  'textAlign',
] as const satisfies readonly (keyof TextStyleSettings)[]
const COLOR_PATTERN = /^#[0-9A-Fa-f]{6}$/

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
  if (typeof value === 'number' && Number.isInteger(value) && value > 0) {
    return value
  }
  return failInvalidConfig(`${fieldName} must be a positive integer`)
}

function expectNonNegativeInt(value: unknown, fieldName: string): number {
  if (typeof value === 'number' && Number.isInteger(value) && value >= 0) {
    return value
  }
  return failInvalidConfig(`${fieldName} must be a non-negative integer`)
}

function expectPositiveFloat(value: unknown, fieldName: string): number {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    return value
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

const BUNDLED_TEXT_STYLE_DEFAULTS = Object.freeze(parseTextStyleDefaults(rawDefaults))

export const TEXT_STYLE_DEFAULTS: Readonly<TextStyleDefaults> = BUNDLED_TEXT_STYLE_DEFAULTS

export function getTextStyleDefaults(): TextStyleDefaults {
  return { ...TEXT_STYLE_DEFAULTS }
}

export function normalizeTextStyleSettings(
  style?: Partial<TextStyleSettings> | null
): TextStyleSettings {
  const base = getTextStyleDefaults()
  return buildTextStyleFields(style ?? {}, base)
}

export function parseCompleteTextStyleSettings(value: unknown): TextStyleSettings {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return failInvalidConfig('backend text style must be an object')
  }
  const source = value as Record<string, unknown>
  const actual = Object.keys(source)
  if (
    actual.length !== TEXT_STYLE_FIELDS.length
    || TEXT_STYLE_FIELDS.some(field => !Object.prototype.hasOwnProperty.call(source, field))
  ) {
    return failInvalidConfig('backend text style fields are incomplete')
  }
  const parsed = buildTextStyleFields(source)
  for (const [field, color] of [
    ['textColor', parsed.textColor],
    ['fillColor', parsed.fillColor],
    ['strokeColor', parsed.strokeColor],
  ] as const) {
    if (!COLOR_PATTERN.test(color)) {
      return failInvalidConfig(`${field} must be a #RRGGBB color`)
    }
  }
  return parsed
}

export function normalizeImageTextStyleFields(
  image?: Partial<ImageData> | Record<string, unknown> | null
): ImageTextStyleFields {
  return buildTextStyleFields(image ?? {}, getTextStyleDefaults())
}
