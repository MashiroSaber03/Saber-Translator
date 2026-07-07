import type {
  TextStyleSettings,
  TranslationSettings,
  TranslationSettingsUpdates,
} from '@/types/settings'
import type { InpaintMethod, TextAlign, TextDirection } from '@/types/bubble'

export interface SessionUiSettingsTarget {
  settings: TranslationSettings
  updateSettings(updates: TranslationSettingsUpdates): void
  updateTextStyle(updates: Partial<TextStyleSettings>): void
}

const VALID_INPAINT_METHODS: readonly InpaintMethod[] = ['solid', 'lama_mpe', 'litelama']
const VALID_LAYOUT_DIRECTIONS: readonly TextDirection[] = ['vertical', 'horizontal', 'auto']
const VALID_TEXT_ALIGNMENTS: readonly TextAlign[] = ['start', 'center', 'end']

function hasSetting(payload: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(payload, key)
}

function readString(payload: Record<string, unknown>, key: string, fallback: string): string {
  const value = payload[key]
  return typeof value === 'string' && value ? value : fallback
}

function readNumber(payload: Record<string, unknown>, key: string, fallback: number): number {
  const value = payload[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function readBoolean(payload: Record<string, unknown>, key: string, fallback: boolean): boolean {
  const value = payload[key]
  return typeof value === 'boolean' ? value : fallback
}

function readEnum<T extends string>(
  payload: Record<string, unknown>,
  key: string,
  allowed: readonly T[],
  fallback: T,
): T {
  const value = payload[key]
  return typeof value === 'string' && allowed.includes(value as T) ? value as T : fallback
}

function assignIfPresent<TValue>(
  target: Partial<TextStyleSettings>,
  payload: Record<string, unknown>,
  key: keyof TextStyleSettings,
  read: () => TValue,
): void {
  if (!hasSetting(payload, String(key))) return
  Object.assign(target, { [key]: read() })
}

export function applySessionUiSettings(
  uiSettings: Record<string, unknown> | null | undefined,
  target: SessionUiSettingsTarget,
): void {
  if (!uiSettings) return

  const languageUpdates: TranslationSettingsUpdates = {}
  if (typeof uiSettings.targetLanguage === 'string' && uiSettings.targetLanguage) {
    languageUpdates.targetLanguage = uiSettings.targetLanguage
  }
  if (typeof uiSettings.sourceLanguage === 'string' && uiSettings.sourceLanguage) {
    languageUpdates.sourceLanguage = uiSettings.sourceLanguage
  }
  if (Object.keys(languageUpdates).length > 0) {
    target.updateSettings(languageUpdates)
  }

  const currentTextStyle = target.settings.textStyle
  const textStyleUpdates: Partial<TextStyleSettings> = {}
  assignIfPresent(textStyleUpdates, uiSettings, 'fontSize', () => readNumber(uiSettings, 'fontSize', currentTextStyle.fontSize))
  assignIfPresent(textStyleUpdates, uiSettings, 'autoFontSize', () => readBoolean(uiSettings, 'autoFontSize', currentTextStyle.autoFontSize))
  assignIfPresent(textStyleUpdates, uiSettings, 'fontFamily', () => readString(uiSettings, 'fontFamily', currentTextStyle.fontFamily))
  assignIfPresent(textStyleUpdates, uiSettings, 'layoutDirection', () => readEnum(uiSettings, 'layoutDirection', VALID_LAYOUT_DIRECTIONS, currentTextStyle.layoutDirection))
  assignIfPresent(textStyleUpdates, uiSettings, 'textColor', () => readString(uiSettings, 'textColor', currentTextStyle.textColor))
  assignIfPresent(textStyleUpdates, uiSettings, 'fillColor', () => readString(uiSettings, 'fillColor', currentTextStyle.fillColor))
  if (hasSetting(uiSettings, 'useInpaintingMethod')) {
    textStyleUpdates.inpaintMethod = readEnum(
      uiSettings,
      'useInpaintingMethod',
      VALID_INPAINT_METHODS,
      currentTextStyle.inpaintMethod,
    )
  }
  assignIfPresent(textStyleUpdates, uiSettings, 'strokeEnabled', () => readBoolean(uiSettings, 'strokeEnabled', currentTextStyle.strokeEnabled))
  assignIfPresent(textStyleUpdates, uiSettings, 'strokeColor', () => readString(uiSettings, 'strokeColor', currentTextStyle.strokeColor))
  assignIfPresent(textStyleUpdates, uiSettings, 'strokeWidth', () => readNumber(uiSettings, 'strokeWidth', currentTextStyle.strokeWidth))
  assignIfPresent(textStyleUpdates, uiSettings, 'lineSpacing', () => readNumber(uiSettings, 'lineSpacing', currentTextStyle.lineSpacing))
  assignIfPresent(textStyleUpdates, uiSettings, 'textAlign', () => readEnum(uiSettings, 'textAlign', VALID_TEXT_ALIGNMENTS, currentTextStyle.textAlign))
  assignIfPresent(textStyleUpdates, uiSettings, 'useAutoTextColor', () => readBoolean(uiSettings, 'useAutoTextColor', currentTextStyle.useAutoTextColor))

  if (Object.keys(textStyleUpdates).length > 0) {
    target.updateTextStyle(textStyleUpdates)
  }
}
