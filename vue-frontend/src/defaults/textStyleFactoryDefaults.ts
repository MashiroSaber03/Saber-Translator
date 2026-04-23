import textStyleFactoryDefaultsJson from '../../../src/shared/text_style_defaults_factory.json'
import type { TextStyleSettings } from '@/types/settings'
import { normalizeTextStyleSettings } from '@/defaults/textStyleDefaults'

const FACTORY_TEXT_STYLE_DEFAULTS = Object.freeze(
  normalizeTextStyleSettings(textStyleFactoryDefaultsJson as Partial<TextStyleSettings>)
)

export function getFactoryTextStyleDefaults(): TextStyleSettings {
  return { ...FACTORY_TEXT_STYLE_DEFAULTS }
}
