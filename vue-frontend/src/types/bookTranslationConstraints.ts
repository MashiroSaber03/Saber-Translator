import type {
  GlossarySettings,
  NonTranslateSettings,
} from './translationConstraints'

export interface BookTranslationConstraints {
  glossary: GlossarySettings
  non_translate: NonTranslateSettings
}
