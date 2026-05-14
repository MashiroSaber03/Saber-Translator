import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'
import { DEFAULT_AUTO_GLOSSARY_PROMPT } from '@/constants'

export function createEmptyBookTranslationConstraints(): BookTranslationConstraints {
  return {
    glossary: {
      enabled: false,
      autoExtractEnabled: false,
      autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT,
      entries: [],
    },
    non_translate: {
      enabled: false,
      entries: [],
    },
  }
}

export function normalizeBookTranslationConstraints(
  payload?: Partial<BookTranslationConstraints> | null,
): BookTranslationConstraints {
  const defaults = createEmptyBookTranslationConstraints()
  const glossaryPayload = payload?.glossary as (
    Partial<BookTranslationConstraints['glossary']> & {
      auto_extract_enabled?: boolean
      auto_extract_prompt?: string
    }
  ) | undefined
  const glossaryEntries = Array.isArray(payload?.glossary?.entries)
    ? payload!.glossary!.entries.filter((entry) => String(entry?.source ?? '').trim() && String(entry?.target ?? '').trim())
    : defaults.glossary.entries
  const nonTranslateEntries = Array.isArray(payload?.non_translate?.entries)
    ? payload!.non_translate!.entries.filter((entry) => String(entry?.pattern ?? '').trim())
    : defaults.non_translate.entries
  return {
    glossary: {
      enabled: Boolean(glossaryPayload?.enabled),
      autoExtractEnabled: Boolean(glossaryPayload?.autoExtractEnabled ?? glossaryPayload?.auto_extract_enabled),
      autoExtractPrompt: String(glossaryPayload?.autoExtractPrompt ?? glossaryPayload?.auto_extract_prompt ?? '').trim() || DEFAULT_AUTO_GLOSSARY_PROMPT,
      entries: glossaryEntries.map((entry) => ({ ...entry })),
    },
    non_translate: {
      enabled: Boolean(payload?.non_translate?.enabled),
      entries: nonTranslateEntries.map((entry) => ({ ...entry })),
    },
  }
}

export function resolveConstraintPayloadForTranslation(options: {
  isBookshelfMode: boolean
  constraints: BookTranslationConstraints
}): {
  glossary_settings?: BookTranslationConstraints['glossary']
  non_translate_settings?: BookTranslationConstraints['non_translate']
} {
  if (!options.isBookshelfMode) {
    return {}
  }
  return {
    glossary_settings: options.constraints.glossary,
    non_translate_settings: options.constraints.non_translate,
  }
}
