import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'

export function createEmptyBookTranslationConstraints(): BookTranslationConstraints {
  return {
    glossary: {
      enabled: false,
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
  const glossaryEntries = Array.isArray(payload?.glossary?.entries)
    ? payload!.glossary!.entries.filter((entry) => String(entry?.source ?? '').trim() && String(entry?.target ?? '').trim())
    : defaults.glossary.entries
  const nonTranslateEntries = Array.isArray(payload?.non_translate?.entries)
    ? payload!.non_translate!.entries.filter((entry) => String(entry?.pattern ?? '').trim())
    : defaults.non_translate.entries
  return {
    glossary: {
      enabled: Boolean(payload?.glossary?.enabled),
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
