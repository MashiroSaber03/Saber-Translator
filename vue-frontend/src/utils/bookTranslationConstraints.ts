import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'
import { DEFAULT_AUTO_GLOSSARY_PROMPT } from '@/constants'

type PlainRecord = Record<string, unknown>

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

function isPlainRecord(value: unknown): value is PlainRecord {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function readString(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function readMatchMode(value: unknown): 'text' | 'regex' {
  return value === 'regex' ? 'regex' : 'text'
}

function normalizeGlossaryEntries(value: unknown): BookTranslationConstraints['glossary']['entries'] {
  if (!Array.isArray(value)) {
    return []
  }

  return value.flatMap((entry) => {
    if (!isPlainRecord(entry)) {
      return []
    }

    const source = readString(entry.source)
    const target = readString(entry.target)
    if (!source.trim() || !target.trim()) {
      return []
    }

    return [{
      source,
      target,
      note: readString(entry.note),
      matchMode: readMatchMode(entry.matchMode),
    }]
  })
}

function normalizeNonTranslateEntries(value: unknown): BookTranslationConstraints['non_translate']['entries'] {
  if (!Array.isArray(value)) {
    return []
  }

  return value.flatMap((entry) => {
    if (!isPlainRecord(entry)) {
      return []
    }

    const pattern = readString(entry.pattern)
    if (!pattern.trim()) {
      return []
    }

    return [{
      pattern,
      note: readString(entry.note),
      matchMode: readMatchMode(entry.matchMode),
    }]
  })
}

export function normalizeBookTranslationConstraints(
  payload?: unknown,
): BookTranslationConstraints {
  const defaults = createEmptyBookTranslationConstraints()
  const payloadRecord = isPlainRecord(payload) ? payload : {}
  const glossaryPayload = isPlainRecord(payloadRecord.glossary) ? payloadRecord.glossary : {}
  const nonTranslatePayload = isPlainRecord(payloadRecord.non_translate)
    ? payloadRecord.non_translate
    : {}
  const glossaryEntries = normalizeGlossaryEntries(glossaryPayload.entries)
  const nonTranslateEntries = normalizeNonTranslateEntries(nonTranslatePayload.entries)

  return {
    glossary: {
      enabled: Boolean(glossaryPayload.enabled),
      autoExtractEnabled: Boolean(glossaryPayload.autoExtractEnabled),
      autoExtractPrompt: readString(glossaryPayload.autoExtractPrompt).trim() || DEFAULT_AUTO_GLOSSARY_PROMPT,
      entries: glossaryEntries.length > 0
        ? glossaryEntries
        : defaults.glossary.entries,
    },
    non_translate: {
      enabled: Boolean(nonTranslatePayload.enabled),
      entries: nonTranslateEntries.length > 0
        ? nonTranslateEntries
        : defaults.non_translate.entries,
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
