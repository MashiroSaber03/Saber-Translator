import { describe, expect, it } from 'vitest'
import { DEFAULT_AUTO_GLOSSARY_PROMPT } from '@/constants'
import { createEmptyBookTranslationConstraints } from '@/utils/bookTranslationConstraints'

describe('book translation constraint mapping', () => {
  it('creates the canonical empty current document', () => {
    expect(createEmptyBookTranslationConstraints()).toEqual({
      glossary: {
        enabled: false,
        autoExtractEnabled: false,
        autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT,
        entries: [],
      },
      nonTranslate: { enabled: false, entries: [] },
    })
  })
})
