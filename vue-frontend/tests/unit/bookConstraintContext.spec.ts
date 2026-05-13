import { describe, expect, it } from 'vitest'

import {
  createEmptyBookTranslationConstraints,
  normalizeBookTranslationConstraints,
  resolveConstraintPayloadForTranslation,
} from '@/utils/bookTranslationConstraints'

describe('bookTranslationConstraints helpers', () => {
  it('returns empty payload outside bookshelf mode', () => {
    const payload = resolveConstraintPayloadForTranslation({
      isBookshelfMode: false,
      constraints: {
        glossary: {
          enabled: true,
          entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' }],
        },
        non_translate: {
          enabled: true,
          entries: [{ pattern: '<keep>', note: '', matchMode: 'text' }],
        },
      },
    })

    expect(payload).toEqual({})
  })

  it('returns glossary and non-translate payload in bookshelf mode', () => {
    const constraints = {
      glossary: {
        enabled: true,
        entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' as const }],
      },
      non_translate: {
        enabled: true,
        entries: [{ pattern: '<keep>', note: '', matchMode: 'text' as const }],
      },
    }

    const payload = resolveConstraintPayloadForTranslation({
      isBookshelfMode: true,
      constraints,
    })

    expect(payload).toEqual({
      glossary_settings: constraints.glossary,
      non_translate_settings: constraints.non_translate,
    })
  })

  it('creates empty default constraints structure', () => {
    expect(createEmptyBookTranslationConstraints()).toEqual({
      glossary: { enabled: false, autoExtractEnabled: false, entries: [] },
      non_translate: { enabled: false, entries: [] },
    })
  })

  it('filters blank constraint rows during normalization', () => {
    expect(
      normalizeBookTranslationConstraints({
        glossary: {
          enabled: true,
          autoExtractEnabled: true,
          entries: [
            { source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' },
            { source: '', target: '', note: '', matchMode: 'text' } as any,
          ],
        },
        non_translate: {
          enabled: true,
          entries: [
            { pattern: '<keep>', note: '', matchMode: 'text' },
            { pattern: '   ', note: '', matchMode: 'regex' } as any,
          ],
        },
      }),
    ).toEqual({
      glossary: {
        enabled: true,
        autoExtractEnabled: true,
        entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' }],
      },
      non_translate: {
        enabled: true,
        entries: [{ pattern: '<keep>', note: '', matchMode: 'text' }],
      },
    })
  })
})
