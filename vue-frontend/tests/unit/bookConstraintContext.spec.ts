import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { DEFAULT_AUTO_GLOSSARY_PROMPT } from '@/constants'

import {
  createEmptyBookTranslationConstraints,
  normalizeBookTranslationConstraints,
  resolveConstraintPayloadForTranslation,
} from '@/utils/bookTranslationConstraints'

describe('bookTranslationConstraints helpers', () => {
  it('keeps constraint normalization fixtures typed without broad casts', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/bookConstraintContext.spec.ts'), 'utf8')

    expect(source).not.toContain('as ' + 'any')
  })

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
      glossary: { enabled: false, autoExtractEnabled: false, autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT, entries: [] },
      non_translate: { enabled: false, entries: [] },
    })
  })

  it('filters blank constraint rows during normalization', () => {
    expect(
      normalizeBookTranslationConstraints({
        glossary: {
          enabled: true,
          autoExtractEnabled: true,
          autoExtractPrompt: '提取当前漫画中的人名',
          entries: [
            { source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' },
            { source: '', target: '', note: '', matchMode: 'text' },
          ],
        },
        non_translate: {
          enabled: true,
          entries: [
            { pattern: '<keep>', note: '', matchMode: 'text' },
            { pattern: '   ', note: '', matchMode: 'regex' },
          ],
        },
      }),
    ).toEqual({
      glossary: {
        enabled: true,
        autoExtractEnabled: true,
        autoExtractPrompt: '提取当前漫画中的人名',
        entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' }],
      },
      non_translate: {
        enabled: true,
        entries: [{ pattern: '<keep>', note: '', matchMode: 'text' }],
      },
    })
  })

  it('falls back to the default auto glossary prompt when the saved prompt is blank', () => {
    expect(
      normalizeBookTranslationConstraints({
        glossary: {
          enabled: true,
          autoExtractEnabled: true,
          autoExtractPrompt: '   ',
          entries: [],
        },
        non_translate: {
          enabled: false,
          entries: [],
        },
      }),
    ).toEqual({
      glossary: {
        enabled: true,
        autoExtractEnabled: true,
        autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT,
        entries: [],
      },
      non_translate: {
        enabled: false,
        entries: [],
      },
    })
  })

  it('does not normalize obsolete snake_case auto glossary fields in the frontend schema', () => {
    expect(
      normalizeBookTranslationConstraints({
        glossary: {
          enabled: true,
          auto_extract_enabled: true,
          auto_extract_prompt: '旧字段提示词',
          entries: [],
        },
        non_translate: {
          enabled: false,
          entries: [],
        },
      }),
    ).toEqual({
      glossary: {
        enabled: true,
        autoExtractEnabled: false,
        autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT,
        entries: [],
      },
      non_translate: {
        enabled: false,
        entries: [],
      },
    })
  })
})
