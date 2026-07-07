import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { getHqModeConfig } from '@/composables/translation/modes/hqMode'
import { getProofreadModeConfig } from '@/composables/translation/modes/proofreadMode'
import { getRemoveTextModeConfig } from '@/composables/translation/modes/removeTextMode'
import { getStandardModeConfig } from '@/composables/translation/modes/standardMode'

describe('page selection pipeline configs', () => {
  it('keeps mode config factories compact and helper-backed', () => {
    const files = [
      'src/composables/translation/modes/index.ts',
      'src/composables/translation/modes/standardMode.ts',
      'src/composables/translation/modes/hqMode.ts',
      'src/composables/translation/modes/proofreadMode.ts',
      'src/composables/translation/modes/removeTextMode.ts',
    ]

    for (const file of files) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).not.toContain('/**')
      expect(source, file).not.toContain('模式配置')
      expect(source, file).not.toContain('@param')
      expect(source, file).not.toContain('    ')
    }

    const hqSource = readFileSync(resolve(process.cwd(), 'src/composables/translation/modes/hqMode.ts'), 'utf8')
    const proofreadSource = readFileSync(resolve(process.cwd(), 'src/composables/translation/modes/proofreadMode.ts'), 'utf8')

    expect(hqSource).toContain("createBatchModeConfig('hq'")
    expect(proofreadSource).toContain("createBatchModeConfig('proofread'")
    expect(`${hqSource}\n${proofreadSource}`).not.toContain('?? 3')
    expect(`${hqSource}\n${proofreadSource}`).not.toContain('?? 2')
    expect(`${hqSource}\n${proofreadSource}`).not.toContain('?? 10')
  })

  it('passes discrete page selections through all mode config factories', () => {
    const selectedPages = [1, 3, 8, 10]

    expect(getStandardModeConfig('selection', { pageSelection: { pages: selectedPages } })).toMatchObject({
      mode: 'standard',
      scope: 'selection',
      pageSelection: { pages: selectedPages },
    })

    expect(getHqModeConfig('selection', { pageSelection: { pages: selectedPages } })).toMatchObject({
      mode: 'hq',
      scope: 'selection',
      pageSelection: { pages: selectedPages },
    })

    expect(getProofreadModeConfig('selection', { pageSelection: { pages: selectedPages } })).toMatchObject({
      mode: 'proofread',
      scope: 'selection',
      pageSelection: { pages: selectedPages },
    })

    expect(getRemoveTextModeConfig('selection', { pageSelection: { pages: selectedPages } })).toMatchObject({
      mode: 'removeText',
      scope: 'selection',
      pageSelection: { pages: selectedPages },
    })
  })

  it('keeps batch defaults centralized while allowing mode overrides', () => {
    expect(getHqModeConfig()).toMatchObject({
      mode: 'hq',
      scope: 'all',
      batchOptions: {
        batchSize: 3,
        maxRetries: 2,
        rpmLimit: 10,
      },
    })

    expect(getProofreadModeConfig('selection', {
      batchSize: 5,
      maxRetries: 4,
      rpmLimit: 12,
      pageSelection: { pages: [2] },
    })).toEqual({
      mode: 'proofread',
      scope: 'selection',
      pageSelection: { pages: [2] },
      batchOptions: {
        batchSize: 5,
        maxRetries: 4,
        rpmLimit: 12,
      },
    })
  })
})
