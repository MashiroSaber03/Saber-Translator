import { describe, expect, it } from 'vitest'

import { getHqModeConfig } from '@/composables/translation/modes/hqMode'
import { getProofreadModeConfig } from '@/composables/translation/modes/proofreadMode'
import { getRemoveTextModeConfig } from '@/composables/translation/modes/removeTextMode'
import { getStandardModeConfig } from '@/composables/translation/modes/standardMode'

describe('page selection pipeline configs', () => {
  it('passes discrete page selections through all mode config factories', () => {
    const selectedPages = [1, 3, 8, 10]

    expect(getStandardModeConfig('selection', { pageSelection: selectedPages })).toMatchObject({
      mode: 'standard',
      scope: 'selection',
      pageSelection: selectedPages,
    })

    expect(getHqModeConfig('selection', { pageSelection: selectedPages })).toMatchObject({
      mode: 'hq',
      scope: 'selection',
      pageSelection: selectedPages,
    })

    expect(getProofreadModeConfig('selection', { pageSelection: selectedPages })).toMatchObject({
      mode: 'proofread',
      scope: 'selection',
      pageSelection: selectedPages,
    })

    expect(getRemoveTextModeConfig('selection', { pageSelection: selectedPages })).toMatchObject({
      mode: 'removeText',
      scope: 'selection',
      pageSelection: selectedPages,
    })
  })
})
