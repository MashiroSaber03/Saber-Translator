import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import TranslationConstraintTable from '@/components/settings/shared/TranslationConstraintTable.vue'

describe('TranslationConstraintTable', () => {
  it('exposes sortable headers as named button controls', async () => {
    const wrapper = mount(TranslationConstraintTable, {
      props: {
        modelValue: [
          { source: 'Beta', target: '乙' },
          { source: 'Alpha', target: '甲' },
        ],
        columns: [
          { key: 'source', label: '原文' },
          { key: 'target', label: '译文' },
        ],
        emptyRow: { source: '', target: '' },
        exportBaseName: '术语表',
        rowKeyPrefix: 'constraint-test',
      },
    })

    const sortBySource = wrapper.get('thead button[aria-label="按原文排序"]')

    await sortBySource.trigger('click')

    const firstSourceInput = wrapper.get('tbody tr input')
    expect((firstSourceInput.element as HTMLInputElement).value).toBe('Alpha')
  })
})
