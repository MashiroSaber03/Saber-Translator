import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import TranslationConstraintTable from '@/components/settings/shared/TranslationConstraintTable.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductSearchField from '@/components/product/ProductSearchField.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import {
  exportRowsToJson,
  importRowsFromJson,
  validateRegexEntries,
} from '@/utils/translationConstraintTable'

describe('TranslationConstraintTable', () => {
  it('keeps the utility row contract named and owner-scoped', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/utils/translationConstraintTable.ts'),
      'utf8',
    )

    expect(source).toContain('export type TranslationConstraintTableRow = Record<string, string>')
    expect(source).not.toContain('type TableRow = Record<string, string>')
  })

  it('normalizes imported JSON rows and validates regex rows through the utility owner', () => {
    const columns = [
      { key: 'source', label: '原文' },
      { key: 'target', label: '译文' },
      { key: 'matchMode', label: '匹配方式' },
    ]

    const rows = importRowsFromJson(
      JSON.stringify([
        { 原文: 'Saber', target: '剑士', 匹配方式: 'text' },
        { source: '^Rin(', target: '凛', matchMode: 'regex' },
      ]),
      columns,
    )

    expect(rows).toEqual([
      { source: 'Saber', target: '剑士', matchMode: 'text' },
      { source: '^Rin(', target: '凛', matchMode: 'regex' },
    ])
    expect(exportRowsToJson(rows)).toContain('"source": "Saber"')
    expect(validateRegexEntries(rows, { patternField: 'source' })).toContain('第 2 行正则无效')
  })

  it('rejects malformed import rows instead of dropping or stringifying them', () => {
    const columns = [{ key: 'source', label: '原文' }]

    expect(() => importRowsFromJson('[null]', columns)).toThrow('JSON 第 1 行必须是对象')
    expect(() => importRowsFromJson('[{"source":42}]', columns)).toThrow(
      '第 1 行原文必须是文本',
    )
  })

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

  it('uses fixed select primitives for select columns', () => {
    const wrapper = mount(TranslationConstraintTable, {
      props: {
        modelValue: [
          { source: 'Saber', matchMode: 'text' },
        ],
        columns: [
          { key: 'source', label: '原文' },
          {
            key: 'matchMode',
            label: '匹配方式',
            type: 'select',
            options: [
              { label: '文本', value: 'text' },
              { label: '正则', value: 'regex' },
            ],
          },
        ],
        emptyRow: { source: '', matchMode: 'text' },
        exportBaseName: '术语表',
        rowKeyPrefix: 'constraint-test',
      },
    })

    const select = wrapper.getComponent(UiSelect)
    expect(select.props('modelValue')).toBe('text')
    expect(select.props('options')).toEqual([
      { label: '文本', value: 'text' },
      { label: '正则', value: 'regex' },
    ])

    select.vm.$emit('change', 'regex')

    expect(wrapper.emitted('update:modelValue')?.[0]).toEqual([[
      { source: 'Saber', matchMode: 'regex' },
    ]])
  })

  it('updates text cells through typed input model events', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/shared/TranslationConstraintTable.vue'),
      'utf8',
    )

    expect(source).not.toContain('@input="\n                updateCell')
    expect(source).not.toContain('($event.target as HTMLInputElement).value')
    expect(source).toContain('@update:model-value="updateCell(originalIndex, column.key, $event)"')
    expect(source).toContain(':key="`${rowKeyPrefix}-${originalIndex}`"')
    expect(source).not.toContain(':key="`${rowKeyPrefix}-${index}`"')

    const wrapper = mount(TranslationConstraintTable, {
      props: {
        modelValue: [{ source: 'Saber', target: '剑士' }],
        columns: [
          { key: 'source', label: '原文' },
          { key: 'target', label: '译文' },
        ],
        emptyRow: { source: '', target: '' },
        exportBaseName: '术语表',
        rowKeyPrefix: 'constraint-test',
      },
    })

    const sourceCellInput = wrapper
      .findAllComponents(UiInput)
      .find(input => input.props('modelValue') === 'Saber')
    expect(sourceCellInput).toBeDefined()

    sourceCellInput?.vm.$emit('update:modelValue', 'Rin')

    expect(wrapper.emitted('update:modelValue')?.[0]).toEqual([[
      { source: 'Rin', target: '剑士' },
    ]])
  })

  it('renders table filtering through the product search field', async () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/shared/TranslationConstraintTable.vue'),
      'utf8',
    )
    expect(source).toContain('class="translation-constraint-table"')
    expect(source).toContain('class="translation-constraint-table__toolbar"')
    expect(source).toContain('class="translation-constraint-table__search-field"')
    expect(source).toContain('class="translation-constraint-table__table"')
    expect(source).toContain('class="translation-constraint-table__sort-action"')
    expect(source).toContain('class="translation-constraint-table__select-cell"')
    expect(source).toContain('class="translation-constraint-table__cell-field"')
    expect(source).toContain('class="translation-constraint-table__action-cell"')
    expect(source).not.toContain('class="constraint-table"')
    expect(source).not.toContain('class="constraint-toolbar"')
    expect(source).not.toContain('class="constraint-search-field"')
    expect(source).not.toContain('class="settings-table"')
    expect(source).not.toContain('class="sortable-header-button"')
    expect(source).not.toContain('class="select-cell"')
    expect(source).not.toContain('class="constraint-cell-field"')
    expect(source).not.toContain('class="action-cell"')

    const wrapper = mount(TranslationConstraintTable, {
      props: {
        modelValue: [
          { source: 'Saber', target: '剑士' },
          { source: 'Rin', target: '凛' },
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

    const searchField = wrapper.getComponent(ProductSearchField)
    expect(searchField.props()).toMatchObject({
      ariaLabel: '搜索约束表格',
      placeholder: '搜索表格内容...',
    })

    searchField.vm.$emit('update:modelValue', 'Rin')
    await wrapper.vm.$nextTick()

    expect(wrapper.findAll('tbody tr')).toHaveLength(1)
    expect((wrapper.get('tbody tr input').element as HTMLInputElement).value).toBe('Rin')
    expect(wrapper.find('.constraint-search').exists()).toBe(false)
  })

  it('renders table toolbar actions through the product action row', () => {
    const wrapper = mount(TranslationConstraintTable, {
      props: {
        modelValue: [{ source: 'Saber', target: '剑士' }],
        columns: [
          { key: 'source', label: '原文' },
          { key: 'target', label: '译文' },
        ],
        emptyRow: { source: '', target: '' },
        exportBaseName: '术语表',
        rowKeyPrefix: 'constraint-test',
      },
    })

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props()).toMatchObject({
      ariaLabel: '约束表格操作',
      justify: 'start',
    })
    expect(actionRow.text()).toContain('导入 JSON')
    expect(wrapper.find('.constraint-actions').exists()).toBe(false)
  })

  it('routes import files through the typed file-input primitive boundary', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/shared/TranslationConstraintTable.vue'),
      'utf8',
    )

    expect(source).toContain('@files-change="handleImport($event, \'json\')"')
    expect(source).toContain('@files-change="handleImport($event, \'xlsx\')"')
    expect(source).toContain('hidden')
    expect(source).toContain('InstanceType<typeof UiFileInput>')
    expect(source).not.toContain('ref<HTMLInputElement')
    expect(source).not.toContain('@change="handleImport')
    expect(source).not.toContain('event.target as HTMLInputElement')
    expect(source).not.toContain('input.files')
    expect(source).not.toContain('input.value =')
    expect(source).not.toContain('hidden-input')
  })

  it('maps editable field colors to semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/shared/TranslationConstraintTable.vue'),
      'utf8',
    )
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(styleBlock).toContain('--color-border-input')
    expect(styleBlock).toContain('--color-focus-brand-subtle')
  })
})
