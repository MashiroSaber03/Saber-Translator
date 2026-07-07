import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import LorebookTreeBranch from '@/components/insight/studio/LorebookTreeBranch.vue'
import LorebookTreeEditor from '@/components/insight/studio/LorebookTreeEditor.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import type { LorebookEntryNode } from '@/types/characterStudio'

function buildEntry(id = 'entry-alpha'): LorebookEntryNode {
  return {
    id,
    comment: '测试条目',
    keys: [],
    secondary_keys: [],
    content: '',
    enabled: true,
    constant: false,
    selective: true,
    priority: 100,
    position: 'before_char',
    depth: 4,
    probability: 100,
    prevent_recursion: true,
    children: [],
  }
}

describe('LorebookTreeEditor import flow', () => {
  it('renders the empty tree state through the product empty-state pattern', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/LorebookTreeEditor.vue'),
      'utf8',
    )
    expect(source).toContain('ProductEmptyState')
    expect(source).not.toContain('class="placeholder"')
    expect(source).not.toContain('--lorebook-tree-editor-empty-text')
    expect(source).not.toMatch(/\.placeholder\b/)

    const wrapper = mount(LorebookTreeEditor, {
      props: {
        entries: [],
        importing: false,
      },
    })

    expect(wrapper.getComponent(ProductEmptyState).props()).toMatchObject({
      iconName: 'book-open',
      role: 'note',
      size: 'compact',
      title: '暂无世界书条目',
    })
  })

  it('emits selected worldbook file to parent', async () => {
    const wrapper = mount(LorebookTreeEditor, {
      props: {
        entries: [],
        importing: false,
      },
    })

    const input = wrapper.find('input[type="file"]')
    expect(input.exists()).toBe(true)

    const file = new File(['{"entries":[]}'], 'worldbook.json', { type: 'application/json' })
    Object.defineProperty(input.element, 'files', {
      value: [file],
      configurable: true,
    })
    await input.trigger('change')

    const emitted = wrapper.emitted('import-worldbook')
    expect(emitted).toBeTruthy()
    expect(emitted?.[0]?.[0]).toBe(file)
  })

  it('disables impossible branch move actions at sibling boundaries', () => {
    const wrapper = mount(LorebookTreeBranch, {
      props: {
        entry: buildEntry(),
        index: 1,
        siblingCount: 2,
      },
    })

    const downButton = wrapper.findAll('button').find(button => button.text() === '下移')
    expect(downButton?.attributes('disabled')).toBeDefined()
  })
})
