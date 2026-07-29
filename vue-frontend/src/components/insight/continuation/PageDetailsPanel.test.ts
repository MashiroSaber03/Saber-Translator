import { mount } from '@vue/test-utils'
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import PageDetailsPanel from './PageDetailsPanel.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

const componentSourcePath = resolve(process.cwd(), 'src/components/insight/continuation/PageDetailsPanel.vue')
const parentSourcePath = resolve(process.cwd(), 'src/components/insight/ContinuationPanel.vue')
const storyTypesSourcePath = resolve(process.cwd(), 'src/components/insight/continuation/pageStoryTypes.ts')

function createPage() {
  return {
    page_number: 1,
    continuity_text: '承接',
    story_text: '剧情',
    dialogue_text: '对白',
    characters: ['主角'],
    character_forms: [],
    final_prompt: '',
    image_url: '',
    previous_url: '',
    status: 'generated',
  }
}

describe('PageDetailsPanel', () => {
  it('uses product cards, fields, and action rows for generated page details', () => {
    const page = createPage()
    const wrapper = mount(PageDetailsPanel, {
      props: {
        pages: [page],
        isGenerating: false,
      },
    })

    expect(wrapper.findAllComponents(ProductRecordCard)).toHaveLength(1)

    const fields = wrapper.findAllComponents(UiField)
    expect(fields.map(field => field.props('label'))).toEqual([
      '上一页剧情承接',
      '本页剧情',
      '关键对白',
      '角色（逗号分隔）',
    ])
    expect(fields.map(field => field.props('variant'))).toEqual([
      'settings',
      'settings',
      'settings',
      'settings',
    ])

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props('ariaLabel')).toBe('页面剧情操作')
  })

  it('renders page statuses through the product chip contract', () => {
    const wrapper = mount(PageDetailsPanel, {
      props: {
        pages: [createPage()],
        isGenerating: false,
      },
    })
    const source = readFileSync(componentSourcePath, 'utf8')

    const statusChip = wrapper.getComponent(ProductChipList)
    expect(statusChip.props('ariaLabel')).toBe('页面 1 状态')
    expect(statusChip.props('items')).toEqual([
      {
        id: 'generated',
        label: '已生成',
        tone: 'success',
      },
    ])
    expect(source).not.toContain('class="page-status"')
    expect(source).not.toMatch(/\.page-status(?:\.|[\s,{])/)
  })

  it('keeps generated page card headers responsive in narrow panels', () => {
    const source = readFileSync(componentSourcePath, 'utf8')
    const pageHeaderStyle = source.match(/\.page-details-panel__page-header \{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''
    const pageTitleStyle = source.match(/\.page-details-panel__page-title \{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''

    expect(pageHeaderStyle).toContain('flex-wrap: wrap')
    expect(pageHeaderStyle).toContain('min-width: 0')
    expect(source).toContain('class="page-details-panel__page-title"')
    expect(source).not.toContain('.page-details-panel__page-header h4')
    expect(pageTitleStyle).toContain('margin: 0')
  })

  it('keeps local page-detail hooks owner-prefixed', () => {
    const source = readFileSync(componentSourcePath, 'utf8')

    expect(source).toContain('class="page-details-panel__page-card"')
    expect(source).toContain('class="page-details-panel__page-header"')
    expect(source).toContain('class="page-details-panel__page-title"')
    expect(source).toContain('class="page-details-panel__field-input"')
    expect(source).not.toContain('class="page-card"')
    expect(source).not.toContain('class="page-header"')
    expect(source).not.toContain('class="field-input"')
  })

  it('does not override shared button primitive variables at the panel root', () => {
    const source = readFileSync(componentSourcePath, 'utf8')
    const rootStyle = source.match(/\.page-details-panel \{(?<body>[\s\S]*?)\n\}/)

    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-button-/)
  })

  it('uses public panel textarea and input sizing instead of root primitive form-control overrides', () => {
    const page = createPage()
    const wrapper = mount(PageDetailsPanel, {
      props: {
        pages: [page],
        isGenerating: false,
      },
    })
    const source = readFileSync(componentSourcePath, 'utf8')
    const rootStyle = source.match(/\.page-details-panel \{(?<body>[\s\S]*?)\n\}/)

    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-(input|textarea)-/)
    expect(wrapper.findAllComponents(UiTextarea).map(field => field.props('size'))).toEqual(['sm', 'sm', 'sm'])
    expect(wrapper.findAllComponents(UiTextarea).map(field => field.props('variant'))).toEqual([
      'panel',
      'panel',
      'panel',
    ])
    expect(wrapper.getComponent(UiInput).props('size')).toBe('sm')
  })

  it('emits typed story edits without mutating page props', async () => {
    const page = createPage()
    const wrapper = mount(PageDetailsPanel, {
      props: {
        pages: [page],
        isGenerating: false,
      },
    })

    await wrapper.get('#continuation-page-1-continuity').setValue('新的承接')
    expect(page.continuity_text).toBe('承接')
    expect(wrapper.emitted('story-change')?.at(-1)).toEqual([1, 'continuity_text', '新的承接'])

    await wrapper.get('#continuation-page-1-story').setValue('新的剧情')
    expect(page.story_text).toBe('剧情')
    expect(wrapper.emitted('story-change')?.at(-1)).toEqual([1, 'story_text', '新的剧情'])

    await wrapper.get('#continuation-page-1-dialogue').setValue('新的对白')
    expect(page.dialogue_text).toBe('对白')
    expect(wrapper.emitted('story-change')?.at(-1)).toEqual([1, 'dialogue_text', '新的对白'])

    await wrapper.get('#continuation-page-1-characters').setValue('主角, 配角')

    expect(page.characters).toEqual(['主角'])
    expect(wrapper.emitted('story-change')?.at(-1)).toEqual([1, 'characters', ['主角', '配角']])
  })

  it('shares the page story edit payload contract with the continuation parent', () => {
    const childSource = readFileSync(componentSourcePath, 'utf8')
    const parentSource = readFileSync(parentSourcePath, 'utf8')
    const hasTypeOwner = existsSync(storyTypesSourcePath)
    const typeSource = hasTypeOwner ? readFileSync(storyTypesSourcePath, 'utf8') : ''

    expect(hasTypeOwner).toBe(true)
    expect(typeSource).toContain('export type PageStoryField')
    expect(typeSource).toContain('export type PageStoryValue')

    for (const [source, importPath] of [
      [childSource, "from './pageStoryTypes'"],
      [parentSource, "from './continuation/pageStoryTypes'"],
    ]) {
      expect(source).toContain(importPath)
      expect(source).not.toMatch(/type PageStoryField\s*=/)
      expect(source).not.toMatch(/type PageStoryValue\s*=/)
    }
  })

  it('renders the empty page-detail state through the product empty-state pattern', async () => {
    const wrapper = mount(PageDetailsPanel, {
      props: {
        pages: [],
        isGenerating: false,
      },
    })

    const emptyState = wrapper.getComponent(ProductEmptyState)
    expect(emptyState.props('iconName')).toBe('file-text')
    expect(emptyState.props('title')).toBe('尚未生成页面剧情')
    expect(wrapper.find('.empty-state').exists()).toBe(false)

    const generateButton = wrapper.findAll('button').find(button => button.text().includes('生成页面剧情'))
    expect(generateButton).toBeTruthy()
    await generateButton!.trigger('click')
    expect(wrapper.emitted('generate-details')).toEqual([[]])
  })

  it('delegates page status tones to the product chip contract', () => {
    const source = readFileSync(componentSourcePath, 'utf8')

    expect(source).not.toMatch(/--page-details-panel-status-[^:]+:\s*#[0-9a-fA-F]{3,8}/)
    expect(source).toContain('ProductChipItem')
    expect(source).toContain("pending: 'warning'")
    expect(source).toContain("generating: 'primary'")
    expect(source).toContain("generated: 'success'")
    expect(source).toContain("failed: 'danger'")
  })
})
