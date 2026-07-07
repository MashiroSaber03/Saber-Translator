import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductSearchField from '@/components/product/ProductSearchField.vue'
import CandidateListPane from '@/components/insight/studio/CandidateListPane.vue'
import CharacterStudioSidebar from '@/components/insight/studio/CharacterStudioSidebar.vue'
import DiagnosticsPanel from '@/components/insight/studio/DiagnosticsPanel.vue'
import DocumentListPane from '@/components/insight/studio/DocumentListPane.vue'

function cssRule(source: string, selector: string): string {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return source.match(new RegExp(`${escapedSelector}\\s*{([\\s\\S]*?)}`))?.[1] ?? ''
}

describe('CharacterStudioSidebar pending feedback', () => {
  it('shows loading copy for manual create, import, document open, and candidate creation', () => {
    const wrapper = mount(CharacterStudioSidebar, {
      props: {
        documents: [
          {
            id: 'doc_alpha',
            title: '阿尔法',
            origin: 'manual',
            source_character: null,
            updated_at: '2026-05-15T00:00:00',
            tags: [],
            is_favorite: false,
            has_avatar: false,
            sample_pages: [],
          },
        ],
        candidates: [
          {
            name: '候选角色',
            aliases: [],
            first_appearance: 1,
            dialogue_count: 2,
            has_dialogues: true,
            sample_pages: [1],
          },
        ],
        search: '',
        currentDocumentId: 'doc_alpha',
        hasTimeline: true,
        workspaceLoading: false,
        creatingManual: true,
        importingFile: true,
        openingDocumentId: 'doc_alpha',
        creatingCandidateName: '候选角色',
      },
    })

    expect(wrapper.text()).toContain('新建中...')
    expect(wrapper.text()).toContain('导入中...')
    expect(wrapper.text()).toContain('打开中...')
    expect(wrapper.text()).toContain('创建中...')
    expect(wrapper.text()).not.toContain('候选仅预填角色名')
  })

  it('uses the product action-row contract and semantic owner tokens', () => {
    const wrapper = mount(CharacterStudioSidebar, {
      props: {
        documents: [],
        candidates: [],
        search: '',
        currentDocumentId: '',
        hasTimeline: false,
        workspaceLoading: false,
        creatingManual: false,
        importingFile: false,
        openingDocumentId: '',
        creatingCandidateName: '',
      },
    })

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props('ariaLabel')).toBe('角色资源操作')
    expect(actionRow.props('justify')).toBe('start')

    const searchField = wrapper.getComponent(ProductSearchField)
    expect(searchField.props()).toMatchObject({
      ariaLabel: '搜索角色资源',
      placeholder: '搜索角色 / 标签 / 来源',
      disabled: false,
    })
    searchField.vm.$emit('update:modelValue', 'Saber')
    expect(wrapper.emitted('update:search')?.[0]).toEqual(['Saber'])

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioSidebar.vue'),
      'utf8',
    )
    expect(source).not.toContain('variant="toolbar"')
    expect(source).not.toContain('action-primary')
    expect(source).not.toContain('action-ghost')
    expect(source).not.toContain('候选仅预填角色名')
    expect(source).not.toMatch(/\.toolbar-copy p\s*\{/)
    expect(source).not.toContain('--ui-input-')
    expect(source).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)

    for (const oldClass of [
      'sidebar-shell',
      'sidebar-toolbar',
      'toolbar-copy',
      'toolbar-actions',
      'search-input',
      'action-row',
      'resource-create-action',
      'resource-import-action',
      'sidebar-content',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldClass}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldClass}\\b`))
    }

    for (const ownerClass of [
      'character-studio-sidebar',
      'character-studio-sidebar__toolbar',
      'character-studio-sidebar__toolbar-copy',
      'character-studio-sidebar__title',
      'character-studio-sidebar__actions',
      'character-studio-sidebar__search',
      'character-studio-sidebar__action-row',
      'character-studio-sidebar__create-action',
      'character-studio-sidebar__import-action',
      'character-studio-sidebar__content',
    ]) {
      expect(source).toContain(ownerClass)
    }

    expect(source).not.toContain('.character-studio-sidebar__toolbar-copy h2')
  })

  it('uses the typed file-input primitive boundary for card imports', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioSidebar.vue'),
      'utf8',
    )

    expect(source).toContain('@files-change="handleFileSelect"')
    expect(source).not.toContain('ref<HTMLInputElement')
    expect(source).not.toContain('event.target as HTMLInputElement')
    expect(source).not.toContain('target.files')
    expect(source).not.toContain("target.value = ''")
  })

  it('uses product record and chip contracts in sidebar support panels', () => {
    const panelContracts = [
      {
        file: 'src/components/insight/studio/CandidateListPane.vue',
        required: ['ProductRecordCard', 'ProductActionRow'],
      },
      {
        file: 'src/components/insight/studio/DocumentListPane.vue',
        required: ['ProductRecordCard', 'ProductChipList'],
      },
      {
        file: 'src/components/insight/studio/DiagnosticsPanel.vue',
        required: ['ProductRecordCard', 'ProductChipList'],
      },
    ]

    for (const contract of panelContracts) {
      const source = readFileSync(resolve(process.cwd(), contract.file), 'utf8')

      for (const requiredImport of contract.required) {
        expect(source, contract.file).toContain(requiredImport)
      }

      expect(source, contract.file).not.toContain('variant="toolbar"')
      expect(source, contract.file).not.toMatch(/class="[^"]*(?:create-btn|opening-pill|favorite-pill|source-pill|check-pill)/)
      expect(source, contract.file).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    }
  })

  it('lets resource list rows wrap metadata and actions in narrow drawers', () => {
    const documentSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/DocumentListPane.vue'),
      'utf8',
    )
    const candidateSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CandidateListPane.vue'),
      'utf8',
    )

    expect(cssRule(documentSource, '.document-list-pane__item-body')).toContain('flex-wrap: wrap')
    expect(cssRule(documentSource, '.document-list-pane__item-main')).toContain('min-width: 0')
    expect(cssRule(candidateSource, '.candidate-list-pane__row-body')).toContain('flex-wrap: wrap')
    expect(cssRule(candidateSource, '.candidate-list-pane__candidate-main')).toContain('min-width: 0')
  })

  it('keeps sidebar support panel hooks under their component owners', () => {
    const ownerContracts = [
      {
        file: 'src/components/insight/studio/CandidateListPane.vue',
        required: ['candidate-list-pane', 'candidate-list-pane__head', 'candidate-list-pane__title', 'candidate-list-pane__count', 'candidate-list-pane__list', 'candidate-list-pane__row', 'candidate-list-pane__candidate-main', 'candidate-list-pane__candidate-name'],
        legacy: ['pane', 'pane-head', 'list', 'candidate-row', 'candidate-row-body', 'candidate-main', 'candidate-meta'],
      },
      {
        file: 'src/components/insight/studio/DiagnosticsPanel.vue',
        required: ['diagnostics-panel', 'diagnostics-panel__summary-grid', 'diagnostics-panel__summary-card', 'diagnostics-panel__check-list'],
        legacy: ['diagnostics-shell', 'summary-grid', 'summary-card', 'label', 'block', 'danger', 'warning', 'checks-block', 'check-list'],
      },
    ]

    for (const contract of ownerContracts) {
      const source = readFileSync(resolve(process.cwd(), contract.file), 'utf8')
      const classTokens = [...source.matchAll(/class="([^"]+)"/g)]
        .flatMap(match => match[1]!.split(/\s+/).filter(Boolean))

      for (const requiredClass of contract.required) {
        expect(classTokens, contract.file).toContain(requiredClass)
      }

      for (const legacyClass of contract.legacy) {
        expect(classTokens, contract.file).not.toContain(legacyClass)
      }
    }
  })

  it('keeps document list pane hooks under the component owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/DocumentListPane.vue'),
      'utf8',
    )
    const classTokens = [...source.matchAll(/class="([^"]+)"/g)]
      .flatMap(match => match[1]!.split(/\s+/).filter(Boolean))

    expect(classTokens).toContain('document-list-pane')
    expect(classTokens).toContain('document-list-pane__head')
    expect(classTokens).toContain('document-list-pane__title')
    expect(classTokens).toContain('document-list-pane__count')
    expect(classTokens).toContain('document-list-pane__list')
    expect(classTokens).toContain('document-list-pane__item')
    expect(classTokens).toContain('document-list-pane__item-main')
    expect(classTokens).toContain('document-list-pane__item-title')

    for (const legacyClass of [
      'pane',
      'pane-head',
      'list',
      'item',
      'item-body',
      'item-main',
      'item-meta',
      'item-badges',
      'active',
      'opening',
    ]) {
      expect(classTokens).not.toContain(legacyClass)
    }
    expect(source).not.toMatch(/['"](?:active|opening)['"]\s*:/)
    expect(source).not.toContain('.document-list-pane__head h3')
    expect(source).not.toContain('.document-list-pane__head span')
    expect(source).not.toContain('.document-list-pane__item-main strong')
  })

  it('exposes the current document row as the current item', () => {
    const wrapper = mount(DocumentListPane, {
      props: {
        documents: [
          {
            id: 'doc_alpha',
            title: '阿尔法',
            origin: 'manual',
            source_character: null,
            updated_at: '2026-05-15T00:00:00',
            tags: [],
            is_favorite: false,
            has_avatar: false,
            sample_pages: [],
          },
          {
            id: 'doc_beta',
            title: '贝塔',
            origin: 'imported',
            source_character: '贝塔',
            updated_at: '2026-05-16T00:00:00',
            tags: [],
            is_favorite: true,
            has_avatar: true,
            sample_pages: [2],
          },
        ],
        currentDocumentId: 'doc_alpha',
        openingDocumentId: '',
      },
    })

    const rows = wrapper.findAll('.document-list-pane__item')
    expect(rows[0]!.attributes('aria-current')).toBe('true')
    expect(rows[1]!.attributes('aria-current')).toBeUndefined()
  })

  it('renders sidebar support empty states through compact product empty states', () => {
    const panelFiles = [
      'src/components/insight/studio/CandidateListPane.vue',
      'src/components/insight/studio/DocumentListPane.vue',
      'src/components/insight/studio/DiagnosticsPanel.vue',
    ]

    for (const file of panelFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, file).toContain('ProductEmptyState')
      expect(source, file).not.toContain('empty-copy')
    }

    const noTimeline = mount(CandidateListPane, {
      props: {
        candidates: [],
        hasTimeline: false,
        creatingCandidateName: '',
      },
    })
    const noTimelineState = noTimeline.getComponent(ProductEmptyState)
    expect(noTimelineState.props()).toMatchObject({
      iconName: 'bar-chart',
      role: 'note',
      size: 'compact',
      title: '暂无增强时间线',
    })

    const noCandidates = mount(CandidateListPane, {
      props: {
        candidates: [],
        hasTimeline: true,
        creatingCandidateName: '',
      },
    })
    expect(noCandidates.getComponent(ProductEmptyState).props('title')).toBe('没有可用候选角色')

    const emptyDocuments = mount(DocumentListPane, {
      props: {
        documents: [],
        currentDocumentId: '',
        openingDocumentId: '',
      },
    })
    expect(emptyDocuments.getComponent(ProductEmptyState).props()).toMatchObject({
      iconName: 'file-text',
      size: 'compact',
      title: '当前书还没有角色文档',
    })

    const emptyDiagnostics = mount(DiagnosticsPanel, {
      props: {
        diagnostics: null,
      },
    })
    expect(emptyDiagnostics.getComponent(ProductEmptyState).props()).toMatchObject({
      iconName: 'scan-search',
      size: 'compact',
      title: '还没有运行诊断',
    })
  })
})
