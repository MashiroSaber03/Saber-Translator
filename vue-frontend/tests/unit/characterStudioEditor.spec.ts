import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it } from 'vitest'
import { defineComponent, h, ref } from 'vue'
import { mount } from '@vue/test-utils'
import type { VueWrapper } from '@vue/test-utils'
import CharacterStudioEditor from '@/components/insight/studio/CharacterStudioEditor.vue'
import GreetingWorkbench from '@/components/insight/studio/GreetingWorkbench.vue'
import RegexWorkbench from '@/components/insight/studio/RegexWorkbench.vue'
import TaskWorkbench from '@/components/insight/studio/TaskWorkbench.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import type { CharacterStudioDocument } from '@/types/characterStudio'

function cssRules(source: string, selector: string): string[] {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return Array.from(source.matchAll(new RegExp(`(?:^|\\n)${escapedSelector}\\s*{([\\s\\S]*?)}`, 'g')))
    .map(match => match[1] ?? '')
}

function buildDocument(): CharacterStudioDocument {
  return {
    id: 'doc-alpha',
    bookId: 'book-demo',
    origin: {
      type: 'analysis',
      source_character: '上杉风太郎',
    },
    status: {
      is_favorite: false,
      frozen_sections: [],
      last_validated_at: null,
    },
    meta: {
      title: '上杉风太郎',
      tags: ['主角', '分析生成'],
      created_at: '2026-05-15T00:00:00',
      updated_at: '2026-05-15T00:00:00',
    },
    avatar: {
      asset_path: null,
    },
    identity: {
      name: '上杉风太郎',
      aliases: ['风太郎'],
      description: '一个认真但嘴硬的学生。',
      personality: '冷静，略带防备心。',
      scenario: '当前处于学园日常阶段。',
    },
    coreMessages: {
      first_message: '我是上杉风太郎。',
      message_example: '<START>\n{{user}}: 你好\n{{char}}: 你好。',
      alternate_greetings: ['今天也要继续努力。'],
      system_prompt: '保持角色稳定。',
      post_history_instructions: '保持叙事连续。',
      creator_notes: '测试备注',
      character_version: '2.0.0',
    },
    lorebook: {
      name: '风太郎世界书',
      entries: [],
    },
    regexScripts: [],
    stateTasks: [],
    chatPreset: {
      opening_mode: 'first_message',
    },
    grounding: {
      timeline_mode: 'enhanced',
      sample_pages: [1, 3],
      relationships: [],
      key_moments: [],
    },
    exportArtifacts: {},
  }
}

describe('CharacterStudioEditor tabs', () => {
  let document: CharacterStudioDocument

  beforeEach(() => {
    document = buildDocument()
  })

  function mountHarness() {
    return mount(defineComponent({
      components: { CharacterStudioEditor },
      setup() {
        const currentDocument = ref<CharacterStudioDocument | null>(document)
        const activeTab = ref<'overview' | 'character' | 'greetings' | 'lorebook' | 'scripts' | 'export'>('overview')
        const activeScriptTab = ref<'regex' | 'tasks'>('regex')
        return () => h(CharacterStudioEditor, {
          document: currentDocument.value,
          avatarUrl: '',
          diagnostics: null,
          activeTab: activeTab.value,
          activeScriptTab: activeScriptTab.value,
          pendingState: {
            generatingSection: null,
            validating: false,
            importingWorldbook: false,
            deleting: false,
            saving: false,
            downloadingFormat: null,
          },
          'onUpdate:document': (value: CharacterStudioDocument | null) => { currentDocument.value = value },
          'onUpdate:activeTab': (value: typeof activeTab.value) => { activeTab.value = value },
          'onUpdate:activeScriptTab': (value: typeof activeScriptTab.value) => { activeScriptTab.value = value },
        })
      },
    }), {
      global: {
        stubs: {
          LorebookTreeEditor: {
            template: '<div class="lorebook-stub">世界书树编辑器</div>',
          },
        },
      },
    })
  }

  it('maps editor owner colors through semantic tokens without a parent token warehouse', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioEditor.vue'),
      'utf8',
    )
    const childFiles = [
      'src/components/insight/studio/DiagnosticsPanel.vue',
      'src/components/insight/studio/GreetingWorkbench.vue',
      'src/components/insight/studio/LorebookTreeBranch.vue',
      'src/components/insight/studio/LorebookTreeEditor.vue',
      'src/components/insight/studio/RegexWorkbench.vue',
      'src/components/insight/studio/TaskWorkbench.vue',
      'src/components/insight/studio/editor/StudioEditorSectionPanel.vue',
      'src/components/insight/studio/editor/StudioHeroSection.vue',
      'src/components/insight/studio/editor/StudioOverviewTab.vue',
    ]

    expect(source).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(source).not.toMatch(/--character-studio-editor-/)
    for (const file of childFiles) {
      const childSource = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(childSource, file).not.toMatch(/--character-studio-editor-/)
    }
  })

  it('keeps editor grids responsive to split-pane width instead of only viewport breakpoints', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioEditor.vue'),
      'utf8',
    )

    for (const selector of ['.studio-editor__onboarding-tip-grid', '.studio-editor__download-grid', '.studio-editor__form-grid']) {
      expect(
        cssRules(source, selector).some(rule => rule.includes('repeat(auto-fit, minmax(min(100%, 280px), 1fr))')),
        selector,
      ).toBe(true)
    }
    expect(source).not.toContain('repeat(2, minmax(0, 1fr))')
  })

  it('keeps Studio workbench forms responsive inside resizable panes', () => {
    for (const file of [
      'src/components/insight/studio/RegexWorkbench.vue',
      'src/components/insight/studio/TaskWorkbench.vue',
      'src/components/insight/studio/LorebookTreeBranch.vue',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      const gridSelector = file.endsWith('RegexWorkbench.vue')
        ? '.regex-workbench__grid'
        : file.endsWith('TaskWorkbench.vue')
          ? '.task-workbench__grid'
          : '.lorebook-tree-branch__grid'
      expect(
        cssRules(source, gridSelector).some(rule => rule.includes('repeat(auto-fit, minmax(min(100%, 280px), 1fr))')),
        file,
      ).toBe(true)
      expect(source, file).not.toContain('repeat(2, minmax(0, 1fr))')
    }

    const diagnosticsSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/DiagnosticsPanel.vue'),
      'utf8',
    )
    expect(
      cssRules(diagnosticsSource, '.diagnostics-panel__summary-grid').some(rule => rule.includes('repeat(auto-fit, minmax(min(100%, 160px), 1fr))')),
    ).toBe(true)
    expect(diagnosticsSource).not.toContain('repeat(3, minmax(0, 1fr))')
  })

  it('keeps Studio workbench responsive rules separated by layout type', () => {
    for (const [file, owner] of [
      ['src/components/insight/studio/RegexWorkbench.vue', 'regex-workbench'],
      ['src/components/insight/studio/TaskWorkbench.vue', 'task-workbench'],
    ] as const) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, file).not.toContain(`.${owner}__head,\r\n  .${owner}__card-head,\r\n  .${owner}__grid`)
      expect(source, file).not.toContain(`.${owner}__head,\n  .${owner}__card-head,\n  .${owner}__grid`)
    }
  })

  it('lets Studio workbench headers wrap instead of relying on viewport breakpoints', () => {
    const greetingSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/GreetingWorkbench.vue'),
      'utf8',
    )
    const regexSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/RegexWorkbench.vue'),
      'utf8',
    )
    const taskSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/TaskWorkbench.vue'),
      'utf8',
    )
    const branchSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/LorebookTreeBranch.vue'),
      'utf8',
    )
    const treeSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/LorebookTreeEditor.vue'),
      'utf8',
    )

    expect(cssRules(greetingSource, '.greeting-workbench__section-head').some(rule => rule.includes('flex-wrap: wrap'))).toBe(true)
    expect(cssRules(greetingSource, '.greeting-workbench__alternate-head').some(rule => rule.includes('flex-wrap: wrap'))).toBe(true)
    expect(cssRules(regexSource, '.regex-workbench__head').some(rule => rule.includes('flex-wrap: wrap'))).toBe(true)
    expect(cssRules(regexSource, '.regex-workbench__card-head').some(rule => rule.includes('flex-wrap: wrap'))).toBe(true)
    expect(cssRules(taskSource, '.task-workbench__head').some(rule => rule.includes('flex-wrap: wrap'))).toBe(true)
    expect(cssRules(taskSource, '.task-workbench__card-head').some(rule => rule.includes('flex-wrap: wrap'))).toBe(true)
    expect(cssRules(branchSource, '.lorebook-tree-branch__summary').some(rule => rule.includes('flex-wrap: wrap'))).toBe(true)
    expect(cssRules(treeSource, '.lorebook-tree-editor__head').some(rule => rule.includes('flex-wrap: wrap'))).toBe(true)
  })

  it('keeps editor parent, overview, and lorebook editor hooks under their component owners', () => {
    const ownerContracts = [
      {
        file: 'src/components/insight/studio/CharacterStudioEditor.vue',
        required: ['studio-editor__shell', 'studio-editor__panel-stack', 'studio-editor__form-grid', 'studio-editor__onboarding-tip-title', 'studio-editor__onboarding-tip-description', 'studio-editor__download-card', 'studio-editor__download-title', 'studio-editor__download-description'],
        legacy: ['editor-onboarding', 'onboarding-tip-grid', 'onboarding-tip-card', 'editor-shell', 'panel-stack', 'form-grid', 'full', 'option-row', 'toggle-chip', 'script-panel', 'download-grid', 'download-card', 'download-icon'],
      },
      {
        file: 'src/components/insight/studio/editor/StudioOverviewTab.vue',
        required: ['studio-overview-tab', 'studio-overview-tab__summary-grid', 'studio-overview-tab__summary-card', 'studio-overview-tab__summary-value', 'studio-overview-tab__summary-description', 'studio-overview-tab__quick-card', 'studio-overview-tab__quick-title', 'studio-overview-tab__quick-description', 'studio-overview-tab__freeze-item'],
        legacy: ['panel-stack', 'summary-grid', 'summary-card', 'summary-label', 'workspace-row', 'quick-grid', 'quick-card', 'quick-icon', 'freeze-grid', 'freeze-item', 'freeze-item-label', 'freeze-item-control', 'review-summary', 'review-list', 'suggestions', 'single'],
      },
      {
        file: 'src/components/insight/studio/editor/StudioEditorSectionPanel.vue',
        required: ['studio-editor-section-panel', 'studio-editor-section-panel__head', 'studio-editor-section-panel__title', 'studio-editor-section-panel__description'],
        legacy: ['section-panel', 'section-head'],
      },
      {
        file: 'src/components/insight/studio/LorebookTreeEditor.vue',
        required: ['lorebook-tree-editor', 'lorebook-tree-editor__head', 'lorebook-tree-editor__title', 'lorebook-tree-editor__description', 'lorebook-tree-editor__tree-list'],
        legacy: ['workshop-card', 'section-head', 'tree-list'],
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

    for (const selector of [
      '.studio-editor__onboarding-tip-card strong',
      '.studio-editor__onboarding-tip-card p',
      '.studio-editor__download-card strong',
      '.studio-editor__download-card p',
      '.studio-overview-tab__summary-card strong',
      '.studio-overview-tab__summary-card p',
      '.studio-overview-tab__quick-card strong',
      '.studio-overview-tab__quick-card p',
      '.studio-editor-section-panel__head h3',
      '.studio-editor-section-panel__head p',
      '.lorebook-tree-editor__head h3',
      '.lorebook-tree-editor__head p',
    ]) {
      for (const file of [
        'src/components/insight/studio/CharacterStudioEditor.vue',
        'src/components/insight/studio/editor/StudioOverviewTab.vue',
        'src/components/insight/studio/editor/StudioEditorSectionPanel.vue',
        'src/components/insight/studio/LorebookTreeEditor.vue',
      ]) {
        const source = readFileSync(resolve(process.cwd(), file), 'utf8')
        expect(source, file).not.toContain(selector)
      }
    }
  })

  it('keeps lorebook tree branch hooks under the component owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/LorebookTreeBranch.vue'),
      'utf8',
    )
    const classTokens = [...source.matchAll(/class="([^"]+)"/g)]
      .flatMap(match => match[1]!.split(/\s+/).filter(Boolean))

    expect(classTokens).toContain('lorebook-tree-branch')
    expect(classTokens).toContain('lorebook-tree-branch__summary')
    expect(classTokens).toContain('lorebook-tree-branch__grid')
    expect(classTokens).toContain('lorebook-tree-branch__meta-item')
    expect(classTokens).toContain('lorebook-tree-branch__children')

    for (const legacyClass of [
      'branch-node',
      'node-details',
      'node-summary',
      'summary-main',
      'title-input',
      'meta-line',
      'summary-actions',
      'node-body',
      'workbench-grid',
      'full',
      'toggles',
      'children',
    ]) {
      expect(classTokens).not.toContain(legacyClass)
    }
  })

  it('keeps Studio workbench child hooks under their component owners', () => {
    const ownerContracts = [
      {
        file: 'src/components/insight/studio/GreetingWorkbench.vue',
        required: ['greeting-workbench', 'greeting-workbench__section-head', 'greeting-workbench__section-title', 'greeting-workbench__section-description', 'greeting-workbench__textarea', 'greeting-workbench__alternate-card', 'greeting-workbench__alternate-name'],
        legacy: ['workbench', 'hero-block', 'hero-head', 'list-block', 'list-head', 'workbench-textarea', 'alternate-list', 'alternate-card', 'alternate-head', 'title', 'index-chip'],
      },
      {
        file: 'src/components/insight/studio/RegexWorkbench.vue',
        required: ['regex-workbench', 'regex-workbench__head', 'regex-workbench__title', 'regex-workbench__description', 'regex-workbench__script-card', 'regex-workbench__grid'],
        legacy: ['workbench', 'workbench-head', 'script-list', 'script-card', 'card-head', 'title-input', 'workbench-grid', 'full', 'toggles'],
      },
      {
        file: 'src/components/insight/studio/TaskWorkbench.vue',
        required: ['task-workbench', 'task-workbench__head', 'task-workbench__title', 'task-workbench__description', 'task-workbench__task-card', 'task-workbench__grid'],
        legacy: ['workbench', 'workbench-head', 'task-list', 'task-card', 'card-head', 'title-input', 'workbench-grid', 'full', 'toggles'],
      },
      {
        file: 'src/components/insight/studio/DiagnosticsPanel.vue',
        required: ['diagnostics-panel', 'diagnostics-panel__summary-value', 'diagnostics-panel__issue-title', 'diagnostics-panel__issue-list', 'diagnostics-panel__checks-title'],
        legacy: [],
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

    const forbiddenSelectorsByFile = new Map([
      ['src/components/insight/studio/GreetingWorkbench.vue', [
        '.greeting-workbench__section-head h3',
        '.greeting-workbench__section-head p',
      ]],
      ['src/components/insight/studio/RegexWorkbench.vue', [
        '.regex-workbench__head h3',
        '.regex-workbench__head p',
      ]],
      ['src/components/insight/studio/TaskWorkbench.vue', [
        '.task-workbench__head h3',
        '.task-workbench__head p',
      ]],
      ['src/components/insight/studio/DiagnosticsPanel.vue', [
        '.diagnostics-panel__summary-card strong',
        '.diagnostics-panel__issue-card h4',
        '.diagnostics-panel__checks-card h4',
        '.diagnostics-panel__issue-card ul',
      ]],
    ])

    for (const [file, forbiddenSelectors] of forbiddenSelectorsByFile) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      for (const selector of forbiddenSelectors) {
        expect(source, file).not.toContain(selector)
      }
    }
  })

  it('uses product action rows for editor head actions', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioEditor.vue'),
      'utf8',
    )

    expect(source).toContain('ProductActionRow')
    expect(source).not.toContain('variant="toolbar"')
    expect(source).not.toContain('class="action-ghost"')
    expect(source).not.toContain('class="action-primary"')
    expect(source).not.toMatch(/\.action-(?:ghost|primary|danger)\b/)
  })

  it('uses the shared segmented-tab primitive without Studio-only wrapper tabs', () => {
    const editorSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioEditor.vue'),
      'utf8',
    )
    const previewSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioPreview.vue'),
      'utf8',
    )

    expect(editorSource).toContain('ProductSegmentedTabs')
    expect(editorSource).not.toContain('<StudioSectionTabs')
    expect(editorSource).not.toContain("from './StudioSectionTabs.vue'")
    expect(previewSource).toContain('ProductSegmentedTabs')
    expect(previewSource).not.toContain('<PreviewTabs')
    expect(previewSource).not.toContain("from './preview/PreviewTabs.vue'")
    expect(existsSync(resolve(process.cwd(), 'src/components/insight/studio/StudioSectionTabs.vue'))).toBe(false)
    expect(existsSync(resolve(process.cwd(), 'src/components/insight/studio/preview/PreviewTabs.vue'))).toBe(false)
  })

  it('uses a Studio editor section panel shell instead of repeated local workspace cards', () => {
    const editorSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioEditor.vue'),
      'utf8',
    )
    const overviewSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/editor/StudioOverviewTab.vue'),
      'utf8',
    )

    expect(existsSync(resolve(process.cwd(), 'src/components/insight/studio/editor/StudioEditorSectionPanel.vue'))).toBe(true)
    expect(editorSource).toContain('StudioEditorSectionPanel')
    expect(overviewSource).toContain('StudioEditorSectionPanel')
    for (const source of [editorSource, overviewSource]) {
      expect(source).not.toContain('class="workspace-card"')
      expect(source).not.toContain('class="card-head"')
      expect(source).not.toMatch(/\.workspace-card\b/)
      expect(source).not.toMatch(/\.card-head\b/)
    }
  })

  it('renders the no-document onboarding state through product feedback and record cards', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioEditor.vue'),
      'utf8',
    )
    expect(source).toContain('ProductEmptyState')
    expect(source).not.toContain('empty-card')
    expect(source).not.toContain('empty-tip')
    expect(source).not.toContain('empty-mark')

    const wrapper = mount(CharacterStudioEditor, {
      props: {
        document: null,
        avatarUrl: '',
        diagnostics: null,
        activeTab: 'overview',
        activeScriptTab: 'regex',
        pendingState: {
          generatingSection: null,
          validating: false,
          importingWorldbook: false,
          deleting: false,
          saving: false,
          downloadingFormat: null,
        },
      },
    })

    expect(wrapper.getComponent(ProductEmptyState).props()).toMatchObject({
      eyebrow: '角色工坊',
      iconName: 'users',
      role: 'note',
      title: '选择或创建角色文档',
    })
    expect(wrapper.text()).toContain('先从左侧候选锁定角色名')
    expect(wrapper.findAllComponents(ProductRecordCard).map(card => card.text())).toEqual([
      expect.stringContaining('从候选开始'),
      expect.stringContaining('空白新建或导入'),
    ])
  })

  it('does not reskin editor head buttons through local UiButton variables', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioEditor.vue'),
      'utf8',
    )

    expect(source).not.toContain('class="head-actions"')
    expect(source).not.toMatch(/\.head-actions\s*\{[\s\S]*--ui-button-/)
  })

  it('uses product action contracts in editor workbench children', () => {
    const childFiles = [
      'src/components/insight/studio/GreetingWorkbench.vue',
      'src/components/insight/studio/RegexWorkbench.vue',
      'src/components/insight/studio/TaskWorkbench.vue',
      'src/components/insight/studio/LorebookTreeEditor.vue',
      'src/components/insight/studio/LorebookTreeBranch.vue',
      'src/components/insight/studio/editor/StudioHeroSection.vue',
      'src/components/insight/studio/editor/StudioOverviewTab.vue',
    ]

    for (const file of childFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, file).toContain('ProductActionRow')
      expect(source, file).not.toContain('variant="toolbar"')
      expect(source, file).not.toMatch(/class="[^"]*action-(?:ghost|primary|secondary|danger)/)
      expect(source, file).not.toMatch(/\.action-(?:ghost|primary|secondary|danger)\b/)
      expect(source, file).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    }
  })

  it('uses the product avatar contract in the Studio hero', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/editor/StudioHeroSection.vue'),
      'utf8',
    )

    expect(source).toContain("import ProductAvatar from '@/components/product/ProductAvatar.vue'")
    expect(source).toContain('<ProductAvatar')
    expect(source).toContain('shape="portrait"')
    expect(source).not.toContain('class="avatar-shell"')
    expect(source).not.toContain('avatar-placeholder')
    expect(source).not.toContain('<img v-if="avatarUrl"')
  })

  it('keeps Studio hero hooks under the component owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/editor/StudioHeroSection.vue'),
      'utf8',
    )
    const classTokens = [...source.matchAll(/class="([^"]+)"/g)]
      .flatMap(match => match[1]!.split(/\s+/).filter(Boolean))

    expect(classTokens).toContain('studio-hero-section')
    expect(classTokens).toContain('studio-hero-section__main')
    expect(classTokens).toContain('studio-hero-section__avatar')
    expect(classTokens).toContain('studio-hero-section__title')
    expect(classTokens).toContain('studio-hero-section__meta-pill')
    expect(classTokens).toContain('studio-hero-section__actions')

    for (const legacyClass of [
      'overview-hero',
      'hero-main',
      'hero-avatar',
      'hero-copy',
      'hero-kicker',
      'hero-meta',
      'meta-pill',
      'hero-actions',
    ]) {
      expect(classTokens).not.toContain(legacyClass)
    }
  })

  it('uses product record cards for repeated workbench items', () => {
    for (const file of [
      'src/components/insight/studio/GreetingWorkbench.vue',
      'src/components/insight/studio/RegexWorkbench.vue',
      'src/components/insight/studio/TaskWorkbench.vue',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).toContain('ProductRecordCard')
      expect(source, file).not.toMatch(/<article\b/)
    }
  })

  it('uses the shared number primitive for lorebook numeric fields', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/LorebookTreeBranch.vue'),
      'utf8',
    )

    expect(source).toContain('UiNumberField')
    expect(source).not.toMatch(/<UiInput\b[^\n]*\btype="number"|<UiInput\b[^\n]*\bv-model\.number/)
  })

  it('uses the shared number primitive for state task intervals', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/TaskWorkbench.vue'),
      'utf8',
    )

    expect(source).toContain('UiNumberField')
    expect(source).not.toMatch(/<UiInput\b(?=[^>]*\btype="number")/)
  })

  it('uses shared form primitives in script and lorebook workbench fields', () => {
    for (const file of [
      'src/components/insight/studio/RegexWorkbench.vue',
      'src/components/insight/studio/TaskWorkbench.vue',
      'src/components/insight/studio/LorebookTreeBranch.vue',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).toContain('UiField')
      expect(source, file).toContain('UiFormGrid')
      expect(source, file).not.toMatch(/<label\b/)
      expect(source, file).not.toMatch(/class="grid"|\.grid\b/)
    }
  })

  it('uses the studio form-control variant instead of business-root primitive skins', () => {
    for (const file of [
      'src/components/insight/studio/CharacterStudioEditor.vue',
      'src/components/insight/studio/GreetingWorkbench.vue',
      'src/components/insight/studio/RegexWorkbench.vue',
      'src/components/insight/studio/TaskWorkbench.vue',
      'src/components/insight/studio/LorebookTreeBranch.vue',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).toContain('variant="studio"')
      expect(source, file).not.toMatch(/--ui-(?:input|select|textarea)-/)
    }
  })

  it('renders editor workbench empty states through compact product empty states', () => {
    const workbenchFiles = [
      'src/components/insight/studio/GreetingWorkbench.vue',
      'src/components/insight/studio/RegexWorkbench.vue',
      'src/components/insight/studio/TaskWorkbench.vue',
    ]

    for (const file of workbenchFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, file).toContain('ProductEmptyState')
      expect(source, file).not.toContain('empty-copy')
    }

    const greeting = mount(GreetingWorkbench, {
      props: {
        firstMessage: '',
        alternates: [],
        generating: false,
      },
    })
    expect(greeting.getComponent(ProductEmptyState).props()).toMatchObject({
      iconName: 'message',
      role: 'note',
      size: 'compact',
      title: '还没有备用问候',
    })

    const regex = mount(RegexWorkbench, {
      props: {
        scripts: [],
        generating: false,
      },
    })
    expect(regex.getComponent(ProductEmptyState).props()).toMatchObject({
      iconName: 'case-sensitive',
      role: 'note',
      size: 'compact',
      title: '还没有正则脚本',
    })

    const tasks = mount(TaskWorkbench, {
      props: {
        tasks: [],
        generating: false,
      },
    })
    expect(tasks.getComponent(ProductEmptyState).props()).toMatchObject({
      iconName: 'list',
      role: 'note',
      size: 'compact',
      title: '还没有状态任务',
    })
  })

  it('uses the typed file-input primitive boundary for lorebook imports', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/LorebookTreeEditor.vue'),
      'utf8',
    )

    expect(source).toContain('@files-change="handleWorldbookSelect"')
    expect(source).not.toContain('ref<HTMLInputElement')
    expect(source).not.toContain('event.target as HTMLInputElement')
    expect(source).not.toContain('target.files')
    expect(source).not.toContain("target.value = ''")
  })

  it('does not keep orphaned local form/button selectors after primitive migration', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioEditor.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/^label\s*\{/m)
    expect(source).not.toMatch(/^\.small\s*\{/m)
  })

  it('uses typed model updates for Studio text fields instead of DOM input casts', () => {
    const files = [
      'src/components/insight/studio/CharacterStudioEditor.vue',
      'src/components/insight/studio/GreetingWorkbench.vue',
      'src/components/insight/studio/RegexWorkbench.vue',
      'src/components/insight/studio/TaskWorkbench.vue',
      'src/components/insight/studio/LorebookTreeBranch.vue',
    ]

    for (const file of files) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).not.toContain('@input=')
      expect(source, file).not.toMatch(/\$event\.target as HTML(?:Input|TextArea)Element/)
    }
  })

  it('keeps editor document and lorebook tree sync on the shared clone helper', () => {
    for (const file of [
      'src/components/insight/studio/CharacterStudioEditor.vue',
      'src/components/insight/studio/LorebookTreeEditor.vue',
      'src/components/insight/studio/LorebookTreeBranch.vue',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).toContain("import { deepClone } from '@/utils/deepClone'")
      expect(source, file).not.toContain('JSON.parse(JSON.stringify')
    }
  })

  function getTab(wrapper: VueWrapper, label: string) {
    const tab = wrapper.findAll('[role="tab"]').find(item => item.text().includes(label))
    expect(tab).toBeTruthy()
    return tab!
  }

  it('shows chinese section tabs and defaults to 概览', () => {
    const wrapper = mountHarness()

    expect(wrapper.text()).toContain('概览')
    expect(wrapper.text()).toContain('角色设定')
    expect(wrapper.text()).toContain('问候语')
    expect(wrapper.text()).toContain('脚本任务')
    expect(wrapper.text()).toContain('来源摘要')
  })

  it('preserves edited data when switching tabs', async () => {
    const wrapper = mountHarness()

    await getTab(wrapper, '角色设定').trigger('click')
    const description = wrapper.find('textarea')
    await description.setValue('新的角色简介')

    await getTab(wrapper, '问候语').trigger('click')
    await getTab(wrapper, '角色设定').trigger('click')

    const currentValue = wrapper.find('textarea').element as HTMLTextAreaElement
    expect(currentValue.value).toBe('新的角色简介')
  })

  it('renders character identity fields through shared form primitives', async () => {
    const wrapper = mountHarness()

    await getTab(wrapper, '角色设定').trigger('click')

    expect(wrapper.findComponent(UiFormGrid).exists()).toBe(true)
    const fields = wrapper.findAllComponents(UiField)
    expect(fields.map(field => field.props('label'))).toEqual([
      '角色名称',
      '别名（逗号分隔）',
      '角色简介',
      '性格 / 人设',
      '当前场景',
      '标签（逗号分隔）',
    ])
    expect(wrapper.get('label[for="studioCharacterName"]').exists()).toBe(true)
    expect(wrapper.get('#studioCharacterName').exists()).toBe(true)
    expect(wrapper.get('label[for="studioCharacterDescription"]').exists()).toBe(true)
    expect(wrapper.get('#studioCharacterDescription').exists()).toBe(true)

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioEditor.vue'),
      'utf8',
    )
    expect(source).toContain('UiField')
    expect(source).toContain('UiFormGrid')
  })

  it('renders dialogue metadata fields through shared form primitives', async () => {
    const wrapper = mountHarness()

    await getTab(wrapper, '问候语').trigger('click')

    const metadataLabels = wrapper
      .findAllComponents(UiField)
      .map(field => field.props('label'))
      .filter(Boolean)

    expect(metadataLabels).toEqual([
      '示例对话',
      '系统提示词（System Prompt）',
      '历史后置说明（Post History）',
      '备注',
      '角色版本',
    ])
    expect(wrapper.get('label[for="studioMessageExample"]').exists()).toBe(true)
    expect(wrapper.get('#studioMessageExample').exists()).toBe(true)
    expect(wrapper.get('label[for="studioCharacterVersion"]').exists()).toBe(true)
    expect(wrapper.get('#studioCharacterVersion').exists()).toBe(true)
  })

  it('renders export download cards through the product record-card contract', async () => {
    const wrapper = mountHarness()

    await getTab(wrapper, '导出诊断').trigger('click')

    const downloadCards = wrapper
      .findAllComponents(ProductRecordCard)
      .filter(card => card.props('as') === 'button')

    expect(downloadCards.map(card => card.attributes('aria-label'))).toEqual([
      '导出 V3 JSON',
      '导出 V2 JSON',
      '导出 PNG',
      '导出世界书',
    ])
    expect(downloadCards[0]?.text()).toContain('当前工作台的主导出格式。')

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioEditor.vue'),
      'utf8',
    )
    expect(source).toContain('ProductRecordCard')
    expect(source).not.toContain('class="export-card"')
    expect(source).not.toMatch(/\.export-card\b/)
  })

  it('keeps meta title in sync when identity name changes', async () => {
    const wrapper = mountHarness()

    await getTab(wrapper, '角色设定').trigger('click')
    const nameInput = wrapper.find('input[type="text"]')
    await nameInput.setValue('新角色名')

    expect(wrapper.find('.studio-hero-section__title').text()).toBe('新角色名')
  })

  it('shows persisted review summary when latest review exists', () => {
    document.exportArtifacts = {
      last_review: {
        summary: '建议补强世界书和备用问候。',
        issues: ['世界书覆盖面不足'],
        suggestions: ['增加 2-3 条候选问候语'],
      },
    }

    const wrapper = mountHarness()

    expect(wrapper.text()).toContain('建议补强世界书和备用问候。')
    expect(wrapper.text()).toContain('世界书覆盖面不足')
  })

  it('shows loading copy for section generation and validation actions', async () => {
    const generationWrapper = mount(defineComponent({
      components: { CharacterStudioEditor },
      setup() {
        return () => h(CharacterStudioEditor, {
          document,
          avatarUrl: '',
          diagnostics: null,
          activeTab: 'character',
          activeScriptTab: 'regex',
          pendingState: {
            generatingSection: 'identity',
            validating: true,
            importingWorldbook: false,
            deleting: false,
            saving: false,
            downloadingFormat: null,
          },
        })
      },
    }), {
      global: {
        stubs: {
          LorebookTreeEditor: {
            template: '<div class="lorebook-stub">世界书树编辑器</div>',
          },
        },
      },
    })

    expect(generationWrapper.text()).toContain('重写中...')

    const validationWrapper = mount(defineComponent({
      components: { CharacterStudioEditor },
      setup() {
        return () => h(CharacterStudioEditor, {
          document,
          avatarUrl: '',
          diagnostics: null,
          activeTab: 'overview',
          activeScriptTab: 'regex',
          pendingState: {
            generatingSection: null,
            validating: true,
            importingWorldbook: false,
            deleting: false,
            saving: false,
            downloadingFormat: null,
          },
        })
      },
    }), {
      global: {
        stubs: {
          LorebookTreeEditor: {
            template: '<div class="lorebook-stub">世界书树编辑器</div>',
          },
        },
      },
    })

    expect(validationWrapper.text()).toContain('诊断中...')
  })

  it('shows full card generation entry and loading copy', () => {
    const idleWrapper = mountHarness()
    expect(idleWrapper.text()).toContain('AI 一键补全整卡')

    const wrapper = mount(defineComponent({
      components: { CharacterStudioEditor },
      setup() {
        return () => h(CharacterStudioEditor, {
          document,
          avatarUrl: '',
          diagnostics: null,
          activeTab: 'overview',
          activeScriptTab: 'regex',
          pendingState: {
            generatingSection: 'full',
            validating: false,
            importingWorldbook: false,
            deleting: false,
            saving: false,
            downloadingFormat: null,
          },
        })
      },
    }), {
      global: {
        stubs: {
          LorebookTreeEditor: {
            template: '<div class="lorebook-stub">世界书树编辑器</div>',
          },
        },
      },
    })

    expect(wrapper.text()).toContain('整卡补全中...')
  })

  it('renders freeze settings as aligned rows with separate label and control cells', () => {
    const wrapper = mountHarness()

    const freezeItems = wrapper.findAll('.studio-overview-tab__freeze-item')
    expect(freezeItems.length).toBeGreaterThan(0)
    expect(freezeItems[0]?.element.tagName.toLowerCase()).toBe('div')
    expect(freezeItems[0]?.find('.studio-overview-tab__freeze-item-label').exists()).toBe(true)
    expect(freezeItems[0]?.find('.studio-overview-tab__freeze-item-control').exists()).toBe(true)

    const checkbox = freezeItems[0]!.find('input[type="checkbox"]')
    expect(checkbox.attributes('id')).toMatch(/^studio-freeze-/)
    expect(checkbox.attributes('aria-label')).toContain('钉住')

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/editor/StudioOverviewTab.vue'),
      'utf8',
    )
    expect(source).not.toMatch(/<label\b/)
  })
})
