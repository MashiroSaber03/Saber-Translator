import { mount } from '@vue/test-utils'
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import ProductTabbedWorkspace from '@/components/product/ProductTabbedWorkspace.vue'
import ProductThreePaneWorkspace from '@/components/product/ProductThreePaneWorkspace.vue'
import ProductSplitWorkspace from '@/components/product/ProductSplitWorkspace.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import ProductWorkspacePanel from '@/components/product/ProductWorkspacePanel.vue'
import ProductCollapsibleSection from '@/components/product/ProductCollapsibleSection.vue'
import ProductWizardSteps from '@/components/product/ProductWizardSteps.vue'

describe('ProductWorkspacePanel', () => {
  it('shares class-binding prop types across product workspace shells', () => {
    const typePath = resolve(process.cwd(), 'src/components/product/productClassTypes.ts')
    const hasTypeOwner = existsSync(typePath)
    const typeSource = hasTypeOwner ? readFileSync(typePath, 'utf8') : ''
    const panelSource = readFileSync(
      resolve(process.cwd(), 'src/components/product/ProductWorkspacePanel.vue'),
      'utf8',
    )
    const tabbedSource = readFileSync(
      resolve(process.cwd(), 'src/components/product/ProductTabbedWorkspace.vue'),
      'utf8',
    )

    expect(hasTypeOwner).toBe(true)
    expect(typeSource).toContain('export type ProductClassValue')

    for (const source of [panelSource, tabbedSource]) {
      expect(source).toContain("from '@/components/product/productClassTypes'")
      expect(source).not.toMatch(/type ClassValue\s*=/)
    }
  })

  it('owns the reusable workspace scroll contract for tab and wizard panels', () => {
    const wrapper = mount(ProductWorkspacePanel, {
      props: {
        variant: 'wizard',
        ariaLabel: '续写工作区',
        contentClass: 'continuation-content',
      },
      slots: {
        header: '<div class="test-header">Header</div>',
        default: '<div class="test-body">Body</div>',
        footer: '<div class="test-footer">Footer</div>',
      },
    })

    expect(wrapper.classes()).toEqual(expect.arrayContaining([
      'product-workspace-panel',
      'product-workspace-panel--wizard',
    ]))
    expect(wrapper.attributes('aria-label')).toBe('续写工作区')
    expect(wrapper.find('.product-workspace-panel__header .test-header').exists()).toBe(true)
    expect(wrapper.find('.product-workspace-panel__scroll').classes()).toContain('continuation-content')
    expect(wrapper.find('.product-workspace-panel__scroll .test-body').exists()).toBe(true)
    expect(wrapper.find('.product-workspace-panel__footer .test-footer').exists()).toBe(true)
  })

  it('keeps shared workspace scroll owners shrinkable inside narrow shells', () => {
    const panelSource = readFileSync(
      resolve(process.cwd(), 'src/components/product/ProductWorkspacePanel.vue'),
      'utf8',
    )
    const tabbedSource = readFileSync(
      resolve(process.cwd(), 'src/components/product/ProductTabbedWorkspace.vue'),
      'utf8',
    )
    const panelScrollBlock = panelSource.match(/\.product-workspace-panel__scroll\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''
    const tabPanelsBlock = tabbedSource.match(/\.product-tabbed-workspace__panels\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''

    expect(panelScrollBlock).toContain('min-width: 0')
    expect(tabPanelsBlock).toContain('min-width: 0')
  })

  it('renders shared workspace tabs with stable selection semantics', async () => {
    const wrapper = mount(ProductTabbedWorkspace, {
      props: {
        tabs: [
          { id: 'overview', label: '概览', iconName: 'bar-chart' },
          { id: 'qa', label: '智能问答', iconName: 'message' },
        ],
        activeTab: 'overview',
        ariaLabel: '漫画分析工作区',
        panelsClass: 'insight-panels',
      },
      slots: {
        beforeTabs: '<button class="open-sidebar">导航</button>',
        default: '<div class="test-panel">Panel</div>',
        afterTabs: '<button class="open-notes">笔记</button>',
      },
    })

    expect(wrapper.classes()).toContain('product-tabbed-workspace')
    expect(wrapper.attributes('aria-label')).toBe('漫画分析工作区')
    const tablist = wrapper.get('[role="tablist"]')
    expect(tablist.attributes('aria-label')).toBe('漫画分析工作区标签')
    expect(wrapper.find('svg.ui-icon').exists()).toBe(true)
    expect(wrapper.find('.open-sidebar').exists()).toBe(true)
    expect(wrapper.find('.open-notes').exists()).toBe(true)
    expect(wrapper.find('.product-tabbed-workspace__panels').classes()).toContain('insight-panels')

    const tabs = wrapper.findAll('[role="tab"]')
    expect(tabs).toHaveLength(2)
    expect(tabs[0].attributes('aria-selected')).toBe('true')
    expect(tabs[1].attributes('aria-selected')).toBe('false')

    await tabs[1].trigger('click')
    expect(wrapper.emitted('update:activeTab')?.[0]).toEqual(['qa'])
    expect(wrapper.emitted('select')?.[0]).toEqual(['qa'])
  })

  it('keeps shared workspace tabs horizontally scrollable instead of squeezing labels', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/product/ProductTabbedWorkspace.vue'),
      'utf8',
    )
    const tabsBlock = source.match(/\.product-tabbed-workspace__tabs\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''
    const tabBlock = source.match(/\.product-tabbed-workspace__tab\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''
    const labelBlock = source.match(/\.product-tabbed-workspace__tab-label\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''

    expect(tabsBlock).toContain('overflow-x: auto')
    expect(tabsBlock).toContain('scrollbar-gutter: stable')
    expect(tabBlock).toContain('flex: 0 0 auto')
    expect(labelBlock).toContain('white-space: nowrap')
    expect(source).not.toContain('writing-mode')
  })

  it('keeps shared tabbed workspaces independent from business domain tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/product/ProductTabbedWorkspace.vue'),
      'utf8',
    )

    expect(source).not.toContain('--insight-')
  })

  it('keeps product workspace tests on current typed contracts', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/productWorkspace.spec.ts'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('keeps shared workspace tabs keyboard navigable', async () => {
    const wrapper = mount(ProductTabbedWorkspace, {
      props: {
        tabs: [
          { id: 'overview', label: '概览' },
          { id: 'qa', label: '智能问答' },
          { id: 'timeline', label: '时间线', disabled: true },
          { id: 'continuation', label: '续写' },
        ],
        activeTab: 'overview',
        ariaLabel: '漫画分析工作区',
      },
    })

    const tabs = wrapper.findAll('[role="tab"]')
    expect(tabs.map(tab => tab.attributes('tabindex'))).toEqual(['0', '-1', '-1', '-1'])

    await tabs[0].trigger('keydown', { key: 'ArrowRight' })
    await tabs[0].trigger('keydown', { key: 'End' })

    expect(wrapper.emitted('update:activeTab')).toEqual([['qa'], ['continuation']])
    expect(wrapper.emitted('select')).toEqual([['qa'], ['continuation']])
  })

  it('renders tab icons only through the current iconName contract', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/product/ProductTabbedWorkspace.vue'),
      'utf8',
    )

    expect(source).toContain('tab.iconName')
    expect(source).not.toMatch(/\btab\.icon\b/)

    const wrapper = mount(ProductTabbedWorkspace, {
      props: {
        tabs: [
          { id: 'plain', label: '当前入口' },
        ],
        activeTab: 'plain',
      },
    })

    expect(wrapper.text()).toContain('当前入口')
    expect(wrapper.find('.product-tabbed-workspace__tab-icon').exists()).toBe(false)
  })

  it('renders compact product segmented tabs for modal and settings owners', async () => {
    const wrapper = mount(ProductSegmentedTabs, {
      props: {
        tabs: [
          { id: 'basic', label: '基本设置' },
          { id: 'advanced', label: '高级设置', iconName: 'settings' },
        ],
        activeTab: 'basic',
        ariaLabel: '网页导入设置分类',
        layout: 'scroll',
      },
    })

    expect(wrapper.classes()).toContain('product-segmented-tabs')
    expect(wrapper.classes()).toContain('product-segmented-tabs--scroll')
    expect(wrapper.attributes('role')).toBe('tablist')
    expect(wrapper.attributes('aria-label')).toBe('网页导入设置分类')

    const tabs = wrapper.findAll('[role="tab"]')
    expect(tabs).toHaveLength(2)
    expect(tabs[0].attributes('aria-selected')).toBe('true')
    expect(tabs[1].attributes('aria-selected')).toBe('false')
    expect(wrapper.find('svg.ui-icon').exists()).toBe(true)

    await tabs[1].trigger('click')
    expect(wrapper.emitted('update:activeTab')?.[0]).toEqual(['advanced'])
    expect(wrapper.emitted('select')?.[0]).toEqual(['advanced'])
  })

  it('keeps segmented-tab owner tokens externally overridable', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/product/ProductSegmentedTabs.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/^\s*--product-segmented-tabs-[\w-]+:/m)
    expect(source).toContain('var(--product-segmented-tabs-background, var(--color-surface-muted))')
    expect(source).toContain('var(--product-segmented-tabs-active-background, var(--color-surface-base))')
    expect(source).toMatch(/\.product-segmented-tabs__tab\s*\{[\s\S]*?display: inline-flex;[\s\S]*?align-items: center;/)
  })

  it('renders underline product tabs for modal navigation owners without local CSS overrides', async () => {
    const wrapper = mount(ProductSegmentedTabs, {
      props: {
        tabs: [
          { id: 'ocr', label: 'OCR识别' },
          { id: 'translate', label: '翻译服务' },
        ],
        activeTab: 'translate',
        ariaLabel: '设置分类',
        appearance: 'underline',
      },
    })

    expect(wrapper.classes()).toContain('product-segmented-tabs--appearance-underline')
    expect(wrapper.find('.product-segmented-tabs__tab--active').text()).toBe('翻译服务')

    await wrapper.findAll('[role="tab"]')[0].trigger('click')
    expect(wrapper.emitted('update:activeTab')?.[0]).toEqual(['ocr'])
  })

  it('keeps segmented tabs keyboard navigable without global generated ids', async () => {
    const wrapper = mount(ProductSegmentedTabs, {
      props: {
        tabs: [
          { id: 'basic', label: '基本设置' },
          { id: 'preprocess', label: '图片预处理', disabled: true },
          { id: 'advanced', label: '高级设置' },
        ],
        activeTab: 'basic',
        ariaLabel: '网页导入设置分类',
      },
    })

    const tabs = wrapper.findAll('[role="tab"]')
    expect(tabs[0].attributes('tabindex')).toBe('0')
    expect(tabs[1].attributes('tabindex')).toBe('-1')
    expect(tabs[2].attributes('tabindex')).toBe('-1')
    expect(tabs[0].attributes('id')).toBeUndefined()

    await tabs[0].trigger('keydown', { key: 'ArrowRight' })
    expect(wrapper.emitted('update:activeTab')?.[0]).toEqual(['advanced'])
    expect(wrapper.emitted('select')?.[0]).toEqual(['advanced'])

    await tabs[0].trigger('keydown', { key: 'End' })
    expect(wrapper.emitted('update:activeTab')?.[1]).toEqual(['advanced'])
  })

  it('owns product collapsible section disclosure semantics', async () => {
    const wrapper = mount(ProductCollapsibleSection, {
      props: {
        expanded: false,
        title: '设置',
        hint: '点击展开配置',
        iconName: 'settings',
        ariaLabel: '网页导入设置',
      },
      slots: {
        default: '<div class="settings-body">Body</div>',
      },
    })

    expect(wrapper.classes()).toContain('product-collapsible-section')
    expect(wrapper.get('button').attributes('aria-expanded')).toBe('false')
    expect(wrapper.get('button').attributes('aria-label')).toBe('网页导入设置')
    expect(wrapper.get('button').attributes('aria-controls')).toBeTruthy()
    expect(wrapper.text()).toContain('设置')
    expect(wrapper.text()).toContain('点击展开配置')
    expect(wrapper.find('.settings-body').exists()).toBe(false)

    await wrapper.get('button').trigger('click')

    expect(wrapper.emitted('update:expanded')?.[0]).toEqual([true])
    expect(wrapper.emitted('toggle')?.[0]).toEqual([true])

    await wrapper.setProps({ expanded: true })
    expect(wrapper.get('.product-collapsible-section__body').attributes('id'))
      .toBe(wrapper.get('button').attributes('aria-controls'))
  })

  it('owns wizard step navigation as a reusable product primitive', async () => {
    const wrapper = mount(ProductWizardSteps, {
      props: {
        steps: [
          { label: '角色设置' },
          { label: '生成脚本' },
          { label: '页面剧情', disabled: true },
        ],
        activeIndex: 1,
        ariaLabel: '续写步骤',
      },
    })

    expect(wrapper.classes()).toContain('product-wizard-steps')
    expect(wrapper.attributes('aria-label')).toBe('续写步骤')

    const steps = wrapper.findAll('.product-wizard-steps__step')
    expect(steps).toHaveLength(3)
    expect(steps[0].classes()).toContain('product-wizard-steps__step--completed')
    expect(steps[1].classes()).toContain('product-wizard-steps__step--active')
    expect(steps[1].attributes('aria-current')).toBe('step')
    expect(steps[2].attributes('disabled')).toBeDefined()
    expect(wrapper.findAll('.product-wizard-steps__number').map(step => step.text())).toEqual(['1', '2', '3'])

    await steps[0].trigger('click')

    expect(wrapper.emitted('update:activeIndex')?.[0]).toEqual([0])
    expect(wrapper.emitted('select')?.[0]).toEqual([0])
  })

  it('keeps continuation wizard steps on the product primitive instead of parent-owned step skins', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/ContinuationPanel.vue'),
      'utf8',
    )

    expect(source).toContain('<ProductWizardSteps')
    expect(source).not.toContain('class="step-indicator"')
    expect(source).not.toContain('class="step"')
    expect(source).not.toContain('.step {')
  })

  it('owns three-pane sizing and independent pane scroll classes', () => {
    const wrapper = mount(ProductThreePaneWorkspace, {
      props: {
        as: 'main',
        ariaLabel: '漫画分析三栏工作区',
        leftWidth: '280px',
        rightWidth: '320px',
        leftMobileVisible: true,
      },
      slots: {
        left: '<div class="left-content">Left</div>',
        default: '<div class="main-content">Main</div>',
        right: '<div class="right-content">Right</div>',
      },
    })

    expect(wrapper.element.tagName).toBe('MAIN')
    expect(wrapper.classes()).toContain('product-three-pane-workspace')
    expect(wrapper.attributes('aria-label')).toBe('漫画分析三栏工作区')
    expect(wrapper.attributes('style')).toContain('--product-three-pane-left-width: 280px;')
    expect(wrapper.attributes('style')).toContain('--product-three-pane-right-width: 320px;')
    expect(wrapper.find('.product-three-pane-workspace__pane--left').classes())
      .toContain('product-three-pane-workspace__pane--mobile-visible')
    expect(wrapper.find('.product-three-pane-workspace__pane--left .left-content').exists()).toBe(true)
    expect(wrapper.find('.product-three-pane-workspace__main .main-content').exists()).toBe(true)
    expect(wrapper.find('.product-three-pane-workspace__pane--right .right-content').exists()).toBe(true)
  })

  it('switches drawer panes before fixed sidebars squeeze the main workspace', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/product/ProductThreePaneWorkspace.vue'),
      'utf8',
    )

    expect(source).toContain('@media (--breakpoint-lg-down)')
    expect(source).toContain('.product-three-pane-workspace--mobile-drawer .product-three-pane-workspace__pane')
    expect(source).toContain('--product-three-pane-drawer-z-index: var(--z-dropdown)')
    expect(source).not.toContain('@media (--breakpoint-md-down)')
    expect(source).not.toContain('--z-index-dropdown')
  })

  it('owns split-pane sizing, scroll containers, and keyboard resizing', async () => {
    const wrapper = mount(ProductSplitWorkspace, {
      props: {
        leftPaneWidth: 52,
        min: 35,
        max: 70,
        step: 2,
        ariaLabel: '角色工坊工作区',
        resizerLabel: '调整编辑区和预览区宽度',
        leftScrollTestId: 'editor-scroll',
        rightScrollTestId: 'chat-scroll',
      },
      slots: {
        left: '<div class="editor-content">Editor</div>',
        right: '<div class="preview-content">Preview</div>',
      },
    })

    expect(wrapper.classes()).toContain('product-split-workspace')
    expect(wrapper.attributes('aria-label')).toBe('角色工坊工作区')
    expect(wrapper.attributes('style')).toContain('--product-split-left-track: 52fr;')
    expect(wrapper.attributes('style')).toContain('--product-split-resizer-width: 8px;')
    expect(wrapper.attributes('style')).toContain('--product-split-right-track: 48fr;')
    expect(wrapper.find('[data-testid="editor-scroll"] .editor-content').exists()).toBe(true)
    expect(wrapper.find('[data-testid="chat-scroll"] .preview-content').exists()).toBe(true)

    const resizer = wrapper.get('[role="separator"]')
    expect(resizer.attributes('aria-label')).toBe('调整编辑区和预览区宽度')
    expect(resizer.attributes('aria-orientation')).toBe('vertical')
    expect(resizer.attributes('aria-valuemin')).toBe('35')
    expect(resizer.attributes('aria-valuemax')).toBe('70')
    expect(resizer.attributes('aria-valuenow')).toBe('52')

    await resizer.trigger('keydown', { key: 'ArrowRight' })

    expect(wrapper.emitted('update:leftPaneWidth')?.[0]).toEqual([54])
    expect(wrapper.emitted('resize')?.[0]).toEqual([54])
  })
})
