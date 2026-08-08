import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import ProductHeaderAction from '@/components/product/ProductHeaderAction.vue'
import ProductHeaderMetaPill from '@/components/product/ProductHeaderMetaPill.vue'
import ProductPageHeader from '@/components/product/ProductPageHeader.vue'

describe('ProductPageHeader', () => {
  it('renders the shared page header regions and brand link contract', () => {
    const wrapper = mount(ProductPageHeader, {
      props: {
        variant: 'brand',
        logoTitle: '书架首页',
        homeTo: '/',
        navLabel: '书架链接',
        actionsLabel: '书架操作',
      },
      global: {
        stubs: {
          RouterLink: {
            props: ['to'],
            template: '<a class="router-link-stub" :href="to"><slot /></a>',
          },
        },
      },
      slots: {
        meta: '<span class="test-meta">LAN</span>',
        nav: '<a class="test-nav">Docs</a>',
        actions: '<button class="test-action">Theme</button>',
      },
    })

    expect(wrapper.classes()).toEqual(expect.arrayContaining([
      'product-page-header',
      'product-page-header--brand',
    ]))
    expect(wrapper.get('.product-page-header__brand-link').attributes('title')).toBe('书架首页')
    expect(wrapper.get('.product-page-header__logo').attributes('alt')).toBe('Saber-Translator Logo')
    expect(wrapper.get('.product-page-header__name').text()).toBe('Saber-Translator')
    expect(wrapper.find('.product-page-header__meta .test-meta').exists()).toBe(true)
    expect(wrapper.get('nav[aria-label="书架链接"] .test-nav').exists()).toBe(true)
    expect(wrapper.find('.product-page-header__actions .test-action').exists()).toBe(true)
    expect(wrapper.get('[role="group"][aria-label="书架操作"] .test-action').exists()).toBe(true)
  })

  it('allows workbench headers to replace the default brand region', () => {
    const wrapper = mount(ProductPageHeader, {
      props: {
        variant: 'reader',
      },
      global: {
        stubs: {
          RouterLink: {
            props: ['to'],
            template: '<a class="router-link-stub" :href="to"><slot /></a>',
          },
        },
      },
      slots: {
        brand: '<div class="reader-brand-slot">Reader context</div>',
        meta: '<span class="reader-page-slot">1 / 12</span>',
        actions: '<button class="reader-action-slot">Settings</button>',
      },
    })

    expect(wrapper.classes()).toContain('product-page-header--reader')
    expect(wrapper.find('.product-page-header__brand-link').exists()).toBe(false)
    expect(wrapper.get('.product-page-header__brand .reader-brand-slot').text()).toBe('Reader context')
    expect(wrapper.get('.product-page-header__meta .reader-page-slot').text()).toBe('1 / 12')
    expect(wrapper.get('.product-page-header__actions .reader-action-slot').text()).toBe('Settings')
  })

  it('renders reusable header actions for buttons and links', async () => {
    const button = mount(ProductHeaderAction, {
      props: {
        iconName: 'settings',
        label: '设置',
      },
    })

    expect(button.element.tagName).toBe('BUTTON')
    expect(button.classes()).toContain('product-header-action')
    expect(button.text()).toContain('设置')
    await button.trigger('click')
    expect(button.emitted('click')).toHaveLength(1)

    const textIconButton = mount(ProductHeaderAction, {
      props: {
        iconOnly: true,
        ariaLabel: '阅读设置',
      },
      slots: { icon: '⚙️' },
    })
    expect(textIconButton.get('.product-header-action__icon-text').text()).toBe('⚙️')

    const external = mount(ProductHeaderAction, {
      props: {
        as: 'a',
        href: 'https://example.com',
        target: '_blank',
        rel: 'noopener noreferrer',
        label: '使用教程',
      },
    })

    expect(external.element.tagName).toBe('A')
    expect(external.attributes('href')).toBe('https://example.com')
    expect(external.attributes('rel')).toBe('noopener noreferrer')

    const current = mount(ProductHeaderAction, {
      props: {
        as: 'span',
        active: true,
        iconName: 'search',
        label: '分析',
      },
    })

    expect(current.element.tagName).toBe('SPAN')
    expect(current.classes()).toContain('product-header-action--active')
    expect(current.classes()).toContain('product-header-action--static')
    expect(current.attributes('aria-pressed')).toBeUndefined()
    expect(current.text()).toContain('分析')
    await current.trigger('click')
    expect(current.emitted('click')).toBeUndefined()

    const pressedMode = mount(ProductHeaderAction, {
      props: {
        active: true,
        pressed: true,
        label: '原图',
      },
    })

    expect(pressedMode.classes()).toContain('product-header-action--active')
    expect(pressedMode.attributes('aria-pressed')).toBe('true')
  })

  it('renders reusable header metadata pills with optional actions', () => {
    const wrapper = mount(ProductHeaderMetaPill, {
      props: {
        label: '局域网访问',
        value: 'http://localhost:5173',
        title: '其他设备可通过此地址访问',
      },
      slots: {
        actions: '<button class="copy-lan">复制</button>',
      },
    })

    expect(wrapper.classes()).toContain('product-header-meta-pill')
    expect(wrapper.attributes('title')).toBe('其他设备可通过此地址访问')
    expect(wrapper.get('.product-header-meta-pill__label').text()).toBe('局域网访问')
    expect(wrapper.get('.product-header-meta-pill__value').text()).toBe('http://localhost:5173')
    expect(wrapper.get('.product-header-meta-pill__actions .copy-lan').exists()).toBe(true)
  })

  it('exposes mobile label collapse as a public action contract', () => {
    const action = mount(ProductHeaderAction, {
      props: {
        iconName: 'chevron-left',
        label: '返回',
        collapseLabelOnMobile: true,
      },
    })

    expect(action.classes()).toContain('product-header-action--collapse-label-md')
    expect(action.attributes('aria-label')).toBe('返回')
    expect(action.text()).toContain('返回')
  })

  it('keeps danger header actions on the danger color through hover states', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductHeaderAction.vue'), 'utf8')

    expect(source).not.toContain('--product-header-action-danger-hover-text')
    expect(source).toMatch(/product-header-action--tone-danger:hover[\s\S]*?color:\s*var\(--product-header-action-danger-text-color,\s*var\(--color-status-error\)\);/)
  })

  it('keeps static header actions out of interactive hover rules', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductHeaderAction.vue'), 'utf8')

    expect(source).toContain('product-header-action--static')
    expect(source).toMatch(/\.product-header-action:hover:not\(\.product-header-action--disabled, \.product-header-action--static\)/)
    expect(source).toMatch(/\.product-header-action--solid:hover:not\(\.product-header-action--disabled, \.product-header-action--static\)/)
    expect(source).toMatch(/\.product-header-action--plain:hover:not\(\.product-header-action--disabled, \.product-header-action--static\)/)
  })

  it('keeps header action visual state ownership inside the action primitive', () => {
    const pageHeaderSource = readFileSync(resolve(process.cwd(), 'src/components/product/ProductPageHeader.vue'), 'utf8')
    const studioTopbarSource = readFileSync(resolve(process.cwd(), 'src/components/insight/studio/StudioTopbar.vue'), 'utf8')
    const actionSource = readFileSync(resolve(process.cwd(), 'src/components/product/ProductHeaderAction.vue'), 'utf8')
    const legacyStateToken = /--product-header-action-(background|border|text|hover-background|hover-border|hover-text|solid-background|solid-border|solid-hover-background|solid-hover-border|solid-shadow|solid-text|plain-text|danger-text|active-background|active-text)\b/

    expect(pageHeaderSource).not.toMatch(legacyStateToken)
    expect(studioTopbarSource).not.toMatch(legacyStateToken)
    expect(actionSource).toContain('--product-header-action-context-surface')
    expect(actionSource).toContain('--product-header-action-surface')
  })

  it('keeps header chrome colors on semantic tokens', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductPageHeader.vue'), 'utf8')

    expect(source).not.toMatch(/rgba?\(/)
    expect(source).not.toMatch(/#[0-9a-f]{3,8}\b/i)
  })

  it('keeps the default variant on the base product header spacing contract', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductPageHeader.vue'), 'utf8')

    expect(source).not.toContain('.product-page-header--default .product-page-header__content {\n  gap: 0;')
    expect(source).not.toContain('.product-page-header--default .product-page-header__brand-link {\n  gap: 0;')
    expect(source).not.toContain('.product-page-header--default .product-page-header__logo {\n  width: auto;')
    expect(source).not.toMatch(/product-page-header--default[\s\S]*?min-width:\s*auto/)
  })

  it('lets shared default and brand headers wrap without fixed-height page shims', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductPageHeader.vue'), 'utf8')

    expect(source).toMatch(/\.product-page-header__content\s*\{[^}]*flex-wrap:\s*wrap;/)
    expect(source).toMatch(/\.product-page-header__meta,\s*\.product-page-header__nav,\s*\.product-page-header__actions\s*\{[^}]*flex-wrap:\s*wrap;/)
    expect(source).toMatch(/\.product-page-header--brand\s*\{[^}]*min-height:\s*64px;/)
    expect(source).not.toMatch(/\.product-page-header--brand\s*\{[^}]*\n\s{2}height:\s*64px;/)
    expect(source).toMatch(/\.product-page-header--fixed \.product-page-header__content\s*\{[^}]*flex-wrap:\s*nowrap;/)
    expect(source).toMatch(/\.product-page-header--reader \.product-page-header__content\s*\{[^}]*flex-wrap:\s*nowrap;/)
  })

  it('bounds fixed header content to its visible hit area', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductPageHeader.vue'), 'utf8')
    const fixedContentBlock = source.match(/\.product-page-header--fixed \.product-page-header__content\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''

    expect(fixedContentBlock).toContain('height: 100%')
    expect(fixedContentBlock).toContain('overflow: hidden')
  })

  it('keeps fixed headers compact enough for mobile workbench navigation', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductPageHeader.vue'), 'utf8')
    const mobileBlock = source.match(/@media \(--breakpoint-md-down\)\s*\{(?<body>[\s\S]*)\n\}/)?.groups?.body ?? ''

    expect(mobileBlock).toContain('.product-page-header--fixed {')
    expect(mobileBlock).toContain('padding: 0 12px')
    expect(mobileBlock).toMatch(/\.product-page-header--fixed \.product-page-header__name\s*\{[\s\S]*display:\s*none/)
    expect(mobileBlock).toMatch(/\.product-page-header--fixed \.product-page-header__nav,\s*\.product-page-header--fixed \.product-page-header__actions\s*\{[\s\S]*gap:\s*8px/)
  })

  it('reserves header action space for the global task-center launcher', () => {
    const headerSource = readFileSync(resolve(process.cwd(), 'src/components/product/ProductPageHeader.vue'), 'utf8')
    const launcherSource = readFileSync(resolve(process.cwd(), 'src/components/task-center/TaskCenterLauncher.vue'), 'utf8')
    const studioTopbarSource = readFileSync(resolve(process.cwd(), 'src/components/insight/studio/StudioTopbar.vue'), 'utf8')

    expect(headerSource).toMatch(/\.product-page-header--brand \.product-page-header__actions,\s*\.product-page-header--fixed \.product-page-header__actions,\s*\.product-page-header--reader \.product-page-header__actions\s*\{[^}]*margin-right:\s*136px/)
    expect(headerSource).toMatch(/@media \(--breakpoint-md-down\)[\s\S]*margin-right:\s*72px/)
    expect(launcherSource).toContain('task-center-launcher__label')
    expect(launcherSource).toMatch(/@media \(--breakpoint-md-down\)[\s\S]*\.task-center-launcher__label\s*\{[^}]*display:\s*none/)
    expect(studioTopbarSource).toContain('padding: 10px 156px 10px 20px')
    expect(studioTopbarSource).toMatch(/@media \(--breakpoint-md-down\)[\s\S]*padding-right:\s*84px/)
  })
})
