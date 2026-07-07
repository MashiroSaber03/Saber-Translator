import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import ProductHeaderAction from '@/components/product/ProductHeaderAction.vue'
import StudioTopbar from '@/components/insight/studio/StudioTopbar.vue'

function cssRule(style: string, selector: string): string {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return style.match(new RegExp(`${escapedSelector}\\s*{([\\s\\S]*?)}`))?.[1] ?? ''
}

describe('StudioTopbar compact layout', () => {
  it('renders a compact title block without the large subtitle banner copy', () => {
    const wrapper = mount(StudioTopbar, {
      props: {
        bookTitle: '五等分',
        documentTitle: '上杉风太郎',
        documentOrigin: '分析生成',
        hasDocument: true,
        busy: true,
        busyLabel: '正在保存角色文档',
        savePending: false,
        validatePending: false,
      },
    })

    expect(wrapper.find('.studio-topbar__title-row').exists()).toBe(true)
    expect(wrapper.find('.studio-topbar__meta-row').exists()).toBe(true)
    expect(wrapper.text()).toContain('角色工坊 2.0')
    expect(wrapper.text()).toContain('当前书籍：五等分')
    expect(wrapper.text()).toContain('当前角色：上杉风太郎')
    expect(wrapper.text()).toContain('分析生成')
    expect(wrapper.text()).toContain('正在保存角色文档')
    expect(wrapper.text()).not.toContain('编辑区优先')
    expect(wrapper.text()).not.toContain('漫画分析 / 角色工坊')
    expect(wrapper.findAllComponents(ProductHeaderAction)).toHaveLength(5)
    expect(wrapper.find('.action-primary').exists()).toBe(false)
    expect(wrapper.find('.action-ghost').exists()).toBe(false)
  })

  it('keeps topbar structure on namespaced owner classes', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/StudioTopbar.vue'),
      'utf8',
    )

    for (const genericClass of [
      'topbar-left',
      'topbar-right',
      'title-block',
      'title-row',
      'meta-row',
      'status-pill',
      'busy-pill',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${genericClass}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${genericClass}\\b`))
    }

    for (const ownerClass of [
      'studio-topbar__left',
      'studio-topbar__right',
      'studio-topbar__title-block',
      'studio-topbar__title-row',
      'studio-topbar__title',
      'studio-topbar__meta-row',
      'studio-topbar__status-pill',
      'studio-topbar__status-pill--busy',
    ]) {
      expect(source).toContain(ownerClass)
    }

    expect(source).not.toContain('.studio-topbar__title-row h1')
  })

  it('maps topbar owner colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/StudioTopbar.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(source).toContain('--studio-topbar-backdrop-background: color-mix')
    expect(source).toContain('--product-header-action-context-solid-surface')
    expect(source).not.toMatch(/--product-header-action-(background|border|text|hover-background|hover-border|hover-text|solid-background|solid-shadow|solid-text)\b/)
  })

  it('wraps topbar groups at the compact breakpoint', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/StudioTopbar.vue'),
      'utf8',
    )
    const compactRules = (source.match(/@media \(--breakpoint-lg-down\) \{([\s\S]*)<\/style>/)?.[1] ?? '')
      .replace(/\r\n/g, '\n')

    expect(cssRule(compactRules, '.studio-topbar')).toContain('flex-wrap: wrap')
    expect(cssRule(compactRules, '.studio-topbar__left,\n  .studio-topbar__right')).toContain('min-width: 0')
    expect(compactRules).toContain('.studio-topbar__right {\n    justify-content: flex-start;')
  })
})
