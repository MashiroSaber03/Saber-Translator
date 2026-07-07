import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import SidebarLayout from '@/components/ui/SidebarLayout.vue'

const modeArb = fc.constantFrom('flow' as const, 'fixed' as const, 'overlay' as const)
const collapsedArb = fc.constantFrom('none' as const, 'left' as const, 'right' as const, 'both' as const)
const scrollModeArb = fc.constantFrom('page' as const, 'main' as const, 'panes' as const)
const sidebarsArb = fc.constantFrom('flow' as const, 'sticky' as const, 'fixed' as const, 'overlay' as const)
const mobileModeArb = fc.constantFrom('stack' as const, 'drawer' as const)

describe('responsive sidebar shell contracts', () => {
  it('routes layout options through the shared SidebarLayout class contract', () => {
    fc.assert(
      fc.property(
        modeArb,
        collapsedArb,
        scrollModeArb,
        sidebarsArb,
        mobileModeArb,
        fc.boolean(),
        (mode, collapsed, scrollMode, sidebars, mobileMode, paneScroll) => {
          const wrapper = mount(SidebarLayout, {
            props: {
              mode,
              collapsed,
              scrollMode,
              sidebars,
              mobileMode,
              paneScroll,
            },
            slots: {
              left: '<aside>left</aside>',
              default: '<main>main</main>',
              right: '<aside>right</aside>',
            },
          })

          const layout = wrapper.get('.ui-sidebar-layout')
          expect(layout.classes()).toContain(`ui-sidebar-layout--${mode}`)
          expect(layout.classes()).toContain(`ui-sidebar-layout--${collapsed}-collapsed`)
          expect(layout.classes()).toContain(`ui-sidebar-layout--scroll-${scrollMode}`)
          expect(layout.classes()).toContain(`ui-sidebar-layout--sidebars-${sidebars}`)
          expect(layout.classes()).toContain(`ui-sidebar-layout--mobile-${mobileMode}`)
          expect(layout.classes().includes('ui-sidebar-layout--pane-scroll')).toBe(paneScroll)

          wrapper.unmount()
        },
      ),
      { numRuns: 100 },
    )
  })

  it('routes responsive sizing through SidebarLayout CSS variables', () => {
    const wrapper = mount(SidebarLayout, {
      props: {
        leftWidth: '320px',
        rightWidth: '240px',
        gap: '16px',
        height: 'calc(100dvh - 80px)',
        sidebarTop: '80px',
        leftInset: '320px',
        rightInset: '240px',
        contentInset: '20px',
      },
      slots: {
        left: '<aside>left</aside>',
        default: '<main>main</main>',
        right: '<aside>right</aside>',
      },
    })

    const style = wrapper.get('.ui-sidebar-layout').attributes('style') ?? ''

    for (const token of [
      '--ui-sidebar-left-width: 320px;',
      '--ui-sidebar-right-width: 240px;',
      '--ui-sidebar-gap: 16px;',
      '--ui-sidebar-height: calc(100dvh - 80px);',
      '--ui-sidebar-top: 80px;',
      '--ui-sidebar-left-inset: 320px;',
      '--ui-sidebar-right-inset: 240px;',
      '--ui-sidebar-content-inset: 20px;',
    ]) {
      expect(style).toContain(token)
    }
  })

  it('does not expose a dead custom mobile breakpoint API', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/ui/SidebarLayout.vue'), 'utf8')

    expect(source).not.toContain('mobileBreakpoint')
    expect(source).not.toContain('--ui-sidebar-mobile-breakpoint')
  })
})
