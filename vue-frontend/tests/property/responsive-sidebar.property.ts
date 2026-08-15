import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import SidebarLayout from '@/components/ui/SidebarLayout.vue'

const collapsedArb = fc.constantFrom('none' as const, 'left' as const, 'right' as const, 'both' as const)
const scrollModeArb = fc.constantFrom('page' as const, 'main' as const, 'panes' as const)
const sidebarsArb = fc.constantFrom('flow' as const, 'sticky' as const, 'fixed' as const, 'overlay' as const)
const mobileModeArb = fc.constantFrom('stack' as const, 'drawer' as const)

describe('responsive sidebar shell contracts', () => {
  it('routes layout options through the shared SidebarLayout class contract', () => {
    fc.assert(
      fc.property(
        collapsedArb,
        scrollModeArb,
        sidebarsArb,
        mobileModeArb,
        fc.boolean(),
        (collapsed, scrollMode, sidebars, mobileMode, paneScroll) => {
          const wrapper = mount(SidebarLayout, {
            props: {
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

})
