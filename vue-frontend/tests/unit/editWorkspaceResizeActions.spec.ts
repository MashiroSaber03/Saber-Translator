import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineComponent, h, ref } from 'vue'
import { describe, expect, it } from 'vitest'

import { useEditWorkspaceResizeActions } from '@/composables/edit/useEditWorkspaceResizeActions'

describe('useEditWorkspaceResizeActions', () => {
  it('uses explicit panel refs instead of querying layout classes', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/edit/useEditWorkspaceResizeActions.ts'),
      'utf8',
    )

    expect(source).not.toContain("querySelector('.original-panel')")
    expect(source).not.toContain("querySelector('.translated-panel')")
  })

  it('restores document body resize styles when the owner unmounts during divider drag', () => {
    let actions: ReturnType<typeof useEditWorkspaceResizeActions> | null = null
    const originalPanel = document.createElement('div')
    const translatedPanel = document.createElement('div')
    const editPanel = document.createElement('div')

    const Host = defineComponent({
      setup() {
        actions = useEditWorkspaceResizeActions({
          layoutMode: ref('horizontal'),
          originalPanelRef: ref(originalPanel),
          translatedPanelRef: ref(translatedPanel),
          editPanelRef: ref(editPanel),
        })
        return () => h('div')
      },
    })

    const wrapper = mount(Host)
    actions?.startDividerDrag(new MouseEvent('mousedown', {
      clientX: 32,
      button: 0,
      bubbles: true,
      cancelable: true,
    }))

    expect(document.body.style.cursor).toBe('col-resize')
    expect(document.body.style.userSelect).toBe('none')

    wrapper.unmount()

    expect(document.body.style.cursor).toBe('')
    expect(document.body.style.userSelect).toBe('')
  })
})
