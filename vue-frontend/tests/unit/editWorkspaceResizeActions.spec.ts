import { mount } from '@vue/test-utils'
import { defineComponent, h, ref } from 'vue'
import { describe, expect, it } from 'vitest'

import { useEditWorkspaceResizeActions } from '@/composables/edit/useEditWorkspaceResizeActions'

describe('useEditWorkspaceResizeActions', () => {
  it('restores document body resize styles when the owner unmounts during divider drag', () => {
    let actions: ReturnType<typeof useEditWorkspaceResizeActions> | null = null
    const originalViewport = document.createElement('div')
    const editPanel = document.createElement('div')

    const Host = defineComponent({
      setup() {
        actions = useEditWorkspaceResizeActions({
          layoutMode: ref('horizontal'),
          originalViewportRef: ref(originalViewport),
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
