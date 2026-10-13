import { shallowMount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import EditWorkspace from '@/components/edit/EditWorkspace.vue'
import EditToolbar from '@/components/edit/EditToolbar.vue'
import EditImageComparison from '@/components/edit/EditImageComparison.vue'
import * as workspaceModule from '@/components/edit/useEditWorkspace'

const hooks = vi.hoisted(() => ({ leave: vi.fn(), update: vi.fn(), toast: vi.fn() }))
vi.mock('vue-router', () => ({ onBeforeRouteLeave: hooks.leave, onBeforeRouteUpdate: hooks.update }))
vi.mock('@/utils/toast', async importOriginal => ({
  ...await importOriginal<typeof import('@/utils/toast')>(), showToast: hooks.toast,
}))

let workspace: ReturnType<typeof workspaceModule.useEditWorkspace>
let wrapper: ReturnType<typeof shallowMount>
beforeEach(() => {
  vi.clearAllMocks()
  setActivePinia(createPinia())
  const original = workspaceModule.useEditWorkspace
  vi.spyOn(workspaceModule, 'useEditWorkspace').mockImplementation(emit => {
    workspace = original(emit)
    return workspace
  })
  wrapper = shallowMount(EditWorkspace)
})
afterEach(() => { wrapper?.unmount(); vi.restoreAllMocks() })

describe('unsaved repair range navigation', () => {
  it('blocks route exit and chapter changes when mask saving failed', () => {
    workspace.maskEditor.active.value = true
    workspace.maskEditor.dirty.value = true
    workspace.maskEditor.error.value = 'disk full'
    for (const hook of [hooks.leave, hooks.update]) {
      expect(hook.mock.calls[0]![0]()).toBe(false)
    }
    expect(workspace.maskEditor.dirty.value).toBe(true)
    expect(hooks.toast).toHaveBeenCalledWith('请先保存当前修复范围，或放弃未保存的修改', 'warning')
  })

  it('allows navigation after saving without closing the active mode prematurely', () => {
    workspace.maskEditor.active.value = true
    for (const hook of [hooks.leave, hooks.update]) expect(hook.mock.calls[0]![0]()).toBe(true)
    expect(workspace.maskEditor.active.value).toBe(true)
    expect(hooks.toast).not.toHaveBeenCalled()
  })
})

describe('shared editor comparison state', () => {
  it('returns keyboard focus from an input to either canvas without stealing input focus', () => {
    const root = wrapper.element as HTMLElement
    document.body.appendChild(root)
    const input = document.createElement('input')
    const original = document.createElement('div')
    const translated = document.createElement('div')
    root.append(input, original, translated)
    workspace.imageComparisonRef.value = {
      originalViewportRef: original,
      translatedViewportRef: translated,
    } as NonNullable<typeof workspace.imageComparisonRef.value>
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      callback(0)
      return 0
    })
    for (const canvas of [original, translated]) {
      input.focus()
      input.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }))
      expect(document.activeElement).toBe(input)
      canvas.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }))
      expect(document.activeElement).toBe(root)
    }
    root.remove()
  })

  it('retains opacity across tools and view changes, then resets for a new editor session', async () => {
    const toolbar = wrapper.getComponent(EditToolbar)
    const comparison = wrapper.getComponent(EditImageComparison)
    expect(toolbar.props('originalTextOpacity')).toBe(0)
    expect(toolbar.props('translatedTextOpacity')).toBe(100)
    toolbar.vm.$emit('update-original-text-opacity', 50)
    toolbar.vm.$emit('update-translated-text-opacity', 25)
    await wrapper.vm.$nextTick()
    workspace.maskEditor.active.value = true
    workspace.viewMode.value = 'translated'
    await wrapper.vm.$nextTick()
    expect(toolbar.props('originalTextOpacity')).toBe(50)
    expect(comparison.props('originalTextOpacity')).toBe(50)
    expect(toolbar.props('translatedTextOpacity')).toBe(25)
    expect(comparison.props('translatedTextOpacity')).toBe(25)
    toolbar.vm.$emit('toggle-clean-preview')
    workspace.maskEditor.active.value = false
    workspace.viewMode.value = 'original'
    await wrapper.vm.$nextTick()
    expect(toolbar.props('canOverlaySource')).toBe(false)
    expect(toolbar.props('canFadeTranslated')).toBe(false)
    expect(comparison.props('originalTextOpacity')).toBe(50)
    expect(comparison.props('translatedTextOpacity')).toBe(25)
    wrapper.unmount()
    wrapper = shallowMount(EditWorkspace)
    expect(wrapper.getComponent(EditToolbar).props('originalTextOpacity')).toBe(0)
    expect(wrapper.getComponent(EditToolbar).props('translatedTextOpacity')).toBe(100)
  })
})
