import { shallowMount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import EditWorkspace from '@/components/edit/EditWorkspace.vue'
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
