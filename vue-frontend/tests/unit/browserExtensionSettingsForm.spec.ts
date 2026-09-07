import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import BrowserExtensionSettingsForm from '@/components/settings/BrowserExtensionSettingsForm.vue'
import TextStyleForm from '@/components/settings/TextStyleForm.vue'
import { getTextStyleDefaults } from '@/defaults/textStyleDefaults'
import type { components } from '@/api/generated/v2'
import type { PluginSettingsApi } from '@/types/browserExtensionSettings'

type Schema = components['schemas']
enableAutoUnmount(afterEach)
beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

async function setup() {
  const settings: Schema['SettingsDocument'] = {
    settings: [
      { domain: 'text_style_defaults', revision: 1, schemaVersion: 2, payload: { ...getTextStyleDefaults(), fontFamily: 'font' } },
      { domain: 'browser_dom_agent', revision: 0, schemaVersion: 1, payload: { provider: 'ollama', modelName: 'agent', customBaseUrl: '', openaiOptions: {} } },
    ],
    providerSettings: [], credentials: [], bookSettings: [],
  }
  const put = vi.fn(async (transaction: Schema['SettingsTransaction']): Promise<Schema['SettingsTransactionResult']> => ({
    settings: transaction.settings!.map(row => ({ domain: row.domain, revision: row.baseRevision + 1 })),
    providerSettings: [], credentials: [], bookSettings: [], prompts: [],
  }))
  const api = vi.fn(async (path: string, method?: string, body?: unknown) => {
    if (method === 'PUT') return put(body as Schema['SettingsTransaction'])
    if (path === '/fonts') return { items: [{ id: 'font', displayName: '字体' }] }
    return structuredClone(settings)
  })
  const wrapper = mount(BrowserExtensionSettingsForm, {
    props: { api: api as PluginSettingsApi },
    global: { stubs: { TextStyleForm: true } },
  })
  await flushPromises()
  const edit = async (strokeWidth: number) => {
    wrapper.findComponent(TextStyleForm).vm.$emit('change', { strokeWidth })
    await nextTick()
  }
  return { wrapper, put, api, edit }
}

describe('plugin settings autosave', () => {
  it('does not save on load; debounces edits without reloading the form', async () => {
    const { wrapper, put, api, edit } = await setup()
    await vi.advanceTimersByTimeAsync(500)
    expect(put).not.toHaveBeenCalled()
    expect(wrapper.text()).not.toContain('保存插件配置')
    expect(wrapper.text()).not.toContain('重新读取')
    await edit(1.2)
    await vi.advanceTimersByTimeAsync(300)
    await edit(1.3)
    await vi.advanceTimersByTimeAsync(450)
    expect(put).toHaveBeenCalledTimes(1)
    expect(put.mock.calls[0]![0].settings![0]).toMatchObject({ baseRevision: 1, payload: { strokeWidth: 1.3 } })
    expect(api.mock.calls.filter(([, method]) => method !== 'PUT')).toHaveLength(2)
    expect(wrapper.findComponent(TextStyleForm).props('modelValue').strokeWidth).toBe(1.3)
  })

  it('serializes edits during an in-flight request and flushes them before closing', async () => {
    const { wrapper, put, edit } = await setup()
    let finish!: (value: Schema['SettingsTransactionResult']) => void
    put.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
    await edit(1.2)
    await vi.advanceTimersByTimeAsync(450)
    await edit(2.3)
    await vi.advanceTimersByTimeAsync(500)
    expect(put).toHaveBeenCalledTimes(1)
    const closed = wrapper.vm.save()
    finish({ settings: [{ domain: 'text_style_defaults', revision: 2 }], providerSettings: [], credentials: [], bookSettings: [], prompts: [] })
    expect(await closed).toBe(true)
    expect(put).toHaveBeenCalledTimes(2)
    expect(put.mock.calls[0]![0].settings![0]!.payload.strokeWidth).toBe(1.2)
    expect(put.mock.calls[1]![0].settings![0]).toMatchObject({ baseRevision: 2, payload: { strokeWidth: 2.3 } })
    await vi.advanceTimersByTimeAsync(1000)
    expect(put).toHaveBeenCalledTimes(2)
  })

  it('keeps failed changes for explicit retry without repeatedly sending failed requests', async () => {
    const { wrapper, put, edit } = await setup()
    put.mockRejectedValueOnce(new Error('网络断开'))
    await edit(2.3)
    await vi.advanceTimersByTimeAsync(450)
    expect(wrapper.text()).toContain('自动保存失败，修改尚未保存。网络断开')
    expect(wrapper.findComponent(TextStyleForm).props('modelValue').strokeWidth).toBe(2.3)
    await vi.advanceTimersByTimeAsync(5000)
    expect(put).toHaveBeenCalledTimes(1)
    await wrapper.findAll('button').find(button => button.text() === '重试保存')!.trigger('click')
    await flushPromises()
    expect(put).toHaveBeenCalledTimes(2)
    expect(wrapper.text()).not.toContain('重试保存')
  })

  it('flushes pending edits immediately and retains the dialog on failure', async () => {
    const { wrapper, put, edit } = await setup()
    put.mockRejectedValueOnce(new Error('版本冲突'))
    await edit(0.8)
    expect(await wrapper.vm.save()).toBe(false)
    expect(put).toHaveBeenCalledTimes(1)
    expect(await wrapper.vm.save()).toBe(true)
    expect(put.mock.calls[1]![0].settings![0]!.baseRevision).toBe(1)
  })

  it('flushes pending changes when the component is removed', async () => {
    const { wrapper, put, edit } = await setup()
    await edit(0.9)
    wrapper.unmount()
    await flushPromises()
    expect(put).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1000)
    expect(put).toHaveBeenCalledTimes(1)
  })
})
