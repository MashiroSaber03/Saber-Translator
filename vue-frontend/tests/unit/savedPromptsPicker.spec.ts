import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  getPromptsMock,
  getPromptContentMock,
  getTextboxPromptsMock,
  getTextboxPromptContentMock,
} = vi.hoisted(() => ({
  getPromptsMock: vi.fn(),
  getPromptContentMock: vi.fn(),
  getTextboxPromptsMock: vi.fn(),
  getTextboxPromptContentMock: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  configApi: {
    getPrompts: getPromptsMock,
    getPromptContent: getPromptContentMock,
    getTextboxPrompts: getTextboxPromptsMock,
    getTextboxPromptContent: getTextboxPromptContentMock,
  },
}))

import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve
    reject = promiseReject
  })
  return { promise, resolve, reject }
}

describe('SavedPromptsPicker', () => {
  beforeEach(() => {
    getPromptsMock.mockReset()
    getPromptContentMock.mockReset()
    getTextboxPromptsMock.mockReset()
    getTextboxPromptContentMock.mockReset()
    getPromptsMock.mockResolvedValue({ prompt_names: [] })
    getPromptContentMock.mockResolvedValue({ prompt_content: 'prompt content' })
    getTextboxPromptsMock.mockResolvedValue({ prompt_names: [] })
    getTextboxPromptContentMock.mockResolvedValue({ prompt_content: 'textbox content' })
  })

  it('ignores stale prompt list responses after prompt type changes', async () => {
    const translatePrompts = deferred<{ prompt_names: string[] }>()
    const textboxPrompts = deferred<{ prompt_names: string[] }>()
    getPromptsMock.mockReturnValue(translatePrompts.promise)
    getTextboxPromptsMock.mockReturnValue(textboxPrompts.promise)

    const wrapper = mount(SavedPromptsPicker, {
      props: { promptType: 'translate' },
    })

    await wrapper.setProps({ promptType: 'textbox' })
    textboxPrompts.resolve({ prompt_names: ['textbox-current'] })
    await flushPromises()
    expect(wrapper.text()).toContain('textbox-current')

    translatePrompts.resolve({ prompt_names: ['translate-stale'] })
    await flushPromises()
    expect(wrapper.text()).toContain('textbox-current')
    expect(wrapper.text()).not.toContain('translate-stale')
  })

  it('does not emit stale prompt content after prompt type changes', async () => {
    getPromptsMock.mockResolvedValue({ prompt_names: ['translate-prompt'] })
    const translateContent = deferred<{ prompt_content: string }>()
    getPromptContentMock.mockReturnValue(translateContent.promise)

    const wrapper = mount(SavedPromptsPicker, {
      props: { promptType: 'translate' },
    })
    await flushPromises()

    await wrapper.get('.prompt-chip').trigger('click')
    await wrapper.setProps({ promptType: 'textbox' })
    translateContent.resolve({ prompt_content: 'stale translate content' })
    await flushPromises()

    expect(wrapper.emitted('select')).toBeUndefined()
  })
})
