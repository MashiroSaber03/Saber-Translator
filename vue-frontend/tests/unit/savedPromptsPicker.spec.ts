import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import ProductChipList from '@/components/product/ProductChipList.vue'

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

    const chips = wrapper.getComponent(ProductChipList)
    expect(chips.props('label')).toBe('快速选择')
    expect(chips.props('items')).toEqual([
      expect.objectContaining({
        id: 'translate-prompt',
        interactive: true,
        label: 'translate-prompt',
      }),
    ])

    chips.vm.$emit('select', 'translate-prompt')
    await wrapper.setProps({ promptType: 'textbox' })
    translateContent.resolve({ prompt_content: 'stale translate content' })
    await flushPromises()

    expect(wrapper.emitted('select')).toBeUndefined()
  })

  it('renders loading and empty prompt states as product chip status items', async () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/SavedPromptsPicker.vue'),
      'utf8'
    )
    expect(source).not.toContain('empty-hint')

    const pendingPrompts = deferred<{ prompt_names: string[] }>()
    getPromptsMock.mockReturnValueOnce(pendingPrompts.promise)

    const wrapper = mount(SavedPromptsPicker, {
      props: { promptType: 'translate' },
    })
    await nextTick()

    let chipList = wrapper.getComponent(ProductChipList)
    expect(chipList.props('items')).toEqual([
      expect.objectContaining({
        iconName: 'refresh',
        id: 'loading',
        interactive: false,
        label: '加载中...',
        tone: 'neutral',
      }),
    ])
    expect(wrapper.find('.empty-hint').exists()).toBe(false)

    pendingPrompts.resolve({ prompt_names: [] })
    await flushPromises()

    chipList = wrapper.getComponent(ProductChipList)
    expect(chipList.props('items')).toEqual([
      expect.objectContaining({
        iconName: 'file-text',
        id: 'empty',
        interactive: false,
        label: '暂无保存的提示词',
        tone: 'neutral',
      }),
    ])
    expect(wrapper.find('.empty-hint').exists()).toBe(false)
  })

  it('keeps refresh behavior on lifecycle and props instead of an exposed instance method', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/SavedPromptsPicker.vue'),
      'utf8'
    )

    expect(source).not.toContain('defineExpose')
    expect(source).toContain('watch(() => props.promptType')
    expect(source).toContain('onMounted')
    expect(source).not.toMatch(/var\(--color-[a-z0-9-]+,\s*var\(--color-[a-z0-9-]+\)\)/)
  })
})
