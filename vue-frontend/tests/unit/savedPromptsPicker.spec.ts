import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import ProductChipList from '@/components/product/ProductChipList.vue'

const {
  listV2PromptsMock,
} = vi.hoisted(() => ({
  listV2PromptsMock: vi.fn(),
}))

vi.mock('@/api/v2/settings', () => ({
  listV2Prompts: listV2PromptsMock,
}))

import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'

function prompt(id: string, name: string, content: string) {
  return {
    id,
    name,
    content,
    type: 'translate',
    revision: 1,
    isFactoryDefault: false,
  }
}

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
    listV2PromptsMock.mockReset()
    listV2PromptsMock.mockResolvedValue([])
  })

  it('ignores stale prompt list responses after prompt type changes', async () => {
    const translatePrompts = deferred<ReturnType<typeof prompt>[]>()
    const textboxPrompts = deferred<ReturnType<typeof prompt>[]>()
    listV2PromptsMock.mockImplementation((type: string) =>
      type === 'textbox' ? textboxPrompts.promise : translatePrompts.promise
    )

    const wrapper = mount(SavedPromptsPicker, {
      props: { promptType: 'translate' },
    })

    await wrapper.setProps({ promptType: 'textbox' })
    textboxPrompts.resolve([
      prompt('textbox-current-id', 'textbox-current', 'textbox content'),
    ])
    await flushPromises()
    expect(wrapper.text()).toContain('textbox-current')

    translatePrompts.resolve([
      prompt('translate-stale-id', 'translate-stale', 'stale content'),
    ])
    await flushPromises()
    expect(wrapper.text()).toContain('textbox-current')
    expect(wrapper.text()).not.toContain('translate-stale')
  })

  it('does not emit a prompt that no longer belongs to the active type', async () => {
    listV2PromptsMock.mockImplementation((type: string) => Promise.resolve(
      type === 'translate'
        ? [prompt('translate-prompt-id', 'translate-prompt', 'prompt content')]
        : [],
    ))

    const wrapper = mount(SavedPromptsPicker, {
      props: { promptType: 'translate' },
    })
    await flushPromises()

    const chips = wrapper.getComponent(ProductChipList)
    expect(chips.props('label')).toBe('快速选择')
    expect(chips.props('items')).toEqual([
      expect.objectContaining({
        id: 'translate-prompt-id',
        interactive: true,
        label: 'translate-prompt',
      }),
    ])

    await wrapper.setProps({ promptType: 'textbox' })
    await flushPromises()
    chips.vm.$emit('select', 'translate-prompt-id')
    await flushPromises()

    expect(wrapper.emitted('select')).toBeUndefined()
  })

  it('renders loading and empty prompt states as product chip status items', async () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/SavedPromptsPicker.vue'),
      'utf8'
    )
    expect(source).not.toContain('empty-hint')

    const pendingPrompts = deferred<ReturnType<typeof prompt>[]>()
    listV2PromptsMock.mockReturnValueOnce(pendingPrompts.promise)

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

    pendingPrompts.resolve([])
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
