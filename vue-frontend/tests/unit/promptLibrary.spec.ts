import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, reactive } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const {
  setTranslatePromptModeMock,
  setAiVisionOcrPromptModeMock,
  toastInfoMock,
  listV2PromptsMock,
  createPromptMock,
  updatePromptMock,
  deletePromptMock,
  confirmProductActionMock,
} = vi.hoisted(() => ({
  setTranslatePromptModeMock: vi.fn(),
  setAiVisionOcrPromptModeMock: vi.fn(),
  toastInfoMock: vi.fn(),
  listV2PromptsMock: vi.fn(),
  createPromptMock: vi.fn(),
  updatePromptMock: vi.fn(),
  deletePromptMock: vi.fn(),
  confirmProductActionMock: vi.fn(),
}))

const settingsState = reactive({
  translation: {
    openaiOptions: {
      request: {
        forceJsonOutput: false,
      },
    },
  },
  aiVisionOcr: {
    openaiOptions: {
      request: {
        forceJsonOutput: false,
      },
    },
    promptMode: 'paddleocr_vl' as 'normal' | 'json' | 'paddleocr_vl',
  },
})

vi.mock('@/stores/settings', () => ({
  useSettingsStore: () => ({
    settings: settingsState,
    setTranslatePromptMode: setTranslatePromptModeMock,
    setAiVisionOcrPromptMode: setAiVisionOcrPromptModeMock,
  }),
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: toastInfoMock,
  }),
}))

vi.mock('@/api/v2/settings', () => ({
  listV2Prompts: listV2PromptsMock,
  createV2Prompt: createPromptMock,
  updateV2Prompt: updatePromptMock,
  deleteV2Prompt: deletePromptMock,
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

import PromptLibrary from '@/components/settings/PromptLibrary.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiField from '@/components/ui/UiField.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

function prompt(
  id: string,
  name: string,
  content = '提示词内容',
  isFactoryDefault = false,
) {
  return {
    id,
    name,
    content,
    type: 'translate',
    revision: 1,
    isFactoryDefault,
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

describe('PromptLibrary', () => {
  beforeEach(() => {
    setTranslatePromptModeMock.mockReset()
    setAiVisionOcrPromptModeMock.mockReset()
    toastInfoMock.mockReset()
    listV2PromptsMock.mockReset()
    createPromptMock.mockReset()
    updatePromptMock.mockReset()
    deletePromptMock.mockReset()
    confirmProductActionMock.mockReset()
    listV2PromptsMock.mockResolvedValue([])
    deletePromptMock.mockResolvedValue(undefined)
    confirmProductActionMock.mockResolvedValue(true)
    createPromptMock.mockResolvedValue(prompt('created', 'new prompt', 'content'))
    updatePromptMock.mockResolvedValue(prompt('updated', 'updated prompt', 'content'))

    settingsState.translation.openaiOptions.request.forceJsonOutput = false
    settingsState.aiVisionOcr.openaiOptions.request.forceJsonOutput = false
    settingsState.aiVisionOcr.promptMode = 'paddleocr_vl'
  })

  it('uses fixed select primitives for prompt type and mode', async () => {
    const wrapper = mount(PromptLibrary)
    await flushPromises()

    const selects = wrapper.findAllComponents(UiSelect)
    expect(selects).toHaveLength(2)

    expect(selects[0]!.props('modelValue')).toBe('translate')
    expect(selects[0]!.props('options')).toEqual(expect.arrayContaining([
      expect.objectContaining({ value: 'translate' }),
      expect.objectContaining({ value: 'textbox' }),
      expect.objectContaining({ value: 'proofreading' }),
    ]))

    expect(selects[1]!.props('modelValue')).toBe('normal')
    expect(selects[1]!.props('options')).toEqual([
      { label: '普通模式', value: 'normal' },
      { label: 'JSON格式模式', value: 'json' },
    ])

    selects[1]!.vm.$emit('change', 'json')

    expect(setTranslatePromptModeMock).toHaveBeenLastCalledWith(true)
    expect(toastInfoMock).toHaveBeenCalledWith('已切换到JSON格式模式')
  })

  it('preserves ai vision paddleocr_vl prompt mode instead of collapsing it to normal/json', async () => {
    const wrapper = mount(PromptLibrary)
    await flushPromises()

    const selects = wrapper.findAllComponents(UiSelect)
    expect(selects.length).toBeGreaterThanOrEqual(1)

    const typeSelect = selects[0]!
    typeSelect.vm.$emit('change', 'ai_vision_ocr')
    await flushPromises()

    const refreshedSelects = wrapper.findAllComponents(UiSelect)
    expect(refreshedSelects.length).toBeGreaterThanOrEqual(2)

    const modeSelect = refreshedSelects[1]!
    expect(modeSelect.props('modelValue')).toBe('paddleocr_vl')

    modeSelect.vm.$emit('change', 'paddleocr_vl')

    expect(setAiVisionOcrPromptModeMock).toHaveBeenLastCalledWith('paddleocr_vl')
    expect(toastInfoMock).toHaveBeenCalledWith('已切换到OCR模型提示词模式')
  })

  it('uses separate controls for selecting, loading, and deleting prompt rows', async () => {
    listV2PromptsMock.mockResolvedValue([
      prompt('prompt-default', 'default', '提示词内容', true),
      prompt('prompt-custom', 'custom'),
    ])
    const wrapper = mount(PromptLibrary)
    await flushPromises()

    const promptRow = wrapper.find('.prompt-library__item')
    expect(promptRow.element.tagName).toBe('DIV')

    const selectButton = wrapper.find('.prompt-library__select-action')
    expect(selectButton.element.tagName).toBe('BUTTON')
    expect(selectButton.attributes('aria-label')).toBe('选择提示词：default')
    expect(selectButton.attributes('aria-pressed')).toBe('false')

    await selectButton.trigger('click')
    await flushPromises()

    expect(wrapper.getComponent(UiTextarea).props('modelValue')).toBe('提示词内容')
    expect(wrapper.find('.prompt-library__select-action').attributes('aria-pressed')).toBe('true')

    const loadButton = wrapper.find('.prompt-library__load-action')
    expect(loadButton.element.tagName).toBe('BUTTON')
    expect(loadButton.attributes('aria-label')).toBe('加载提示词：default')
    expect(loadButton.getComponent(UiIconButton).props()).toMatchObject({
      label: '加载提示词：default',
      variant: 'soft',
      size: 'sm',
    })

    const deleteButton = wrapper.find('.prompt-library__delete-action')
    expect(deleteButton.element.tagName).toBe('BUTTON')
    expect(deleteButton.attributes('aria-label')).toBe('删除提示词：default')
    expect(deleteButton.getComponent(UiIconButton).props()).toMatchObject({
      label: '删除提示词：default',
      variant: 'danger',
      size: 'sm',
    })
    expect(promptRow.findAllComponents(UiIconButton)).toHaveLength(2)
  })

  it('uses icon-button primitives instead of root button skins for prompt row actions', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PromptLibrary.vue'), 'utf8')
    const rootStyle = source.match(/\.prompt-library \{(?<body>[\s\S]*?)\n\}/)

    expect(source).toContain("import UiIconButton from '@/components/ui/UiIconButton.vue'")
    expect(source).toContain('class="prompt-library__list"')
    expect(source).toContain('class="prompt-library__item"')
    expect(source).toContain('class="prompt-library__select-action"')
    expect(source).toContain('class="prompt-library__name"')
    expect(source).toContain('class="prompt-library__actions"')
    expect(source).toContain('class="prompt-library__load-action"')
    expect(source).toContain('class="prompt-library__delete-action"')
    expect(source).not.toContain('class="prompt-list"')
    expect(source).not.toContain('class="prompt-item"')
    expect(source).not.toContain('class="prompt-select"')
    expect(source).not.toContain('class="prompt-name"')
    expect(source).not.toContain('class="prompt-actions"')
    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-button-/)
    expect(source).not.toContain('.prompt-library__delete-action:disabled')
  })

  it('routes prompt list loading and empty states through product status feedback', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PromptLibrary.vue'), 'utf8')
    expect(source).not.toContain('loading-hint')
    expect(source).not.toContain('empty-hint')
    expect(source).toContain('ProductStatusBanner')

    const pendingPrompts = deferred<ReturnType<typeof prompt>[]>()
    listV2PromptsMock.mockReturnValueOnce(pendingPrompts.promise)

    const wrapper = mount(PromptLibrary)
    await nextTick()

    const loadingBanner = wrapper.findComponent(ProductStatusBanner)
    expect(loadingBanner.exists()).toBe(true)
    expect(loadingBanner.props()).toMatchObject({
      ariaLive: 'polite',
      iconName: 'refresh',
      title: '正在加载提示词',
      tone: 'neutral',
    })
    expect(wrapper.find('.loading-hint').exists()).toBe(false)

    pendingPrompts.resolve([])
    await flushPromises()

    const emptyBanner = wrapper.findComponent(ProductStatusBanner)
    expect(emptyBanner.exists()).toBe(true)
    expect(emptyBanner.props()).toMatchObject({
      iconName: 'file-text',
      title: '暂无保存的提示词',
      tone: 'neutral',
    })
    expect(wrapper.find('.empty-hint').exists()).toBe(false)
  })

  it('routes prompt library labels and save actions through typed settings primitives', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PromptLibrary.vue'), 'utf8')

    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('prompt-editor-actions')
    expect(source).toContain('ProductActionRow')

    const wrapper = mount(PromptLibrary)
    await flushPromises()

    const fields = wrapper.findAllComponents(UiField)
    const fieldByControlId = (controlId: string) =>
      fields.find(field => field.props('controlId') === controlId)

    expect(fieldByControlId('promptType')?.props('label')).toBe('提示词类型')
    expect(fieldByControlId('promptMode')?.props('label')).toBe('提示词模式')
    expect(fieldByControlId('promptMode')?.props('hint')).toBe('适用于普通翻译场景')
    expect(fieldByControlId('promptName')?.props('label')).toBe('提示词名称')
    expect(fieldByControlId('promptContent')?.props('label')).toBe('提示词内容')
    expect(wrapper.getComponent(UiTextarea).props('variant')).toBe('panel')
  })

  it('ignores stale prompt list responses after the prompt type changes', async () => {
    const translatePrompts = deferred<ReturnType<typeof prompt>[]>()
    const visionPrompts = deferred<ReturnType<typeof prompt>[]>()
    listV2PromptsMock.mockImplementation((type?: string) =>
      type === 'ai_vision_ocr' ? visionPrompts.promise : translatePrompts.promise
    )

    const wrapper = mount(PromptLibrary)

    const typeSelect = wrapper.findAllComponents(UiSelect)[0]!
    typeSelect.vm.$emit('change', 'ai_vision_ocr')

    visionPrompts.resolve([prompt('vision-current', 'vision-current')])
    await flushPromises()
    expect(wrapper.text()).toContain('vision-current')

    translatePrompts.resolve([prompt('translate-stale', 'translate-stale')])
    await flushPromises()
    expect(wrapper.text()).toContain('vision-current')
    expect(wrapper.text()).not.toContain('translate-stale')
  })

  it('clears the previous type list when the next prompt list fails to load', async () => {
    listV2PromptsMock
      .mockResolvedValueOnce([prompt('translate-old', 'translate-old')])
      .mockRejectedValueOnce(new Error('load failed'))

    const wrapper = mount(PromptLibrary)
    await flushPromises()
    expect(wrapper.text()).toContain('translate-old')

    wrapper.findAllComponents(UiSelect)[0]!.vm.$emit('change', 'textbox')
    await flushPromises()

    expect(wrapper.text()).not.toContain('translate-old')
    expect(wrapper.getComponent(ProductStatusBanner).props('title')).toBe('提示词加载失败')
    expect(wrapper.text()).toContain('load failed')
  })

  it('lets the user retry after the prompt list fails', async () => {
    listV2PromptsMock
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce([prompt('retry-id', 'retry prompt')])

    const wrapper = mount(PromptLibrary)
    await flushPromises()
    expect(wrapper.getComponent(ProductStatusBanner).props('title')).toBe('提示词加载失败')

    await wrapper.getComponent(ProductStatusBanner).get('button').trigger('click')
    await flushPromises()

    expect(listV2PromptsMock).toHaveBeenCalledTimes(2)
    expect(wrapper.text()).toContain('retry prompt')
  })

  it('confirms before deleting custom prompts', async () => {
    listV2PromptsMock.mockResolvedValue([
      prompt('prompt-custom', 'custom'),
    ])
    confirmProductActionMock.mockResolvedValueOnce(false)

    const wrapper = mount(PromptLibrary)
    await flushPromises()

    const deleteButton = wrapper.find('button[aria-label="删除提示词：custom"]')
    await deleteButton.trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '删除提示词',
      message: '确定要删除提示词“custom”吗？此操作无法撤销。',
      confirmText: '删除',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(deletePromptMock).not.toHaveBeenCalled()

    confirmProductActionMock.mockResolvedValueOnce(true)
    await deleteButton.trigger('click')
    await flushPromises()

    expect(deletePromptMock).toHaveBeenCalledWith('prompt-custom')
  })

  it('serializes prompt mutations so a double click cannot submit twice', async () => {
    const pendingCreate = deferred<ReturnType<typeof prompt>>()
    createPromptMock.mockReturnValueOnce(pendingCreate.promise)
    const wrapper = mount(PromptLibrary)
    await flushPromises()
    await wrapper.get<HTMLInputElement>('#promptName').setValue('new prompt')
    await wrapper.get<HTMLTextAreaElement>('#promptContent').setValue('content')

    const saveButton = wrapper.findAll('button')
      .find(button => button.text().includes('保存提示词'))
    expect(saveButton).toBeTruthy()
    await saveButton!.trigger('click')
    await saveButton!.trigger('click')

    expect(createPromptMock).toHaveBeenCalledTimes(1)
    expect(wrapper.findAllComponents(UiSelect).every(select => select.props('disabled'))).toBe(true)

    pendingCreate.resolve(prompt('created-id', 'new prompt', 'content'))
    await flushPromises()
  })

  it('uses mutation responses directly instead of refetching the whole list', async () => {
    const created = prompt('created-id', 'created prompt', '')
    createPromptMock.mockResolvedValueOnce(created)
    const wrapper = mount(PromptLibrary)
    await flushPromises()
    await wrapper.get<HTMLInputElement>('#promptName').setValue('created prompt')

    const saveButton = wrapper.findAll('button')
      .find(button => button.text().includes('保存提示词'))
    await saveButton!.trigger('click')
    await flushPromises()

    expect(createPromptMock).toHaveBeenCalledWith('translate', 'created prompt', '')
    expect(listV2PromptsMock).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('created prompt')

    await wrapper.get('button[aria-label="选择提示词：created prompt"]').trigger('click')
    await wrapper.get<HTMLInputElement>('#promptName').setValue('renamed prompt')
    updatePromptMock.mockResolvedValueOnce({ ...created, name: 'renamed prompt', revision: 2 })
    await saveButton!.trigger('click')
    await flushPromises()

    expect(listV2PromptsMock).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('renamed prompt')
    expect(wrapper.text()).not.toContain('created prompt')
  })
})
